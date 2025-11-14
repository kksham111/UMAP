import os

import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from datasets.utils.loader import make_distributed_data_loader
from experiments.utils import set_seed, init_wandb, init_distributed, is_main_process


def process_batch(batch, obs_horizon, action_horizon, device):
    # Take the first `obs_horizon` observations
    obs = {k: v[:, :obs_horizon].to(device) for k, v in batch["obs"].items()}

    # Take the last `action_horizon` actions
    action = batch["action"][:, -action_horizon:].to(device)

    # Add language tokens to observations
    if "input_ids" in batch and "attention_mask" in batch:
        obs["input_ids"] = batch["input_ids"].to(device)
        obs["attention_mask"] = batch["attention_mask"].to(device)
    return obs, action


def eval_one_epoch(config, data_loader, device, model, action_normalizer=None):
    model.eval()
    model = getattr(model, "module", model)  # unwrap DDP

    # Unnormalize actions
    if action_normalizer is not None:
        action_scale = torch.tensor(action_normalizer.scale[None], device=device)
        action_offset = torch.tensor(action_normalizer.offset[None], device=device)
        unnormalize = lambda a: a * action_scale + action_offset
    else:
        unnormalize = lambda a: a

    stats = {"loss": 0, "action_mse": 0}
    for batch in tqdm(data_loader, desc="Evaluating", disable=not is_main_process()):
        # ------------ Preprocess data ------------ #
        obs, action = process_batch(
            batch, config.model.obs_encoder.num_frames, config.model.action_len, device
        )

        with torch.no_grad():
            # ------------ Validation loss ------------ #
            loss = model(obs, action)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            stats["loss"] += loss.item()

            # ------------ BC Inference ------------ #
            # Sample actions
            action_hat = model.sample(obs)

            # Unnormalize action and action_hat
            action_hat = unnormalize(action_hat)
            action = unnormalize(action)

            # Compute MSE loss
            mse = F.mse_loss(action_hat, action)

            # Collect results across all processes
            dist.all_reduce(mse, op=dist.ReduceOp.AVG)
            stats["action_mse"] += mse

    # Average over all batches
    stats = {k: v / len(data_loader) for k, v in stats.items()}
    return stats


def train_one_step(config, model, optimizer, scheduler, scaler, batch, device):
    model.train()

    # --- Preprocess data ---
    obs, action = process_batch(
        batch, config.model.obs_encoder.num_frames, config.model.action_len, device
    )

    # --- DP Training ---
    # Action prediction loss
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=config.use_amp
    ):
        loss = model(obs, action)
        info = {"loss": loss.item(), "action_loss": loss.item()}

    # Step optimizer
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    if config.clip_grad_norm:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    # Step scheduler
    scheduler.step()
    info["lr"] = scheduler.get_last_lr()[0]
    return loss, info


def maybe_resume_checkpoint(
    config, model, optimizer, scheduler, scaler, ckpt_name="models.pt"
):
    """Resume from a checkpoint if config.resume is True."""
    step = 0
    if config.resume:
        ckpt_path = os.path.join(config.logdir, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"] + 1
        print(f"Resumed training from step {step}")
    return step


def maybe_evaluate(config, step, model, loader, device, action_normalizer=None):
    """Evaluate if it's the correct step."""
    if step % config.eval_every == 0 or step == (config.num_steps - 1):
        stats = eval_one_epoch(config, loader, device, model, action_normalizer)
        if is_main_process():
            wandb.log({f"eval/{k}": v for k, v in stats.items()})
            print(f"Step {step} action mse: {stats['action_mse']:.4f}")


def maybe_save_checkpoint(
    config,
    step,
    model,
    optimizer,
    scheduler,
    scaler,
    action_normalizer=None,
    lowdim_normalizer=None,
    ckpt_name="models.pt",
):
    """Save checkpoint on the main process if it's the correct step."""
    if is_main_process() and (
        step % config.save_every == 0 or step == (config.num_steps - 1)
    ):
        ckpt = {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "action_normalizer": action_normalizer,
            "lowdim_normalizer": lowdim_normalizer,
            "step": step,
        }
        ckpt_path = os.path.join(config.logdir, ckpt_name)
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint at step {step} to {ckpt_path}")


def train(rank, world_size, config):
    # Set global seed
    set_seed(config.seed * world_size + rank)

    # Initialize distributed training
    init_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Initialize WANDB
    if is_main_process():
        init_wandb(config, job_type="train")

    # Create dataset
    train_set, val_set = instantiate(config.dataset)
    train_loader, val_loader = make_distributed_data_loader(
        train_set, val_set, config.batch_size, rank, world_size
    )

    # Create model
    model = instantiate(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **config.scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    # Load pretrained model
    if config.pretrain_checkpoint_path:
        ckpt = torch.load(config.pretrain_checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(
            f"Loaded pretraining checkpoint {config.pretrain_checkpoint_path}, step: {ckpt['step']}"
        )

        # Replace dataset normalizers to make sure data is normalized correctly
        if ckpt["action_normalizer"] is not None:
            train_set.action_normalizer = ckpt["action_normalizer"]
            val_set.action_normalizer = ckpt["action_normalizer"]
        if ckpt["lowdim_normalizer"] is not None:
            train_set.lowdim_normalizer = ckpt["lowdim_normalizer"]
            val_set.lowdim_normalizer = ckpt["lowdim_normalizer"]

    # Resume from checkpoint
    step = maybe_resume_checkpoint(config, model, optimizer, scheduler, scaler)
    epoch = step // len(train_loader)

    # Wrap model with DDP
    model = DistributedDataParallel(model, device_ids=[rank], static_graph=True)

    # Training loop
    pbar = tqdm(
        total=config.num_steps,
        initial=step,
        desc="Training",
        disable=not is_main_process(),
    )
    while step < config.num_steps:
        # Set epoch for distributed sampler to shuffle indices
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            # --- Training step ---
            loss, info = train_one_step(
                config, model, optimizer, scheduler, scaler, batch, device
            )

            # --- Logging ---
            if is_main_process():
                pbar.set_description(f"step: {step}, loss: {loss.item():.4f}")
                wandb.log({f"train/{k}": v for k, v in info.items()})

            # --- Evaluate if needed ---
            maybe_evaluate(
                config, step, model, val_loader, device, train_set.action_normalizer
            )

            # --- Save checkpoint if needed ---
            maybe_save_checkpoint(
                config,
                step,
                model,
                optimizer,
                scheduler,
                scaler,
                train_set.action_normalizer,
                train_set.lowdim_normalizer,
            )

            step += 1
            pbar.update(1)
            if step >= config.num_steps:
                break

        epoch += 1


@hydra.main(version_base=None, config_path="../../configs", config_name="train_dp.yaml")
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    # Spawn processes
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
