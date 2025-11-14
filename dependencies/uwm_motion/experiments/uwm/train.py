import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
from diffusers.optimization import get_scheduler
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid
from tqdm import tqdm

from datasets.utils.loader import make_distributed_data_loader
from experiments.utils import set_seed, init_wandb, init_distributed, is_main_process


def process_batch(batch, obs_horizon, action_horizon, device):
    action_start = obs_horizon - 1
    action_end = action_start + action_horizon
    curr_obs = {k: v[:, : action_start + 1].to(device) for k, v in batch["obs"].items()}
    next_obs = {k: v[:, action_end:].to(device) for k, v in batch["obs"].items()}
    actions = batch["action"][:, action_start:action_end].to(device)

    # Add traj3d if present (照搬 action 的处理方式)
    traj3d = None
    if "traj3d" in batch:
        traj3d = batch["traj3d"][:, action_start:action_end].to(device)

    # Add language tokens
    if "input_ids" in batch and "attention_mask" in batch:
        curr_obs["input_ids"] = batch["input_ids"].to(device)
        curr_obs["attention_mask"] = batch["attention_mask"].to(device)
    return curr_obs, next_obs, actions, traj3d


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

    def decode_and_plot(images, nrows=4):
        with torch.no_grad():
            images = model.obs_encoder.apply_vae(images, inverse=True)
            images = rearrange(images, "b v c t h w -> (b v t) c h w")
            images_grid = make_grid(images, nrows)
        return images_grid

    stats = {
        "loss": 0,
        "action_loss": 0,
        "dynamics_loss": 0,
        "traj3d_loss": 0,  # 添加 traj3d_loss 统计
        "action_mse_marginal": 0,
        "action_mse_joint": 0,
        "action_mse_inv": 0,
        "image_mse_marginal": 0,
        "image_mse_joint": 0,
        "image_mse_forward": 0,
    }
    for batch in tqdm(data_loader, desc="Evaluating", disable=not is_main_process()):
        # ------------ Preprocess data ------------ #
        curr_obs_dict, next_obs_dict, action_norm, traj3d = process_batch(
            batch, config.model.obs_encoder.num_frames, config.model.action_len, device
        )

        with torch.no_grad():
            # ------------ Validation loss ------------ #
            _, info = model(curr_obs_dict, next_obs_dict, action_norm, traj3d)
            for k, v in info.items():
                stats[k] += v

            # ------------ UWM Inference ------------ #
            action = unnormalize(action_norm)

            # Sample actions from marginal distribution
            action_hat_marg = model.sample_marginal_action(curr_obs_dict)
            marg_mse = F.mse_loss(unnormalize(action_hat_marg), action)
            dist.all_reduce(marg_mse, op=dist.ReduceOp.AVG)
            stats["action_mse_marginal"] += marg_mse

            # Sample actions from inverse dynamics
            action_hat_inv = model.sample_inverse_dynamics(curr_obs_dict, next_obs_dict)
            inv_mse = F.mse_loss(unnormalize(action_hat_inv), action)
            dist.all_reduce(inv_mse, op=dist.ReduceOp.AVG)
            stats["action_mse_inv"] += inv_mse

            # Encode next observations
            next_obs = model.obs_encoder.encode_next_obs(next_obs_dict)

            # Sample observations from marginal distribution
            next_obs_hat_marg = model.sample_marginal_next_obs(curr_obs_dict)
            marg_mse = F.mse_loss(next_obs_hat_marg, next_obs)
            dist.all_reduce(marg_mse, op=dist.ReduceOp.AVG)
            stats["image_mse_marginal"] += marg_mse

            # Sample observations from forward dynamics
            next_obs_hat_forward = model.sample_forward_dynamics(
                curr_obs_dict, action_norm
            )
            forward_mse = F.mse_loss(next_obs_hat_forward, next_obs)
            dist.all_reduce(forward_mse, op=dist.ReduceOp.AVG)
            stats["image_mse_forward"] += forward_mse

            # Sample next obs and actions from joint distribution
            next_obs_hat_joint, action_hat_joint = model.sample_joint(curr_obs_dict)
            joint_image_mse = F.mse_loss(next_obs_hat_joint, next_obs)
            dist.all_reduce(joint_image_mse, op=dist.ReduceOp.AVG)
            stats["image_mse_joint"] += joint_image_mse
            joint_action_mse = F.mse_loss(unnormalize(action_hat_joint), action)
            dist.all_reduce(joint_action_mse, op=dist.ReduceOp.AVG)
            stats["action_mse_joint"] += joint_action_mse

    # Average over all batches
    stats = {k: v / len(data_loader) for k, v in stats.items()}

    # Plot reconstruction
    stats["images"] = wandb.Image(decode_and_plot(next_obs[0:1]))
    stats["images_marginal"] = wandb.Image(decode_and_plot(next_obs_hat_marg[0:1]))
    stats["images_joint"] = wandb.Image(decode_and_plot(next_obs_hat_joint[0:1]))
    stats["images_forward"] = wandb.Image(decode_and_plot(next_obs_hat_forward[0:1]))
    return stats


def train_one_step(config, model, optimizer, scheduler, scaler, batch, device):
    model.train()

    # --- Preprocess data ---
    curr_obs, next_obs, action, traj3d = process_batch(
        batch, config.model.obs_encoder.num_frames, config.model.action_len, device
    )

    # --- UWM Training ---
    # Joint dynamics and action prediction loss
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=config.use_amp
    ):
        loss, info = model(
            curr_obs, next_obs, action, traj3d, 
            batch.get("action_mask", None), batch.get("traj3d_mask", None)
        )

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
            print(f"Step {step} action mse: {stats['action_mse_marginal']:.4f}")


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


@hydra.main(
    version_base=None, config_path="../../configs", config_name="train_uwm.yaml"
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    # Spawn processes
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
