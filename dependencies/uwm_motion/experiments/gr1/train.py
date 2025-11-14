import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
from experiments.uwm.train import (
    process_batch,
    train_one_step,
    maybe_resume_checkpoint,
    maybe_save_checkpoint,
)


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

    def plot(images, nrows=4):
        images = rearrange(images, "b v c t h w -> (b v t) c h w")
        images_grid = make_grid(images, nrows)
        return images_grid

    stats = {
        "loss": 0,
        "action_loss": 0,
        "dynamics_loss": 0,
        "action_mse_joint": 0,
        "image_mse_joint": 0,
    }
    for batch in tqdm(data_loader, desc="Evaluating", disable=not is_main_process()):
        # ------------ Preprocess data ------------ #
        curr_obs_dict, next_obs_dict, action_norm = process_batch(
            batch, config.model.obs_encoder.num_frames, config.model.action_len, device
        )

        with torch.no_grad():
            # ------------ Validation loss ------------ #
            _, info = model(curr_obs_dict, next_obs_dict, action_norm)
            for k, v in info.items():
                stats[k] += v

            # ------------ UWM Inference ------------ #
            action = unnormalize(action_norm)

            # Encode next observations
            next_obs = model.obs_encoder.apply_transform(next_obs_dict)

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
    stats["images"] = wandb.Image(plot(next_obs[0:1]))
    stats["images_joint"] = wandb.Image(plot(next_obs_hat_joint[0:1]))
    return stats


def maybe_evaluate(config, step, model, loader, device, action_normalizer=None):
    """Evaluate if it's the correct step."""
    if step % config.eval_every == 0:
        stats = eval_one_epoch(config, loader, device, model, action_normalizer)
        if is_main_process():
            wandb.log({f"eval/{k}": v for k, v in stats.items()})
            print(f"Step {step} action mse: {stats['action_mse_joint']:.4f}")


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
    version_base=None, config_path="../../configs", config_name="train_gr1.yaml"
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    # Spawn processes
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
