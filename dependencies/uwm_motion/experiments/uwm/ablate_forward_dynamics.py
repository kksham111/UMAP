import os

import hydra
import imageio
import torch
import numpy as np
from diffusers.optimization import get_scheduler
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from tqdm import tqdm

from datasets.utils.loader import make_distributed_data_loader
from experiments.utils import set_seed, is_main_process


def process_batch(batch, obs_horizon, action_horizon, device):
    action_start = obs_horizon - 1
    action_end = action_start + action_horizon
    curr_obs = {k: v[:, : action_start + 1].to(device) for k, v in batch["obs"].items()}
    next_obs = {k: v[:, action_end:].to(device) for k, v in batch["obs"].items()}
    actions = batch["action"][:, action_start:action_end].to(device)

    # Add language tokens
    if "input_ids" in batch and "attention_mask" in batch:
        curr_obs["input_ids"] = batch["input_ids"].to(device)
        curr_obs["attention_mask"] = batch["attention_mask"].to(device)
    return curr_obs, next_obs, actions


def eval_one_epoch(config, data_loader, device, model):
    model.eval()
    model = getattr(model, "module", model)  # unwrap DDP

    def decode_and_plot(images, nrows=2):
        images = model.obs_encoder.apply_vae(images, inverse=True)
        images = rearrange(images, "b v c t h w -> (b v t) c h w").clamp(0, 1)
        images_grid = make_grid(images, nrows)
        return (
            (images_grid.cpu().numpy() * 255)
            .round()
            .astype(np.uint8)
            .transpose(1, 2, 0)
        )

    save_path = f"viz_{config.dataset.name}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    step = 0
    for batch in tqdm(data_loader, desc="Evaluating", disable=not is_main_process()):
        # ------------ Preprocess data ------------ #
        curr_obs_dict, next_obs_dict, action_norm = process_batch(
            batch, config.model.obs_encoder.num_frames, config.model.action_len, device
        )

        with torch.no_grad():
            # Encode current observations
            curr_obs = model.obs_encoder.encode_next_obs(curr_obs_dict)

            # Encode next observations
            next_obs = model.obs_encoder.encode_next_obs(next_obs_dict)

            # Sample observations from forward dynamics
            next_obs_hat_forward = model.sample_forward_dynamics(
                curr_obs_dict, action_norm
            )

        # Plot current observations
        curr_obs = decode_and_plot(curr_obs[:1])
        imageio.imwrite(f"{save_path}/{step}_curr_obs.png", curr_obs)

        # Plot next observations
        next_obs = decode_and_plot(next_obs[:1])
        imageio.imwrite(f"{save_path}/{step}_next_obs.png", next_obs)

        # Plot predicted next observations
        next_obs_hat_forward = decode_and_plot(next_obs_hat_forward[:1])
        imageio.imwrite(f"{save_path}/{step}_next_obs_hat.png", next_obs_hat_forward)

        step += 1
        if step == 15:
            break


def train(rank, world_size, config):
    # Set global seed
    set_seed(config.seed * world_size + rank)

    # Initialize distributed training
    device = torch.device(f"cuda:{rank}")

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

    # Resume from checkpoint
    ckpt_path = os.path.join(config.logdir, "models.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    step = ckpt["step"] + 1
    print(f"Resumed training from step {step}")

    train_set.action_normalizer = ckpt["action_normalizer"]
    train_set.lowdim_normalizer = ckpt["lowdim_normalizer"]
    val_set.action_normalizer = ckpt["action_normalizer"]
    val_set.lowdim_normalizer = ckpt["lowdim_normalizer"]

    eval_one_epoch(config, val_loader, device, model)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="train_uwm.yaml"
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    train(0, 1, config)


if __name__ == "__main__":
    main()
