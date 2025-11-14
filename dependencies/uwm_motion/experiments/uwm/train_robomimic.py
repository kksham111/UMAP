import hydra
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from diffusers.optimization import get_scheduler
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from tqdm import trange, tqdm

from datasets.utils.loader import make_distributed_data_loader
from environments.robomimic import make_robomimic_env
from experiments.utils import set_seed, init_wandb, init_distributed, is_main_process
from experiments.uwm.train import (
    train_one_step,
    maybe_resume_checkpoint,
    maybe_evaluate,
    maybe_save_checkpoint,
)


def collect_rollout(config, model, device):
    model.eval()
    model = getattr(model, "module", model)  # unwrap DDP

    # Create eval environment
    assert isinstance(config.dataset.hdf5_path_globs, str)
    env = make_robomimic_env(
        dataset_name=config.dataset.name,
        dataset_path=config.dataset.hdf5_path_globs,
        shape_meta=config.dataset.shape_meta,
        obs_horizon=model.obs_encoder.num_frames,
        max_episode_length=config.rollout_length,
        record=True,
    )

    # Collect rollouts
    successes = []
    for e in trange(
        config.num_rollouts, desc="Collecting rollouts", disable=not is_main_process()
    ):
        env.seed(e)
        obs = env.reset()
        done = False
        while not done:
            obs_tensor = {
                k: torch.tensor(v, device=device)[None] for k, v in obs.items()
            }

            # Sample action from model
            action = model.sample(obs_tensor)[0].cpu().numpy()

            # Step environment
            obs, reward, done, info = env.step(action)
        successes.append(info["success"])

    # Compute success rate
    success_rate = sum(successes) / len(successes)

    # Record video of the last episode
    video = env.get_video()
    return success_rate, video


def maybe_collect_rollout(config, step, model, device):
    """Collect rollouts on the main process if it's the correct step."""
    # Skip rollout rollection for pretraining
    if "libero_90" in config.dataset.name:
        return

    if is_main_process() and (
        step % config.rollout_every == 0 or step == (config.num_steps - 1)
    ):
        success_rate, video = collect_rollout(config, model, device)
        print(f"Step: {step} success rate: {success_rate}")
        # Video shape: (T, H, W, C) -> (N, T, C, H, W)
        video = video.transpose(0, 3, 1, 2)[None]
        wandb.log(
            {
                "rollout/success_rate": success_rate,
                "rollout/video": wandb.Video(video, fps=10),
            }
        )
    dist.barrier()


def train(rank, world_size, config):
    # Set global seed
    set_seed(config.seed * world_size + rank)

    # Initialize distributed training
    init_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Initialize WANDB
    if is_main_process():
        os.environ["WANDB_MODE"] = "offline"
        init_wandb(config, job_type="train")

    # Create dataset and loader
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

        # Use a smaller learning rate for the image encoder
        encoder_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("obs_encoder.img_encoder"):
                encoder_params.append(param)
            else:
                other_params.append(param)

        # Define learning rates
        base_lr = config.optimizer.lr
        encoder_lr = base_lr * 0.05
        wd = config.optimizer.weight_decay

        # Construct optimizer with custom parameter groups
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": encoder_lr, "weight_decay": wd},
                {"params": other_params, "lr": base_lr, "weight_decay": wd},
            ],
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
        )
        scheduler = get_scheduler(optimizer=optimizer, **config.scheduler)

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

        # Train for one epoch
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
            maybe_evaluate(config, step, model, val_loader, device)

            # ---Collect environment rollouts if needed ---
            maybe_collect_rollout(config, step, model, device)

            # --- Save checkpoint if needed ---
            maybe_save_checkpoint(config, step, model, optimizer, scheduler, scaler)

            step += 1
            pbar.update(1)
            if step >= config.num_steps:
                break

        epoch += 1


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_uwm_robomimic.yaml",
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    # Spawn processes
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
