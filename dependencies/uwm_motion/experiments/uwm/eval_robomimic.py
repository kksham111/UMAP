import os

import hydra
import imageio
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import trange

from environments.robomimic import make_robomimic_env
from experiments.uwm.train import maybe_resume_checkpoint
from experiments.utils import set_seed, is_main_process


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
    video_dir = os.path.join(config.logdir, "videos")
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
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
        video = env.get_video()
        imageio.mimwrite(os.path.join(video_dir, f"{e}.mp4"), video, fps=30)
        print(
            f"Episode {e} success: {info['success']}, cumulative: {np.mean(successes):.2f}"
        )

    # Compute success rate
    success_rate = sum(successes) / len(successes)
    return success_rate


def maybe_collect_rollout(config, step, model, device):
    """Collect rollouts on the main process if it's the correct step."""
    # Skip rollout rollection for pretraining
    if "libero_90" in config.dataset.name:
        return

    if is_main_process() and (
        step % config.rollout_every == 0 or step == (config.num_steps - 1)
    ):
        success_rate = collect_rollout(config, model, device)
        print(f"Step: {step} success rate: {success_rate}")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_uwm_robomimic.yaml",
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    set_seed(0)
    device = torch.device(f"cuda:0")

    # Create model
    model = instantiate(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **config.scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    # Resume from checkpoint
    config.resume = True
    step = maybe_resume_checkpoint(config, model, optimizer, scheduler, scaler)
    maybe_collect_rollout(config, 0, model, device)


if __name__ == "__main__":
    main()
