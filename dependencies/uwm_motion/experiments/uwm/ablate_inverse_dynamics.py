import os

import hydra
import h5py
import imageio
import torch
from hydra.utils import instantiate
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
from omegaconf import OmegaConf
from tqdm import trange


from datasets.utils.loader import make_distributed_data_loader
from environments.robomimic.wrappers import LIBEROEnvWrapper
from experiments.utils import set_seed


def eval_inverse_dynamics(
    hdf5_path, obs_keys, obs_horizon, action_horizon, max_episode_length, model, device
):
    # Set eval mode
    model.eval()

    # Make environment to verify demo
    bddl_file_name = os.path.join(
        get_libero_path("bddl_files"),
        "libero_10",
        hdf5_path.split("/")[-1].replace("_demo.hdf5", ".bddl"),
    )
    env_kwargs = {
        "bddl_file_name": bddl_file_name,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES", None):
        env_kwargs["render_gpu_device_id"] = int(
            os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
        )
    env = OffScreenRenderEnv(**env_kwargs)
    env = LIBEROEnvWrapper(env, obs_keys, obs_horizon, max_episode_length, record=True)

    successes = []
    with h5py.File(hdf5_path) as f:
        demos = f["data"]
        for i in trange(len(demos)):
            demo = demos[f"demo_{i}"]
            obs = env.reset_to_state(demo["states"][0])
            next_index = action_horizon
            done = False
            while next_index + obs_horizon <= len(demo["actions"]) and not done:
                next_obs = {
                    key: demo["obs"][key][next_index : next_index + obs_horizon]
                    for key in obs_keys
                }
                obs_tensor = {
                    k: torch.tensor(v, device=device)[None] for k, v in obs.items()
                }
                next_obs_tensor = {
                    k: torch.tensor(v, device=device)[None] for k, v in next_obs.items()
                }
                action = model.sample_inverse_dynamics(obs_tensor, next_obs_tensor)
                # action = model.sample(obs_tensor)
                obs, _, done, info = env.step(action[0].cpu().numpy())
                next_index += action_horizon
            successes.append(info["success"])
            print(f"Episode {i}, success: {successes[-1]}")

            # Save video locally
            video = env.get_video()
            imageio.mimsave(f"episode_{i}.gif", video)

        print(f"Total: {len(demos)}, success: {sum(successes)}")


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

    # Resume from checkpoint
    ckpt_path = os.path.join(config.logdir, "models.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    step = ckpt["step"] + 1
    print(f"Loading checkpoint from step {step}")

    if ckpt["action_normalizer"] is not None:
        train_set.action_normalizer = ckpt["action_normalizer"]
        val_set.action_normalizer = ckpt["action_normalizer"]
    if ckpt["lowdim_normalizer"] is not None:
        train_set.lowdim_normalizer = ckpt["lowdim_normalizer"]
        val_set.lowdim_normalizer = ckpt["lowdim_normalizer"]

    obs_keys = list(model.obs_encoder.rgb_keys) + list(model.obs_encoder.low_dim_keys)
    eval_inverse_dynamics(
        config.dataset.hdf5_path_globs,
        obs_keys,
        config.model.obs_encoder.num_frames,
        config.model.action_len,
        config.rollout_length,
        model,
        device,
    )


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_uwm_robomimic.yaml",
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    train(0, 1, config)


if __name__ == "__main__":
    main()
