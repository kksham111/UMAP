import os

from .wrappers import RoboMimicEnvWrapper, LIBEROEnvWrapper


def make_robomimic_env(
    dataset_name,
    dataset_path,
    shape_meta,
    obs_horizon,
    max_episode_length,
    record=False,
):
    if "robomimic" in dataset_name:
        from robomimic.utils.env_utils import get_env_type, get_env_class
        from robomimic.utils.file_utils import (
            get_env_metadata_from_dataset,
            get_shape_metadata_from_dataset,
        )
        from robomimic.utils.obs_utils import initialize_obs_utils_with_obs_specs

        # Initialize observation modalities
        rgb_keys = [k for k, v in shape_meta["obs"].items() if v["type"] == "rgb"]
        low_dim_keys = [
            k for k, v in shape_meta["obs"].items() if v["type"] == "low_dim"
        ]
        all_obs_keys = rgb_keys + low_dim_keys
        initialize_obs_utils_with_obs_specs(
            {"obs": {"rgb": rgb_keys, "low_dim": low_dim_keys}}
        )

        # Create environment
        env_meta = get_env_metadata_from_dataset(dataset_path=dataset_path)
        env_type = get_env_type(env_meta=env_meta)
        env_class = get_env_class(env_type=env_type)
        shape_meta = get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            all_obs_keys=all_obs_keys,
            verbose=True,
        )
        # Set render device if CUDA_VISIBLE_DEVICES is set
        if os.environ.get("CUDA_VISIBLE_DEVICES", None):
            env_meta["env_kwargs"]["render_gpu_device_id"] = int(
                os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
            )
        env = env_class(
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=True,
            use_image_obs=shape_meta["use_images"],
            use_depth_obs=shape_meta["use_depths"],
            postprocess_visual_obs=False,  # use raw images
            **env_meta["env_kwargs"],
        )
        env = RoboMimicEnvWrapper(
            env, all_obs_keys, obs_horizon, max_episode_length, record=record
        )
    elif "libero" in dataset_name:
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero import get_libero_path

        # Construct environment kwargs
        bddl_file_name = os.path.join(
            get_libero_path("bddl_files"),
            "libero_10",
            dataset_path.split("/")[-1].replace("_demo.hdf5", ".bddl"),
        )
        env_kwargs = {
            "bddl_file_name": bddl_file_name,
            "camera_heights": 128,
            "camera_widths": 128,
        }

        # Set render device if CUDA_VISIBLE_DEVICES is set
        if os.environ.get("CUDA_VISIBLE_DEVICES", None):
            env_kwargs["render_gpu_device_id"] = int(
                os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
            )

        # Create environment
        env = OffScreenRenderEnv(**env_kwargs)
        obs_keys = list(shape_meta["obs"].keys())
        env = LIBEROEnvWrapper(
            env, obs_keys, obs_horizon, max_episode_length, record=record
        )
    else:
        raise NotImplementedError(f"Unsupported environment: {dataset_name}")
    return env
