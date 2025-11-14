import copy
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.utils.buffer import CompressedTrajectoryBuffer
from datasets.utils.file_utils import glob_all
from datasets.utils.sampler import TrajectorySampler
from datasets.utils.obs_utils import unflatten_obs


class RobomimicDataset(Dataset):
    def __init__(
        self,
        name: str,
        hdf5_path_globs: str,
        buffer_path: str,
        shape_meta: dict,
        seq_len: int,
        val_ratio: float = 0.0,
        subsample_ratio: float = 1.0,
        flip_rgb: bool = False,
    ):
        self.name = name
        self.seq_len = seq_len
        self.flip_rgb = flip_rgb

        # Parse observation and action shapes
        obs_shape_meta = shape_meta["obs"]
        self._image_shapes = {}
        self._lowdim_shapes = {}
        for key, attr in obs_shape_meta.items():
            obs_type = attr["type"]
            obs_shape = tuple(attr["shape"])
            if obs_type == "rgb":
                self._image_shapes[key] = obs_shape
            elif obs_type == "low_dim":
                self._lowdim_shapes[key] = obs_shape
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        
        # Parse action shape (original code)
        self._action_shape = tuple(shape_meta["action"]["shape"])
        
        # Parse traj3d shape (照搬 action 的处理方式)
        self._has_traj3d = "traj3d" in shape_meta
        if self._has_traj3d:
            self._traj3d_shape = tuple(shape_meta["traj3d"]["shape"])
            print(f"[Motion] traj3d detected in shape_meta with shape: {self._traj3d_shape}")

        # Compressed buffer to store episode data
        self.buffer = self._init_buffer(hdf5_path_globs, buffer_path)

        # Create training-validation split
        num_episodes = self.buffer.num_episodes
        val_mask = np.zeros(num_episodes, dtype=bool)
        if val_ratio > 0:
            num_val_episodes = round(val_ratio * num_episodes)
            num_val_episodes = min(max(num_val_episodes, 1), num_episodes - 1)
            rng = np.random.default_rng(seed=0)
            val_inds = rng.choice(num_episodes, num_val_episodes, replace=False)
            val_mask[val_inds] = True
        self.val_mask = val_mask
        self.train_mask = ~val_mask

        # Apply subsample_ratio to training episodes
        if subsample_ratio < 1.0:
            train_indices = np.where(self.train_mask)[0]
            num_train_episodes = len(train_indices)
            num_subsampled = round(num_train_episodes * subsample_ratio)
            num_subsampled = max(1, num_subsampled)  # Ensure at least one episode

            # Create a new mask with only the subsampled training episodes
            subsampled_train_mask = np.zeros(num_episodes, dtype=bool)
            rng = np.random.default_rng(seed=1)
            sampled_indices = rng.choice(train_indices, num_subsampled, replace=False)
            subsampled_train_mask[sampled_indices] = True
            self.train_mask = subsampled_train_mask

        # Sampler to draw sequences from buffer
        self.sampler = TrajectorySampler(self.buffer, self.seq_len, self.train_mask)

    def _init_buffer(self, hdf5_path_globs, buffer_path):
        hdf5_paths = glob_all(hdf5_path_globs)

        # Create metadata
        metadata = {}
        for key, shape in self._image_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.uint8}
        for key, shape in self._lowdim_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.float32}
        
        # Add action to metadata (original code)
        metadata["action"] = {"shape": self._action_shape, "dtype": np.float32}
        
        # Add traj3d to metadata (照搬 action 的处理方式)
        if self._has_traj3d:
            metadata["traj3d"] = {"shape": self._traj3d_shape, "dtype": np.float32}

        # Compute buffer capacity
        capacity = 0
        num_episodes = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path) as f:
                demos = f["data"]
                for i in range(len(demos)):
                    demo = demos[f"demo_{i}"]
                    capacity += demo["actions"].shape[0]
                num_episodes += len(demos)

        # Initialize buffer
        buffer = CompressedTrajectoryBuffer(
            storage_path=buffer_path,
            metadata=metadata,
            capacity=capacity,
        )

        # If buffer is restored from disk, return it
        if buffer.restored:
            return buffer

        # Otherwise, load episodes to buffer
        pbar = tqdm(total=num_episodes, desc="Loading episodes to buffer")
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path) as f:
                demos = f["data"]
                for i in range(len(demos)):
                    demo = demos[f"demo_{i}"]
                    episode = {}
                    
                    # Load observations (original code)
                    for key in self._image_shapes.keys():
                        if self.flip_rgb:
                            episode[f"obs.{key}"] = demo["obs"][key][:][:, ::-1]
                        else:
                            episode[f"obs.{key}"] = demo["obs"][key][:]
                    for key in self._lowdim_shapes.keys():
                        episode[f"obs.{key}"] = demo["obs"][key][:]
                    
                    # Load action (original code)
                    episode["action"] = demo["actions"][:]
                    
                    # Load traj3d (照搬 action 的处理方式，增加向后兼容)
                    if self._has_traj3d:
                        # 优先尝试新的 key 名称 "traj3d"
                        if "traj3d" in demo:
                            episode["traj3d"] = demo["traj3d"][:]
                        # 向后兼容旧的 key 名称 "3d_traj"
                        elif "3d_traj" in demo:
                            episode["traj3d"] = demo["3d_traj"][:]
                            print(f"[Motion] Using backward compatible key '3d_traj' in demo_{i}")
                        else:
                            raise RuntimeError(
                                f"[Motion] traj3d data not found in demo_{i} of {hdf5_path}. "
                                f"Expected key 'traj3d' or '3d_traj' but neither exists."
                            )
                    
                    buffer.add_episode(episode)
                    pbar.update(1)
        pbar.close()
        return buffer

    def __len__(self) -> int:
        return len(self.sampler)

    def __repr__(self) -> str:
        return f"<RobomimicDataset>\nname: {self.name}\nnum_samples: {len(self)}\n{self.buffer}"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Sample a sequence and apply preprocessing
        
        数据流：
        1. 从 buffer 采样序列（包含 obs, action, traj3d）
        2. 转换为 torch.Tensor
        3. Unflatten observations
        """
        # Sample a sequence of observations and actions from the dataset
        data = self.sampler.sample_sequence(idx)

        # Convert data to torch tensors
        data = {k: torch.from_numpy(v) for k, v in data.items()}

        # Unflatten observations
        data = unflatten_obs(data)
        return data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = self.val_mask
        val_set.sampler = TrajectorySampler(self.buffer, self.seq_len, self.val_mask)
        return val_set