import copy
import math
import pathlib

import dask.array as da
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.utils.buffer import CompressedTrajectoryBuffer
from datasets.utils.normalizer import LinearNormalizer, NestedDictLinearNormalizer
from datasets.utils.obs_utils import unflatten_obs
from datasets.utils.sampler import TrajectorySampler


class DroidDataset(Dataset):
    def __init__(
        self,
        name: str,
        buffer_path: str,
        shape_meta: dict,
        seq_len: int,
        history_len: int = 1,
        normalize_lowdim: bool = False,
        normalize_action: bool = False,
        val_ratio: float = 0.0,
        num_workers: int = 8,
    ):
        self.name = name
        self.seq_len = seq_len
        self.history_len = history_len
        self.num_workers = num_workers

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
        self._action_shape = tuple(shape_meta["action"]["shape"])

        # Compressed buffer to store episode data
        self.buffer_dir = pathlib.Path(buffer_path).parent
        self.buffer = self._init_buffer(buffer_path)

        # Create training-validation split
        num_episodes = self.buffer.num_episodes
        val_mask = np.zeros(num_episodes, dtype=bool)
        if val_ratio > 0:
            num_val_episodes = round(val_ratio * num_episodes)
            num_val_episodes = min(max(num_val_episodes, 1), num_episodes - 1)
            rng = np.random.default_rng(seed=0)
            val_inds = rng.choice(num_episodes, num_val_episodes, replace=False)
            val_mask[val_inds] = True
        self.train_mask = ~val_mask
        self.is_validation = False  # flag for __getitem__

        # Sampler to sample sequences from buffer
        self.sampler = TrajectorySampler(self.buffer, self.seq_len, self.train_mask)

        # Low-dim observation normalizer
        if normalize_lowdim:
            self.lowdim_normalizer = self._init_lowdim_normalizer()

        # Action normalizer
        if normalize_action:
            self.action_normalizer = self._init_action_normalizer()

    def _init_buffer(self, buffer_path):
        # Create metadata
        metadata = {}
        for key, shape in self._image_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.uint8}
        for key, shape in self._lowdim_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.float32}
        metadata["action"] = {"shape": self._action_shape, "dtype": np.float32}

        # Load buffer
        buffer = CompressedTrajectoryBuffer(storage_path=buffer_path, metadata=metadata)
        assert buffer.restored, f"Buffer not found at {buffer_path}"
        return buffer

    def _init_lowdim_normalizer(self):
        # Load cached normalizer statistics
        normalizer_stats_path = self.buffer_dir / "lowdim_normalizer_stats.npz"
        if normalizer_stats_path.exists():
            print(f"Loading lowdim normalizer stats from {normalizer_stats_path}")
            stats = np.load(normalizer_stats_path)
            return NestedDictLinearNormalizer(stats)

        stats = {}
        for key in self._lowdim_shapes.keys():
            data = da.from_zarr(self.buffer[f"obs.{key}"])
            min_val = data.min(axis=0).compute()
            max_val = data.max(axis=0).compute()
            scale = (max_val - min_val) / 2.0
            offset = (max_val + min_val) / 2.0
            stats[key] = (scale, offset)

        # Cache normalizer statistics
        np.savez(normalizer_stats_path, **stats)
        return NestedDictLinearNormalizer(stats)

    def _init_action_normalizer(self):
        # Load cached normalizer statistics
        normalizer_stats_name = f"action_normalizer_stats_len{self.seq_len}.npz"
        normalizer_stats_path = self.buffer_dir / normalizer_stats_name
        if normalizer_stats_path.exists():
            print(f"Loading action normalizer stats from {normalizer_stats_path}")
            stats = np.load(normalizer_stats_path)
            return LinearNormalizer(stats["scale"], stats["offset"])

        # Use dask to compute normalization statistics
        actions = da.from_zarr(self.buffer["action"])
        min_action = actions.min(axis=0).compute()
        max_action = actions.max(axis=0).compute()

        # Compute normalizer statistics
        scale = (max_action - min_action) / 2.0
        offset = (max_action + min_action) / 2.0

        # Cache normalizer statistics
        np.savez(normalizer_stats_path, scale=scale, offset=offset)
        return LinearNormalizer(scale, offset)

    def __len__(self) -> int:
        return len(self.sampler)

    def __repr__(self) -> str:
        return (
            "<DroidDataset>\n"
            f"name: {self.name}\n"
            f"num_samples: {len(self)}\n"
            f"{self.buffer}"
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Sample a sequence of observations and actions from the dataset
        data = self.sampler.sample_sequence(idx)

        # Normalize low-dim observations
        if hasattr(self, "lowdim_normalizer"):
            for key in self._lowdim_shapes.keys():
                data[f"obs.{key}"] = self.lowdim_normalizer[key](data[f"obs.{key}"])

        # Normalize actions
        if hasattr(self, "action_normalizer"):
            data["action"] = self.action_normalizer(data["action"])

        # Convert data to torch tensors
        data = {k: torch.from_numpy(v) for k, v in data.items()}

        # Unflatten observations
        data = unflatten_obs(data)
        return data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set.sampler = TrajectorySampler(self.buffer, self.seq_len, ~self.train_mask)
        val_set.is_validation = True
        return val_set


class DroidMixtureDataset(Dataset):
    def __init__(
        self,
        base_dataset: DroidDataset,
        video_buffer_path: str,
        balance_datasets: bool = False,
    ):
        self.base_dataset = base_dataset

        # Video buffer and sampler
        self.video_buffer = self.base_dataset._init_buffer(video_buffer_path)
        self.video_sampler = TrajectorySampler(
            self.video_buffer, self.base_dataset.seq_len
        )

        if balance_datasets:
            # Balance robot and video datasets
            # 1) figure out lengths
            len_robot = len(self.base_dataset)
            len_video = len(self.video_sampler)
            max_len = max(len_robot, len_video)

            # 2) build expanded index lists
            robot_factor = math.ceil(max_len / len_robot)
            video_factor = math.ceil(max_len / len_video)

            # pad indices
            robot_indices = []
            for _ in range(robot_factor):
                robot_indices.extend(range(len_robot))
            video_indices = []
            for _ in range(video_factor):
                video_indices.extend(range(len_video))

            # trim to max_len
            robot_indices = robot_indices[:max_len]
            video_indices = video_indices[:max_len]

            # 3) combine robot and video data
            combined_indices = []
            for r_i in robot_indices:
                combined_indices.append((True, r_i))
            for v_i in video_indices:
                combined_indices.append((False, v_i))
            self.index_map = combined_indices  # list of (bool, idx)
        else:
            # just combine the two datasets
            combined_indices = []
            for r_i in range(len(self.base_dataset)):
                combined_indices.append((True, r_i))
            for v_i in range(len(self.video_sampler)):
                combined_indices.append((False, v_i))
            self.index_map = combined_indices

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"Attempting to access private attribute '{name}'")
        return getattr(self.base_dataset, name)

    def __len__(self):
        return len(self.index_map)

    def __repr__(self) -> str:
        return (
            "<DroidMixtureDataset>\n"
            f"name: {self.base_dataset.name}\n"
            f"num_robot_samples: {len(self.base_dataset)}\n"
            f"num_video_samples: {len(self.video_sampler)}\n"
            f"{self.base_dataset.buffer}"
        )

    def __getitem__(self, idx):
        is_robot, dataset_idx = self.index_map[idx]
        if is_robot:
            data = self.base_dataset[dataset_idx]
            data["action_mask"] = torch.tensor(1, dtype=torch.bool)
        else:
            data = self.video_sampler.sample_sequence(dataset_idx)
            data = {k: torch.from_numpy(v) for k, v in data.items()}
            data = unflatten_obs(data)
            data["action_mask"] = torch.tensor(0, dtype=torch.bool)
        return data
