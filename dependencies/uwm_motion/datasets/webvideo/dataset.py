import math
import os
import warnings

import numpy as np
import torch

from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, center_crop


def compute_resize_shape(original_size: tuple[int, int], target_size: tuple[int, int]):
    """
    This method calculates the dimensions (height and width) needed to resize an image so that it
    minimally bounds the target dimensions defined in `target_size` while preserving the original
    aspect ratio.

    Args:
        original_size (tuple): The original height and width of the image.
        target_size (tuple): The target height and width of the image.

    Returns:
        tuple: A tuple representing the computed height and width for resizing.

    """
    h, w = original_size
    target_h, target_w = target_size
    scale = max(target_h / h, target_w / w)
    new_h = math.ceil(h * scale)
    new_w = math.ceil(w * scale)
    return (new_h, new_w)


class MultiviewVideoDataset(Dataset):
    def __init__(
        self,
        index_paths: list[str],
        shape_meta: dict,
        clip_len: int,
        frame_skip: int = 2,
        obs_padding: str = "same",
    ):
        """A video dataset that augments single-view videos to multi-view videos
        by padding missing observations.

        Args:
            index_paths: list of paths to index files each containing paths to video files.
            shape_meta: dictionary containing metadata for observations and actions.
            clip_len: length of each clip in frames.
            frame_skip: number of frames to skip between frames.
            obs_padding: padding method for observations, chosen from ["none", "same", "random"]
        """
        self.index_paths = index_paths
        self.clip_len = clip_len
        self.frame_skip = frame_skip
        self.obs_padding = obs_padding

        self.image_shapes = {}
        self.lowdim_shapes = {}
        for key, attr in shape_meta["obs"].items():
            obs_type, obs_shape = attr["type"], tuple(attr["shape"])
            if obs_type == "rgb":
                self.image_shapes[key] = obs_shape
            elif obs_type == "low_dim":
                self.lowdim_shapes[key] = obs_shape
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        self.action_shape = shape_meta["action"]["shape"]

        # Check that all image observations have the same shape
        assert (
            len(set(self.image_shapes.values())) == 1
        ), "Image observations must have the same shape"
        self.image_size = next(iter(self.image_shapes.values()))[:2]
        self.image_keys = list(self.image_shapes.keys())

        self.samples = []
        for index_path in index_paths:
            # Each line in a data file is a path to a video clip
            with open(index_path, "r") as f:
                video_paths = f.read().splitlines()
                self.samples.extend(video_paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Keep trying to load video until successful
        clip = None
        while clip is None:
            clip = self.load_video(self.samples[index])
            if clip is None:
                index = np.random.randint(self.__len__())

        # Construct image observations based on padding method
        obs_dict = {}
        if self.obs_padding == "none":
            main_key = np.random.choice(self.image_keys)
            obs_dict[main_key] = clip
            for key in [k for k in self.image_keys if k != main_key]:
                obs_dict[key] = torch.zeros_like(clip)
        elif self.obs_padding == "same":
            for key in self.image_keys:
                obs_dict[key] = clip
        elif self.obs_padding == "random":
            main_key = np.random.choice(self.image_keys)
            obs_dict[main_key] = clip
            for key in [k for k in self.image_keys if k != main_key]:
                rand_clip = None
                while rand_clip is None:
                    rand_index = np.random.randint(self.__len__())
                    rand_clip = self.load_video(self.samples[rand_index])
                obs_dict[key] = rand_clip
        else:
            raise ValueError(f"Invalid padding method {self.obs_padding}")

        # Zero pad lowdim observations
        for key in self.lowdim_shapes.keys():
            obs_dict[key] = torch.zeros(self.clip_len, *self.lowdim_shapes[key])

        # Construct sample
        sample = {
            "obs": obs_dict,
            "action": torch.zeros(self.clip_len, *self.action_shape),
            "action_mask": torch.tensor(0, dtype=torch.bool),
        }
        return sample

    def load_video(self, fname: str):
        if not os.path.exists(fname):
            warnings.warn(f"video path not found {fname}")
            return None

        # Skip short or long videos
        fsize = os.path.getsize(fname)
        if fsize < 1 * 1024 or fsize > int(10**9):
            warnings.warn(f"video size {fsize} out of bounds {fname}")
            return None

        # Try loading video
        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return None

        # Compute full clip length
        full_clip_len = int(self.clip_len * self.frame_skip)

        # Filter videos shorter than a single clip
        if len(vr) < full_clip_len:
            warnings.warn(f"video length {len(vr)} shorter than a single clip {fname}")
            return None

        # Sample random clip from video
        start_indx = np.random.randint(0, len(vr) - full_clip_len + 1)
        end_indx = start_indx + full_clip_len
        indices = np.linspace(start_indx, end_indx, self.clip_len)
        indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)

        # Load clip
        vr.seek(0)
        clip = vr.get_batch(indices).asnumpy()

        # Postprocess video
        clip = torch.from_numpy(clip)
        clip = clip.permute(0, 3, 1, 2)  # (T, C, H, W)
        clip = resize(clip, compute_resize_shape(clip.shape[2:], self.image_size))
        clip = center_crop(clip, self.image_size)
        return clip.permute(0, 2, 3, 1)  # (T, H, W, C)
