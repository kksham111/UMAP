import os
from typing import Union

import torch.distributed as dist
from .dataset import RobomimicDataset


def make_robomimic_dataset(
    name: str,
    hdf5_path_globs: Union[str, list[str]],
    buffer_path: str,
    shape_meta: dict,
    seq_len: int,
    val_ratio: float = 0.0,
    subsample_ratio: float = 1.0,
    flip_rgb: bool = False,
):
    # Cache compressed dataset in the main process
    if not os.path.exists(buffer_path):
        if not dist.is_initialized() or dist.get_rank() == 0:
            RobomimicDataset(
                name=name,
                hdf5_path_globs=hdf5_path_globs,
                buffer_path=buffer_path,
                shape_meta=shape_meta,
                seq_len=seq_len,
                flip_rgb=flip_rgb,
            )
    if dist.is_initialized():
        dist.barrier()

    # Training dataset
    train_set = RobomimicDataset(
        name=name,
        hdf5_path_globs=hdf5_path_globs,
        buffer_path=buffer_path,
        shape_meta=shape_meta,
        seq_len=seq_len,
        val_ratio=val_ratio,
        subsample_ratio=subsample_ratio,
        flip_rgb=flip_rgb,
    )
    val_set = train_set.get_validation_dataset()
    return train_set, val_set
