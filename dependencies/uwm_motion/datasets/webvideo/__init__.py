from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from .dataset import MultiviewVideoDataset


def make_multiview_video_dataset(
    name: str,
    index_paths: list[str],
    shape_meta: dict,
    seq_len: int,
    frame_skip: int = 2,
    obs_padding: str = "same",
    val_ratio: float = 0.0,
):
    dataset = MultiviewVideoDataset(
        index_paths=index_paths,
        shape_meta=shape_meta,
        clip_len=seq_len,
        frame_skip=frame_skip,
        obs_padding=obs_padding,
    )

    train_size = int(len(dataset) * (1 - val_ratio))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    return train_set, val_set
