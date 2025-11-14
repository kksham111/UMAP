from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def make_distributed_data_loader(
    train_set: Dataset,
    val_set: Dataset,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = True,
):
    # Training sampler and loader
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )

    # Validation sampler and loader
    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader
