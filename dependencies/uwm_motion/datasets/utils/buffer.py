import math
import os
from multiprocessing import Lock
from os.path import expanduser, expandvars
from typing import Union, Optional

import numcodecs
import numpy as np
import zarr


def get_optimal_chunks(
    shape: tuple[int, ...], dtype: Union[str, np.dtype], target_chunk_bytes: float = 2e6
) -> tuple[int, ...]:
    """
    Calculates the optimal chunk sizes for an array given its shape, data type, and a target chunk size in bytes.

    Args:
        shape: The shape of the array.
        dtype: The data type of the array.
        target_chunk_bytes: The target size for each chunk in bytes. Defaults to 2e6 (2 MB).

    Returns:
        The optimal chunk dimensions for the given array shape and data type, aiming to not exceed
            the target chunk size in bytes.
    """
    itemsize = np.dtype(dtype).itemsize
    rshape = list(shape[::-1])

    # Find the index to split the shape, starting from the rightmost dimension
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if this_chunk_bytes <= target_chunk_bytes < next_chunk_bytes:
            split_idx = i
            break

    # Handle jagged chunk dimension
    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rchunks)
    next_chunk_length = min(
        rshape[split_idx], math.ceil(target_chunk_bytes / item_chunk_bytes)
    )
    rchunks.append(next_chunk_length)

    # Handle remaining dimensions
    rchunks.extend([1] * (len(shape) - len(rchunks)))
    chunks = tuple(rchunks[::-1])
    return chunks


class CompressedTrajectoryBuffer:
    """
    A class that stores trajectory data in a compressed zarr array.
    """

    def __init__(
        self,
        storage_path: str,
        metadata: dict[str, dict[str, any]],
        capacity: Optional[int] = None,
        lock: Optional[Lock] = None,
    ):
        """
        Initialize the trajectory buffer. If there is an existing buffer at the given path, it will be restored.

        Args:
            storage_path: Path to the buffer storage.
            metadata: Dictionary containing metadata for each data key. Each key should
                map to a dictionary containing the following keys:
                - shape: shape of the data
                - dtype: dtype of the data
            capacity: Maximum number of transition steps that can be stored in the buffer.
                Only used when creating a new buffer.
            lock: Multiprocessing lock to synchronize access to the buffer. If None, a new lock will be created.
        """
        # Create zarr storage and root group
        storage_path = expandvars(expanduser(storage_path))
        self.restored = os.path.exists(storage_path)
        self.storage = zarr.DirectoryStore(storage_path)

        # Mutex for zarr storage
        self.lock = Lock() if lock is None else lock

        # Create data and metadata groups
        self.root = zarr.group(store=self.storage)
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")

        if self.restored:
            print(f"Restoring buffer from {storage_path}")
            assert "episode_ends" in self.meta
            assert all(key in self.data for key in metadata)
            assert all(
                self.data[key].shape[1:] == value["shape"]
                for key, value in metadata.items()
            )

            # Check that all data have the same length and restore capacity
            lengths = {self.data[key].shape[0] for key in self.data}
            assert len(lengths) == 1, "Inconsistent data lengths in the buffer"
            self.capacity = lengths.pop()
        else:
            with self.lock:
                print(f"Creating new buffer at {storage_path}")
                assert capacity is not None, "Capacity must be specified for new buffer"
                self.capacity = capacity

                # Create empty episode_ends
                self.meta.zeros(
                    name="episode_ends",
                    shape=(0,),
                    dtype=np.int64,
                    compressor=None,
                )

                # Allocate space for data
                for key, value in metadata.items():
                    shape = (capacity,) + tuple(value["shape"])
                    dtype = value["dtype"]
                    if dtype == np.uint8:
                        # Chunk and compress images individually
                        chunks = (1,) + shape[1:]
                        compressor = numcodecs.Blosc(
                            cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
                        )
                    else:
                        # Chunk and compress other data by computing optimal chunks
                        chunks = get_optimal_chunks(shape, dtype)
                        compressor = numcodecs.Blosc(
                            cname="lz4", clevel=0, shuffle=numcodecs.Blosc.NOSHUFFLE
                        )
                    # Create new array
                    self.data.zeros(
                        name=key,
                        shape=shape,
                        chunks=chunks,
                        dtype=dtype,
                        compressor=compressor,
                        object_codec=numcodecs.Pickle(),
                    )

    @property
    def episode_ends(self) -> np.ndarray:
        return self.meta["episode_ends"]

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ends)

    @property
    def num_steps(self) -> int:
        return self.episode_ends[-1] if len(self.episode_ends) > 0 else 0

    def add_episode(self, data: dict[str, np.ndarray]):
        with self.lock:
            # Get episode length
            episode_lens = np.array([v.shape[0] for v in data.values()])
            assert np.all(episode_lens == episode_lens[0])
            episode_len = episode_lens[0]

            # Compute corresponding buffer indices
            start_ind = self.num_steps
            end_ind = start_ind + episode_len
            if end_ind > self.capacity:
                raise RuntimeError("Buffer capacity exceeded")

            # Copy data to buffer
            for key, value in data.items():
                arr = self.data[key]
                arr[start_ind:end_ind] = value

            # Update episode_ends
            self.episode_ends.resize(len(self.episode_ends) + 1)
            self.episode_ends[-1] = end_ind

            # Rechunk and recompress episode_ends if necessary
            if self.episode_ends.chunks[0] < self.episode_ends.shape[0]:
                new_chunk_len = self.episode_ends.shape[0] * 1.5
                new_chunks = (new_chunk_len,) + self.episode_ends.chunks[1:]
                self.meta.move("episode_ends", "_temp")
                zarr.copy(
                    source=self.meta["_temp"],
                    dest=self.meta,
                    name="episode_ends",
                    chunks=new_chunks,
                    compressor=None,
                )
                del self.meta["_temp"]

    def __repr__(self) -> str:
        return str(self.root.tree())

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data
