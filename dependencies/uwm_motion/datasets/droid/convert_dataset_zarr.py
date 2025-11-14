import argparse
import multiprocessing as mp
import os

import numpy as np
import torch
from tqdm import tqdm

import tensorflow_datasets as tfds

from datasets.utils.buffer import CompressedTrajectoryBuffer
from datasets.droid.utils import euler_angles_to_rot_6d

# Disable GPU otherwise jax allocates lots of memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Shape metadata for zarr dataset
shape_meta = {
    "obs": {
        "exterior_image_1_left": {
            "shape": (180, 320, 3),
            "type": "rgb",
        },
        "exterior_image_2_left": {
            "shape": (180, 320, 3),
            "type": "rgb",
        },
        "wrist_image_left": {
            "shape": (180, 320, 3),
            "type": "rgb",
        },
        "cartesian_position": {
            "shape": (6,),
            "type": "low_dim",
        },
        "gripper_position": {
            "shape": (1,),
            "type": "low_dim",
        },
    },
    "action": {
        "shape": (10,),
    },
}

rgb_keys = [k for k, v in shape_meta["obs"].items() if v["type"] == "rgb"]
lowdim_keys = [k for k, v in shape_meta["obs"].items() if v["type"] == "low_dim"]


class TruncatedDataset:
    def __init__(self, dataset, num_episodes, filter_key=None, except_key=None):
        # Generate up to num_episodes episodes that contain filter_key
        # and doesn't contain except_key in their folderpath
        self.dataset = dataset
        self.num_episodes = num_episodes
        if filter_key and except_key:
            assert (
                filter_key != except_key
            ), "filter_key and except_key cannot be the same"
        self.filter_key = filter_key
        self.except_key = except_key

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        episode_count = 0
        for episode in self.dataset:
            folderpath = (
                episode["episode_metadata"]["recording_folderpath"]
                .numpy()
                .decode("UTF-8")
            )
            if self.filter_key and self.filter_key not in folderpath:
                continue
            if self.except_key and self.except_key in folderpath:
                continue
            yield episode
            episode_count += 1
            if episode_count >= self.num_episodes:
                break


def _load_episode(episode):
    obs_dict = {k: [] for k in shape_meta["obs"].keys()}
    actions = []
    for step in episode:
        obs = step["observation"]
        # Load image observations
        for k in rgb_keys:
            obs_dict[k].append(obs[k].numpy())
        # Load low dimensional obervations
        obs_dict["cartesian_position"].append(obs["cartesian_position"].numpy())
        obs_dict["gripper_position"].append(obs["gripper_position"].numpy())

        # Convert and load actions
        action_dict = step["action_dict"]
        cartesian_velocity = action_dict["cartesian_velocity"].numpy()
        xyz, euler = cartesian_velocity[:3], cartesian_velocity[3:6]
        rot6d = euler_angles_to_rot_6d(torch.tensor(euler)).numpy()
        grippers = action_dict["gripper_position"].numpy()
        action = np.concatenate([xyz, rot6d, grippers])
        actions.append(action)

    episode_dict = {"obs." + k: np.stack(v) for k, v in obs_dict.items()}
    episode_dict["action"] = np.stack(actions)
    return episode_dict


def preprocess_episode(episode):
    """Preprocess episode by removing variant tensors and standardizing data types."""
    processed_episode = []
    for step in episode["steps"]:
        processed_step = {
            "observation": step["observation"],
            "action_dict": step["action_dict"],
        }
        processed_episode.append(processed_step)
    return processed_episode


def episode_loader(queue, buffer_args):
    """Process episodes from input queue and put results in output queue."""
    pid = mp.current_process().name
    print(f"Starting episode loader process {pid}")

    # Initialize buffer
    buffer = CompressedTrajectoryBuffer(**buffer_args)

    while True:
        episode = queue.get()
        if episode is None:
            print(f"Episode loader {pid} received a termination signal")
            break
        else:
            print(f"Episode loader {pid} received an episode")
            buffer.add_episode(_load_episode(episode))


def main(args):
    # Load dataset
    raw_dataset = tfds.load(args.data_name, data_dir=args.data_dir, split=f"train")
    dataset = TruncatedDataset(
        raw_dataset, args.num_episodes, args.filter_key, args.except_key
    )

    # Create metadata
    metadata = {}
    for key, meta in shape_meta["obs"].items():
        metadata[f"obs.{key}"] = {
            "shape": meta["shape"],
            "dtype": np.uint8 if meta["type"] == "rgb" else np.float32,
        }
    metadata["action"] = {"shape": shape_meta["action"]["shape"], "dtype": np.float32}

    # Compute buffer capacity
    capacity = sum([len(episode["steps"]) for episode in dataset])
    print(f"Buffer capacity: {capacity}")

    # Create temporary buffer to check if buffer is restored
    buffer = CompressedTrajectoryBuffer(
        storage_path=args.buffer_path,
        metadata=metadata,
        capacity=capacity,
    )
    if buffer.restored:
        print("Buffer restored from disk")
        return

    # Multiprocessing setup
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=args.num_workers * 2)
    lock = context.Lock()  # share lock across processes
    buffer_args = {
        "storage_path": args.buffer_path,
        "metadata": metadata,
        "capacity": capacity,
        "lock": lock,
    }

    # Start episode loader processes
    episode_loader_processes = []
    for i in range(args.num_workers):
        p = context.Process(
            target=episode_loader,
            args=(queue, buffer_args),
            name=f"EpisodeLoaderProcess-{i}",
        )
        p.start()
        episode_loader_processes.append(p)

    # Preprocess episodes on main process
    for episode in tqdm(dataset, desc="Preprocessing"):
        processed_episode = preprocess_episode(episode)
        queue.put(processed_episode)

    # Send termination signals to loaders
    for _ in range(args.num_workers):
        queue.put(None)

    # Wait for all processes to complete
    for p in episode_loader_processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="droid")
    parser.add_argument(
        "--data_dir", type=str, default="/gscratch/weirdlab/memmelma/data/"
    )
    parser.add_argument(
        "--buffer_path",
        type=str,
        default="/gscratch/weirdlab/zchuning/data/droid/buffer.zarr",
    )
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--filter_key", type=str, default=None)
    parser.add_argument("--except_key", type=str, default=None)
    args = parser.parse_args()
    main(args)
