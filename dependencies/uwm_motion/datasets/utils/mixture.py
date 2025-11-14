import math

import torch
from torch.utils.data import Dataset


class RobotVideoMixtureDataset(Dataset):
    def __init__(
        self,
        robot_dataset: Dataset,
        video_dataset: Dataset,
        balance_datasets: bool = False,
    ):
        self.robot_dataset = robot_dataset
        self.video_dataset = video_dataset

        # Copy action and lowdim normalizers from robot dataset
        if hasattr(self.robot_dataset, "action_normalizer"):
            self.action_normalizer = self.robot_dataset.action_normalizer
        if hasattr(self.robot_dataset, "lowdim_normalizer"):
            self.lowdim_normalizer = self.robot_dataset.lowdim_normalizer

        # Balance robot and video datasets
        if balance_datasets:
            # Figure out dataset lengths
            len_robot = len(self.robot_dataset)
            len_video = len(self.video_dataset)
            max_len = max(len_robot, len_video)
            print(
                f"Balancing data: {len_robot} robot, {len_video} video -> upsample to {max_len} each"
            )

            # Upsample robot data
            robot_factor = math.ceil(max_len / len_robot)
            robot_indices = []
            for _ in range(robot_factor):
                robot_indices.extend(range(len_robot))
            robot_indices = robot_indices[:max_len]

            # Upsample video data
            video_factor = math.ceil(max_len / len_video)
            video_indices = []
            for _ in range(video_factor):
                video_indices.extend(range(len_video))
            video_indices = video_indices[:max_len]

            # Combine robot and video data
            combined_indices = []
            for r_i in robot_indices:
                combined_indices.append((True, r_i))
            for v_i in video_indices:
                combined_indices.append((False, v_i))
            self.index_map = combined_indices  # list of (bool, idx)
        else:
            # Just combine the two datasets
            combined_indices = []
            for r_i in range(len(self.robot_dataset)):
                combined_indices.append((True, r_i))
            for v_i in range(len(self.video_dataset)):
                combined_indices.append((False, v_i))
            self.index_map = combined_indices

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        is_robot, dataset_idx = self.index_map[idx]
        if is_robot:
            data = self.robot_dataset[dataset_idx]
            data["action_mask"] = torch.tensor(1, dtype=torch.bool)
        else:
            data = self.video_dataset[dataset_idx]
            data["action_mask"] = torch.tensor(0, dtype=torch.bool)
        return data


def make_robot_video_mixture_dataset(
    robot_train_val_sets: tuple[Dataset, Dataset],
    video_train_val_sets: tuple[Dataset, Dataset],
    balance_datasets: bool = False,
    **kwargs,
):
    """
    Combine robot and video datasets into a mixture dataset.

    This function merges the training sets from both robot and video datasets to create a single training set,
    while using the robot dataset's validation set for evaluation. Additional keyword arguments (kwargs) are
    captured for compatibility with other functions in the codebase.
    """
    robot_train_set, robot_val_set = robot_train_val_sets
    video_train_set, video_val_set = video_train_val_sets
    train_set = RobotVideoMixtureDataset(
        robot_train_set, video_train_set, balance_datasets
    )
    return train_set, robot_val_set
