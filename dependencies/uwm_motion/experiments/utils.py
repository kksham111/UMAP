import json
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

DATA_TYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_wandb(config, job_type):
    """Initialize WANDB logging. Only use in main process.

    Args:
        config: config dictionary
        job_type: "train" or "eval"
    """
    run_id_path = os.path.join(config.logdir, f"run_id_{job_type}.json")
    if config.resume and os.path.exists(run_id_path):
        # Load WANDB run ID from log directory
        with open(run_id_path, "r") as f:
            run_id = json.load(f)["run_id"]
    else:
        # Generate new WANDB run ID
        run_id = wandb.util.generate_id()
        with open(run_id_path, "w") as f:
            json.dump({"run_id": run_id}, f)

    wandb.init(
        project="video-action-learning",
        job_type=job_type,
        group=config.algo,
        name="_".join([config.exp_id, str(config.seed)]),
        config=OmegaConf.to_container(config, resolve=True),
        resume=config.resume,
        id=run_id,
    )


def init_distributed(rank, world_size):
    """Initialize distributed training and set visible device.

    Args:
        rank: unique identifier of each process
        world_size: total number of processes
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "25678")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=3600),
    )


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def soft_update(target, source, tau):
    """Soft update target model with source model."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class FreezeParameters:
    def __init__(self, params):
        self.params = params
        self.param_states = [p.requires_grad for p in self.params]

    def __enter__(self):
        for param in self.params:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.params):
            param.requires_grad = self.param_states[i]
