# Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets

####  [[Website]](https://weirdlabuw.github.io/uwm/) [[Paper]](https://arxiv.org/abs/2504.02792) [[Talk]](https://www.youtube.com/watch?v=WwPRxBbZ4kw)

[Chuning Zhu<sup>1</sup>](https://homes.cs.washington.edu/~zchuning/), [Raymond Yu<sup>1</sup>](https://raymondyu5.github.io/), [Siyuan Feng<sup>2</sup>](https://www.cs.cmu.edu/~sfeng/), [Benjamin Burchfiel<sup>2</sup>](https://scholar.google.com/citations?user=eGoTK1YAAAAJ&hl=en), [Paarth Shah<sup>2</sup>](https://www.paarthshah.me/about), [Abhishek Gupta<sup>1</sup>](https://homes.cs.washington.edu/~abhgupta/)<br/>

<sup>1</sup>University of Washington <sup>2</sup>Toyota Research Institute

This repository provides a PyTorch implementation of Unified World Model (UWM). UWM combines action diffusion and video diffusion to enable scalable pretraining on large, heterogeneous robotics datasets.


## Code structure
* `configs`: Configuration files for pretraining and finetuning experiments.
* `datasets`: Dataset wrappers for DROID, Robomimic, and LIBERO. We standardize all datasets using compressed [Zarr](https://zarr.readthedocs.io/en/stable/) buffers.
* `environments`: Interface wrappers for Robomimic and LIBERO environments.
* `experiments`: Training and evaluation scripts.
* `models`: Model definitions for UWM and baselines.
* `scripts`: Bash scripts for running DROID experiments.


## Setup
Install the package via
```
pip install -e .
``` 
> Note: if you encounter issues using tensorflow-dataset with DROID, consider installing tensorflow-dataset from [source](https://github.com/tensorflow/datasets).

## Robomimic Experiments
To run a Robomimic single-task experiment,
1. Install the [Robomimic](https://github.com/ARISE-Initiative/robomimic) dataset.
2. Update `hdf5_path` and `buffer_path` in the config (e.g., `configs/dataset/robomimic_cap_ph.yaml`).
3. Run:
```
python experiments/uwm/train_robomimic.py --config_name train_uwm_robomimic.yaml dataset=robomimic_can_ph exp_id=singletask
```
This command will generate a Zarr compressed buffer at the `buffer_path` specified in the config file.

## LIBERO Experiments
The LIBERO experiments share most infrastructure with the Robomimic experiments. 

### Pretraining
To pretrain a UWM on LIBERO-90,
1. Install the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) dataset.
2. Update `hdf5_path` and `buffer_path` in `configs/dataset/libero_90.yaml`.
3. Run:
```
python experiments/uwm/train_robomimic.py --config_name train_uwm_robomimic.yaml dataset=libero_90 exp_id=pretrain
```

### Finetuning
To finetune a pretrained UWM on a downstream LIBERO task (e.g., Book-Caddy),
1. Update `hdf5_path` and `buffer_path` in `configs/dataset/libero_book_caddy.yaml`.
2. Run:
```
python experiments/uwm/train_robomimic.py --config-name finetune_uwm_robomimic.yaml dataset=libero_book_caddy exp_id=finetune pretrain_checkpoint_path="logdir/uwm/libero_90/pretrain/0/models.pt"
```

We release the pretrained LIBERO-90 checkpoint [here](https://drive.google.com/drive/folders/1M4AuVLMRpSwOf_YAp56bV9AqyZI9ul6g?usp=sharing). You can download and directly finetune from this checkpoint.

## DROID Experiments
We provide shell scripts for DROID pretraining / cotraining / finetuning experiments in the `scripts` directory. Each script runs a dataset conversion pipeline to create a Zarr buffer for the corresponding DROID TFDS dataset and then launches training.

### Pretraining
To launch a DROID pretraining experiment, 
1. Install the [DROID](https://droid-dataset.github.io/) dataset
2. Update `DATA_DIR` and `BUFFER_PATH` in `scripts/launch_droid_pretrain.sh`
3. Run:
```
source scripts/launch_droid_pretrain.sh
```

### Cotraining
To launch a video cotraining experiment,
1. Install the [DROID](https://droid-dataset.github.io/) dataset
2. Update `DATA_DIR`, `ROBOT_BUFFER_PATH`, and `VIDEO_BUFFER_PATH` in `scripts/launch_droid_cotrain.sh`
3. Run:
```
source scripts/launch_droid_cotrain.sh
```

### Finetuning
To fineune a pretrained model to a downstream task, 
1. Collect demonstrations using the DROID interface
2. Convert them into a TFDS dataset (via this [pipeline](https://github.com/kpertsch/droid_dataset_builder))
3. Modify and run:
```
source scripts/launch_droid_finetune.sh
```

We release the pretrained and cotrained DROID UWM checkpoints [here](https://drive.google.com/drive/folders/1M4AuVLMRpSwOf_YAp56bV9AqyZI9ul6g?usp=sharing). You can download and directly finetune from these checkpoints.

## Bibtex
If you find this code useful, please cite:

```
@inproceedings{zhu2025uwm,
    author    = {Zhu, Chuning and Yu, Raymond and Feng, Siyuan and Burchfiel, Benjamin and Shah, Paarth and Gupta, Abhishek},
    title     = {Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets},
    booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
    year      = {2025},
}
```