# Cache dataset
DATA_NAME=stack_red_bowl_on_blue_bowl
DATA_DIR="/gscratch/weirdlab/zchuning/data/"
BUFFER_PATH="/tmp/weirdlab/zchuning/data/droid/buffer_$DATA_NAME.zarr"
if [ ! -d $BUFFER_PATH ]; then
  # Cache dataset
  echo "Caching dataset..."
  python datasets/droid/convert_dataset_zarr.py --data_name $DATA_NAME --data_dir $DATA_DIR --buffer_path $BUFFER_PATH --num_episodes 2000 --num_workers 8
fi

# UWM 
PRETRAIN_CHECKPOINT_PATH="/gscratch/weirdlab/zchuning/video-action-learning/logdir/uwm/droid/benchmark/0/models.pt"
python experiments/uwm/train.py --config-name finetune_uwm.yaml dataset=droid exp_id=finetune_benchmark \
  dataset.name=droid_$DATA_NAME \
  dataset.buffer_path=$BUFFER_PATH \
  pretrain_checkpoint_path=$PRETRAIN_CHECKPOINT_PATH

# UWM cotrained
# PRETRAIN_CHECKPOINT_PATH="/gscratch/weirdlab/zchuning/video-action-learning/logdir/uwm/droid_mixture/benchmark_cotrain/0/models.pt"
# python experiments/uwm/finetune.py dataset=droid exp_id=finetune_benchmark_cotrain \
#   dataset.name=droid_$DATA_NAME \
#   dataset.buffer_path=$BUFFER_PATH \
#   pretrain_checkpoint_path=$PRETRAIN_CHECKPOINT_PATH

# DP
# PRETRAIN_CHECKPOINT_PATH="/gscratch/weirdlab/zchuning/video-action-learning/logdir/dp/droid/benchmark/0/models.pt"
# python experiments/dp/finetune.py dataset=droid exp_id=finetune_benchmark \
#   dataset.name=droid_$DATA_NAME \
#   dataset.buffer_path=$BUFFER_PATH \
#   pretrain_checkpoint_path=$PRETRAIN_CHECKPOINT_PATH

# GR1
# PRETRAIN_CHECKPOINT_PATH="/gscratch/weirdlab/zchuning/video-action-learning/logdir/gr1/droid/benchmark/0/models.pt"
# python experiments/gr1/finetune.py dataset=droid exp_id=finetune_benchmark \
#   dataset.name=droid_$DATA_NAME \
#   dataset.buffer_path=$BUFFER_PATH \
#   pretrain_checkpoint_path=$PRETRAIN_CHECKPOINT_PATH

# GR1 cotrained
# PRETRAIN_CHECKPOINT_PATH="/gscratch/weirdlab/zchuning/video-action-learning/logdir/gr1/droid_mixture/benchmark_cotrain/0/models.pt"
# python experiments/gr1/finetune.py dataset=droid exp_id=finetune_benchmark_cotrain \
#   dataset.name=droid_$DATA_NAME \
#   dataset.buffer_path=$BUFFER_PATH \
#   pretrain_checkpoint_path=$PRETRAIN_CHECKPOINT_PATH

# PAD
# PRETRAIN_CHECKPOINT_PATH="/gscratch/weirdlab/zchuning/video-action-learning/logdir/pad/droid/benchmark/0/models.pt"
# python experiments/pad/finetune.py dataset=droid exp_id=finetune_benchmark \
#   dataset.name=droid_$DATA_NAME \
#   dataset.buffer_path=$BUFFER_PATH \
#   pretrain_checkpoint_path=$PRETRAIN_CHECKPOINT_PATH


# PAD cotrained
# PRETRAIN_CHECKPOINT_PATH="/gscratch/weirdlab/zchuning/video-action-learning/logdir/pad/droid_mixture/benchmark_cotrain/0/models.pt"
# python experiments/pad/finetune.py dataset=droid exp_id=finetune_benchmark_cotrain \
#   dataset.name=droid_$DATA_NAME \
#   dataset.buffer_path=$BUFFER_PATH \
#   pretrain_checkpoint_path=$PRETRAIN_CHECKPOINT_PATH