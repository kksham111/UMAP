# Cache dataset
DATA_DIR="/gscratch/weirdlab/memmelma/data/"
BUFFER_PATH="/tmp/weirdlab/zchuning/data/droid/buffer_weird.zarr"
if [ ! -d $BUFFER_PATH ]; then
  # Cache dataset
  echo "Caching dataset..."
  python datasets/droid/convert_dataset_zarr.py --data_dir $DATA_DIR --buffer_path $BUFFER_PATH --num_episodes 2000 --num_workers 8 --filter_key WEIRD
fi

# UWM
python experiments/uwm/train.py dataset=droid exp_id=benchmark dataset.buffer_path=$BUFFER_PATH

# DP
# python experiments/dp/train.py dataset=droid exp_id=benchmark dataset.buffer_path=$BUFFER_PATH

# GR1
# python experiments/gr1/train.py dataset=droid exp_id=benchmark dataset.buffer_path=$BUFFER_PATH

# PAD
# python experiments/pad/train.py dataset=droid exp_id=benchmark dataset.buffer_path=$BUFFER_PATH