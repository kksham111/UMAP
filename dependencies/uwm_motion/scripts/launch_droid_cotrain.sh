DATA_DIR="/gscratch/weirdlab/memmelma/data/"

# Cache robot dataset
ROBOT_BUFFER_PATH="/tmp/weirdlab/zchuning/data/droid/buffer_weird.zarr"
if [ ! -d $ROBOT_BUFFER_PATH ]; then
  # Cache dataset
  echo "Caching robot dataset..."
  python datasets/droid/convert_dataset_zarr.py --data_dir $DATA_DIR --buffer_path $ROBOT_BUFFER_PATH --num_episodes 2000 --num_workers 8 --filter_key WEIRD
fi

# Cache video dataset
VIDEO_BUFFER_PATH="/tmp/weirdlab/zchuning/data/droid/buffer_video.zarr"
if [ ! -d $VIDEO_BUFFER_PATH ]; then
  # Cache dataset
  echo "Caching video dataset..."
  python datasets/droid/convert_dataset_zarr.py --data_dir $DATA_DIR --buffer_path $VIDEO_BUFFER_PATH --num_episodes 2000 --num_workers 8 --except_key WEIRD
fi

# UWM
python experiments/uwm/train.py dataset=droid_mixture exp_id=benchmark_cotrain \
  dataset.buffer_path=$ROBOT_BUFFER_PATH \
  dataset.video_buffer_path=$VIDEO_BUFFER_PATH

# GR1
# python experiments/gr1/train.py dataset=droid_mixture exp_id=benchmark_cotrain \
#   dataset.buffer_path=$ROBOT_BUFFER_PATH \
#   dataset.video_buffer_path=$VIDEO_BUFFER_PATH

# PAD
# python experiments/pad/train.py dataset=droid_mixture exp_id=benchmark_cotrain \
#   dataset.buffer_path=$ROBOT_BUFFER_PATH \
#   dataset.video_buffer_path=$VIDEO_BUFFER_PATH
