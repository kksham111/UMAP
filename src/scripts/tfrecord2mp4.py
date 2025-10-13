import os
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

TFRECORD_ROOT = '/inspire/hdd/project/robot-body/public/xkwang/libero_spatial_mv/1.0.0'
OUTPUT_DIR = '/mnt/homes/jialin-ldap/UMAP/workspace/output'
MAX_FILES = int(os.environ.get('TFRECORD_MAX_FILES', '0')) or None
MAX_EXAMPLES_PER_FILE = int(os.environ.get('TFRECORD_MAX_EXAMPLES', '0')) or None

os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_tfrecord_fields(tfrecord_path, num_examples=3):
    fields = set()
    for i, record in enumerate(tf.data.TFRecordDataset(tfrecord_path)):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        fields.update(example.features.feature.keys())
        if i >= num_examples - 1:
            break
    print(f'{tfrecord_path} 字段: {sorted(fields)}')
    return fields

def parse_example(example_proto):
    return tf.train.Example.FromString(example_proto.numpy())

def extract_and_save_mp4(tfrecord_path, output_dir, max_examples=None):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    for idx, record in enumerate(tqdm(dataset, desc='Processing TFRecord')):
        example = parse_example(record)
        # 读取 steps/observation/image 字段
        if 'steps/observation/image' in example.features.feature:
            frames = example.features.feature['steps/observation/image'].bytes_list.value
            imgs = [cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR) for f in frames]
            imgs = [img for img in imgs if img is not None]
            if len(imgs) == 0:
                print(f'Example {idx} 没有有效图片帧，跳过')
                continue
            height, width, _ = imgs[0].shape
            out_path = os.path.join(output_dir, f'video_{idx:04d}.mp4')
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
            for img in imgs:
                writer.write(img)
            writer.release()
        else:
            print(f'Example {idx} 没有 "steps/observation/image" 字段，跳过')

        if max_examples is not None and idx + 1 >= max_examples:
            break


def find_tfrecord_files(root_dir):
    tfrecord_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            lower = name.lower()
            if '.tfrecord' in lower:
                tfrecord_files.append(os.path.join(dirpath, name))
    tfrecord_files.sort()
    return tfrecord_files


def process_directory(root_dir, output_root, max_files=None, max_examples=None):
    tfrecord_files = find_tfrecord_files(root_dir)
    if not tfrecord_files:
        print(f'未在 {root_dir} 找到 TFRecord 文件')
        return

    print(f'找到 {len(tfrecord_files)} 个 TFRecord 文件')
    for file_idx, tfrecord_path in enumerate(tfrecord_files):
        relative_name = Path(tfrecord_path).name
        per_file_output = os.path.join(output_root, relative_name)
        os.makedirs(per_file_output, exist_ok=True)
        print(f"\n处理 {tfrecord_path}")
        list_tfrecord_fields(tfrecord_path)
        extract_and_save_mp4(tfrecord_path, per_file_output, max_examples=max_examples)

        if max_files is not None and file_idx + 1 >= max_files:
            print(f'达到最大处理文件数量 {max_files}，提前结束')
            break


if __name__ == '__main__':
    print(f'开始遍历 {TFRECORD_ROOT} 并转换 TFRecord -> MP4')
    process_directory(
        TFRECORD_ROOT,
        OUTPUT_DIR,
        max_files=MAX_FILES,
        max_examples=MAX_EXAMPLES_PER_FILE,
    )
    print('全部完成，mp4 已输出到', OUTPUT_DIR)
