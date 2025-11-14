"""
批量处理：为 LIBERO 数据集中的所有任务合并 HDF5 和 3D 轨迹数据。

使用方式：
python batch_merge_libero_3dtraj.py \
  --libero_root /path/to/libero_90_test \
  --traj_root /path/to/3dtraj/libero_90_test \
  --output_root /path/to/output/libero_90_test
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from natsort import natsorted
import re
from tqdm import tqdm


def extract_video_index(filename):
    """从 video_0.npz 中提取索引 0"""
    match = re.search(r'video_(\d+)', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract index from {filename}")


def load_and_concatenate_npz(traj_dir):
    """
    读取 traj_dir 中所有 video_*.npz 文件并按索引顺序拼接。
    主要目标是读取 'coords' 键，其 shape 为 (T, N, 3)。
    """
    traj_dir = Path(traj_dir)
    
    # 查找所有 video_*.npz 或 video_*_data.npz 文件
    npz_files = list(traj_dir.glob("video_*.npz"))
    if not npz_files:
        npz_files = list(traj_dir.glob("video_*_data.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No video_*.npz or video_*_data.npz files found in {traj_dir}")

    # 按文件名中的数字进行自然排序
    npz_files = natsorted(npz_files, key=lambda x: extract_video_index(x.name))
    
    arrays = []
    for npz_file in npz_files:
        data = np.load(npz_file)
        # 目标数据存储在 'coords' 键下
        if 'coords' in data.files:
            arr = data['coords']
        else:
            # 如果找不到 'coords'，则发出警告并跳过此文件
            print(f"  WARNING: 'coords' key not found in {npz_file.name}. Skipping this file.")
            continue
        
        arrays.append(arr)
    
    if not arrays:
        raise ValueError(f"No valid data with 'coords' key found in any npz files in {traj_dir}")

    # 沿第一维（时间 T）拼接
    concatenated = np.concatenate(arrays, axis=0)
    
    return concatenated


def copy_obs_recursively(src_obs_group, dst_demo):
    """
    递归复制 obs 组中的所有数据集和子组。
    """
    for key in src_obs_group.keys():
        src_item = src_obs_group[key]
        if isinstance(src_item, h5py.Dataset):
            dst_demo.create_dataset(key, data=src_item[:])
        elif isinstance(src_item, h5py.Group):
            sub_group = dst_demo.create_group(key)
            copy_obs_recursively(src_item, sub_group)


def collect_3d_trajectories(traj_root, task_name):
    """
    从 3D 轨迹目录中收集所有 demo 的 coords 数据。
    
    目录结构预期：
    traj_root/task_name/demo_0.npz 或 demo_0_data.npz
    traj_root/task_name/demo_1.npz 或 demo_1_data.npz
    ...
    
    返回格式: {
        'demo_0': coords_array (shape: T, N, 3),
        'demo_1': coords_array,
        ...
    }
    """
    traj_data = {}
    traj_root = Path(traj_root)
    task_dir = traj_root / task_name
    
    if not task_dir.exists():
        print(f"WARNING: Trajectory task directory not found: {task_dir}")
        return traj_data
    
    # 查找所有 demo_*.npz 文件（支持两种格式）
    npz_files = list(task_dir.glob("demo_*.npz"))
    
    if not npz_files:
        print(f"  WARNING: No demo_*.npz files found in {task_dir}")
        return traj_data
    
    # 排序：从文件名中提取 demo 编号
    def extract_demo_number(path):
        stem = path.stem  # 例如 'demo_0' 或 'demo_0_data'
        parts = stem.split('_')
        # 提取第一个数字部分
        for part in parts[1:]:  # 跳过 'demo'
            if part.isdigit():
                return int(part)
        return 0
    
    npz_files = natsorted(npz_files, key=lambda x: extract_demo_number(x))
    
    for npz_file in npz_files:
        # 从文件名提取 demo_key，例如 'demo_0_data.npz' -> 'demo_0'
        stem = npz_file.stem  # 例如 'demo_0' 或 'demo_0_data'
        if stem.endswith('_data'):
            demo_key = stem[:-5]  # 去掉 '_data' 后缀
        else:
            demo_key = stem  # 保持原样
        
        try:
            # 直接加载该 demo 的 npz 文件
            data = np.load(npz_file)
            if 'coords' in data.files:
                traj_array = data['coords']
                traj_data[demo_key] = traj_array
            else:
                print(f"  WARNING: 'coords' key not found in {npz_file.name}. Skipping this file.")
                continue
        except Exception as e:
            print(f"  WARNING: Could not load 3d_traj for {demo_key} in {task_name}: {e}")
            continue
    
    return traj_data


def merge_single_task(source_hdf5, traj_root, task_name, output_hdf5):
    """
    合并单个任务的 HDF5 和 3D 轨迹数据。
    分别调用两个收集函数，然后合并生成新的 HDF5。
    """
    output_hdf5 = Path(output_hdf5)
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    
    # 步骤 1: 收集 3D 轨迹数据
    traj_data = collect_3d_trajectories(traj_root, task_name)
    
    # 步骤 2: 读取 LIBERO HDF5 并同时写入新 HDF5
    with h5py.File(source_hdf5, 'r') as src_file:
        with h5py.File(output_hdf5, 'w') as dst_file:
            src_data_group = src_file['data']
            dst_data_group = dst_file.create_group('data')
            
            # 获取所有 demo 键并排序
            demo_keys = sorted(src_data_group.keys(), key=lambda x: int(x.split('_')[1]))
            
            for demo_key in demo_keys:
                src_demo = src_data_group[demo_key]
                dst_demo = dst_data_group.create_group(demo_key)
                
                # 1. 复制 obs 数据
                if 'obs' in src_demo:
                    src_obs = src_demo['obs']
                    dst_obs = dst_demo.create_group('obs')
                    copy_obs_recursively(src_obs, dst_obs)
                
                # 2. 复制 actions 数据
                if 'actions' in src_demo:
                    actions = src_demo['actions'][:]
                    dst_demo.create_dataset('actions', data=actions)
                
                # 3. 添加 traj3d 数据
                if demo_key in traj_data:
                    dst_demo.create_dataset('traj3d', data=traj_data[demo_key])
                else:
                    print(f"  WARNING: No traj3d data found for {demo_key}")


def batch_merge(libero_root, traj_root, output_root):
    """
    批量处理所有任务。
    """
    libero_root = Path(libero_root)
    traj_root = Path(traj_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    hdf5_files = natsorted(libero_root.glob("*.hdf5"))
    
    print(f"Found {len(hdf5_files)} HDF5 files in {libero_root}\n")
    
    for hdf5_file in tqdm(hdf5_files, desc="Merging tasks"):
        task_name = hdf5_file.stem
        output_hdf5 = output_root / hdf5_file.name
        
        try:
            merge_single_task(str(hdf5_file), str(traj_root), task_name, str(output_hdf5))
        except Exception as e:
            print(f"ERROR processing {task_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="批量合并 LIBERO HDF5 文件与 3D 轨迹数据"
    )
    parser.add_argument("--libero_root", type=str, required=True, help="LIBERO HDF5 文件所在目录")
    parser.add_argument("--traj_root", type=str, required=True, help="3D 轨迹数据所在根目录")
    parser.add_argument("--output_root", type=str, required=True, help="输出合并后 HDF5 文件的根目录")
    
    args = parser.parse_args()
    
    libero_root = Path(args.libero_root)
    traj_root = Path(args.traj_root)
    
    if not libero_root.exists():
        raise FileNotFoundError(f"LIBERO root not found: {libero_root}")
    if not traj_root.exists():
        raise FileNotFoundError(f"Trajectory root not found: {traj_root}")
    
    batch_merge(args.libero_root, args.traj_root, args.output_root)
    print("\nBatch merge completed!")


if __name__ == "__main__":
    main()
