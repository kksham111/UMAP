"""
恢复跳帧：根据frame_stride将coords中跳过的帧补上。

使用方式：
# 验证并匹配视频的真实帧数
python restore_frames_from_stride.py \
  --input_dir /mnt/homes/jialin-ldap/UMAP/workspace/output/libero_3dtraj/libero_90_test \
  --video_dir /mnt/homes/jialin-ldap/UMAP/data/LIBERO_mod/libero_video/libero_90_test \
  --verify_frame_count

# 保存恢复后的数据到输出目录
python restore_frames_from_stride.py \
  --input_dir /mnt/homes/jialin-ldap/UMAP/workspace/output/libero_3dtraj/libero_90_test \
  --video_dir /mnt/homes/jialin-ldap/UMAP/data/LIBERO_mod/libero_video/libero_90_test \
  --verify_frame_count \
  --output_dir /mnt/homes/jialin-ldap/UMAP/workspace/output/libero_3dtraj_restored/libero_90_test
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

def get_video_frame_count(video_path):
    """
    获取视频的总帧数。
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        int: 视频的总帧数，如果无法读取则返回None
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def restore_coords_with_stride(coords, frame_stride, target_frame_count=None):
    """
    根据frame_stride恢复跳过的帧，使用线性插值。
    
    原始数据是通过 video_tensor[::frame_stride] 跳帧得到的，
    现在需要将跳过的帧补上，使用线性插值方法。
    
    Args:
        coords: shape (T, N, 3) 的坐标数组，已经是跳帧后的数据
        frame_stride: 跳帧间隔，例如 frame_stride=2 表示每2帧取1帧
        target_frame_count: 目标帧数（如果提供，确保恢复后的帧数与此匹配）
    
    Returns:
        restored_coords: shape (T_original, N, 3) 的恢复后的坐标数组
        
    线性插值说明：
        对于frame_stride=3，coords[0]和coords[1]之间插入2帧：
        - frame 1 = coords[0] + 1/3 * (coords[1] - coords[0])
        - frame 2 = coords[0] + 2/3 * (coords[1] - coords[0])
        每个点的(x,y,z)都独立线性插值
    """
    T, N, _ = coords.shape
    
    if frame_stride == 1:
        # 没有跳帧，直接返回
        if target_frame_count is not None and T != target_frame_count:
            # 如果目标帧数不同，需要调整
            if T < target_frame_count:
                # 需要补充帧，使用最后一帧填充
                additional_frames = target_frame_count - T
                last_frame = coords[-1:].copy()
                additional_coords = np.tile(last_frame, (additional_frames, 1, 1))
                restored_coords = np.concatenate([coords, additional_coords], axis=0)
                return restored_coords
            else:
                # 需要截断
                return coords[:target_frame_count]
        return coords
    
    # 计算原始帧数
    T_original = (T - 1) * frame_stride + 1
    
    # 如果提供了目标帧数，使用目标帧数
    if target_frame_count is not None:
        T_original = target_frame_count
    
    restored_coords = np.zeros((T_original, N, 3), dtype=coords.dtype)
    
    # 填充已知的帧
    for i in range(T):
        orig_idx = i * frame_stride
        if orig_idx < T_original:
            restored_coords[orig_idx] = coords[i]
    
    # 使用线性插值填充跳过的帧
    for i in range(T - 1):
        start_idx = i * frame_stride
        end_idx = min((i + 1) * frame_stride, T_original)
        
        # 获取起始和结束帧的坐标
        start_coords = coords[i]      # shape: (N, 3)
        end_coords = coords[i + 1]    # shape: (N, 3)
        
        # 在start_idx和end_idx之间进行线性插值
        for k in range(1, end_idx - start_idx):
            # 计算插值比例
            alpha = k / frame_stride
            # 线性插值：对每个点的(x,y,z)都独立插值
            # interpolated = start + alpha * (end - start)
            restored_coords[start_idx + k] = start_coords + alpha * (end_coords - start_coords)
    
    # 处理最后一段（如果有剩余帧）
    last_known_idx = (T - 1) * frame_stride
    if last_known_idx < T_original:
        # 使用最后一个已知帧填充剩余部分（无法插值，因为没有下一帧）
        for j in range(last_known_idx + 1, T_original):
            restored_coords[j] = coords[-1]
    
    return restored_coords


def save_restored_npz(original_npz_path, restored_coords, output_dir=None):
    """
    保存恢复后的数据为npz文件，保持原始格式。
    
    Args:
        original_npz_path: 原始npz文件路径
        restored_coords: 恢复后的coords数据
        output_dir: 输出目录（如果为None，则保存到原文件同目录）
    
    Returns:
        str: 保存的文件路径
    """
    # 加载原始数据
    original_data = np.load(original_npz_path)
    
    # 准备保存的数据字典
    save_data = {}
    
    # 复制所有原始数据
    for key in original_data.files:
        if key == 'coords':
            # 使用恢复后的coords
            save_data[key] = restored_coords
        else:
            # 保持其他字段不变
            save_data[key] = original_data[key]
    
    # 确定保存路径
    original_path = Path(original_npz_path)
    if output_dir:
        output_path = Path(output_dir)
        # 保持相对目录结构
        # 例如: input/.../TASK/demo_0_data.npz -> output/.../TASK/demo_0_data.npz
        task_name = original_path.parent.name
        output_path = output_path / task_name
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / original_path.name
    else:
        # 保存到原文件同目录，文件名添加_restored后缀
        save_path = original_path.parent / f"{original_path.stem}_restored.npz"
    
    # 保存为npz文件
    np.savez_compressed(save_path, **save_data)
    print(f"  ✓ 已保存恢复后的数据: {save_path}")
    
    return str(save_path)


def process_single_npz(npz_path, video_dir=None, verify_frame_count=False, output_dir=None):
    """
    处理单个npz文件。
    
    Args:
        npz_path: npz文件路径
        video_dir: 视频文件所在目录（用于验证帧数）
        verify_frame_count: 是否验证并匹配视频帧数
        output_dir: 输出目录（如果指定，保存恢复后的npz文件）
    """
    print(f"\n处理文件: {npz_path}")
    
    # 加载数据
    data = np.load(npz_path)
    
    if 'coords' not in data:
        print(f"  警告: 文件中没有找到 'coords' 键")
        return None
    
    coords = data['coords']
    
    extrinsics = data['extrinsics'] if 'extrinsics' in data.files else None
    
    if 'frame_stride' in data:
        frame_stride_val = data['frame_stride']
        if isinstance(frame_stride_val, np.ndarray):
            frame_stride = int(frame_stride_val.item())
        else:
            frame_stride = int(frame_stride_val)
    else:
        print(f"  警告: 文件中没有找到 'frame_stride' 键，假设 frame_stride=1")
        frame_stride = 1
    
    print(f"  原始coords shape: {coords.shape}")
    print(f"  frame_stride: {frame_stride}")
    
    # 如果启用了帧数验证，尝试找到对应的视频文件
    target_frame_count = None
    if verify_frame_count and video_dir:
        video_dir_path = Path(video_dir)
        npz_file = Path(npz_path)
        
        # 从npz文件路径推断任务名称和demo名称
        # 例如: .../libero_90_test/TASK_NAME/demo_0_data.npz
        task_name = npz_file.parent.name
        demo_name = npz_file.stem.replace('_data', '')
        
        # 尝试找到对应的视频文件
        video_file = video_dir_path / task_name / f"{demo_name}.mp4"
        
        if video_file.exists():
            video_frame_count = get_video_frame_count(video_file)
            if video_frame_count is not None:
                target_frame_count = video_frame_count
                print(f"  找到对应视频: {video_file}")
                print(f"  视频总帧数: {target_frame_count}")
            else:
                print(f"  警告: 无法读取视频帧数: {video_file}")
        else:
            print(f"  提示: 未找到对应的视频文件: {video_file}")
    
    # 恢复帧
    if frame_stride > 1 or target_frame_count is not None:
        restored_coords = restore_coords_with_stride(coords, frame_stride, target_frame_count=target_frame_count)
        print(f"  恢复后coords shape: {restored_coords.shape}")
        
        # 验证帧数匹配
        if target_frame_count is not None:
            if restored_coords.shape[0] == target_frame_count:
                print(f"  ✓ 帧数匹配: {restored_coords.shape[0]} == {target_frame_count}")
            else:
                print(f"  ⚠ 帧数不匹配: {restored_coords.shape[0]} != {target_frame_count}")
    else:
        print(f"  frame_stride=1，无需恢复")
        restored_coords = coords
    
    # 保存恢复后的数据
    if output_dir:
        save_restored_npz(npz_path, restored_coords, output_dir=output_dir)
    
    return restored_coords


def process_task_directory(input_dir, task_name=None, demo_idx=None, video_dir=None, verify_frame_count=False, output_dir=None):
    """
    处理任务目录下的所有npz文件。
    
    Args:
        input_dir: 输入目录
        task_name: 任务名称（可选，如果指定则只处理该任务）
        demo_idx: demo索引（可选，如果指定则只处理该demo）
        video_dir: 视频文件所在目录（用于验证帧数）
        verify_frame_count: 是否验证帧数
        output_dir: 输出目录（保存恢复后的npz文件）
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 获取所有任务目录
    if task_name:
        task_dirs = [input_dir / task_name]
    else:
        task_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    print(f"找到 {len(task_dirs)} 个任务目录")
    
    for task_dir in task_dirs:
        print(f"\n{'='*60}")
        print(f"处理任务: {task_dir.name}")
        print(f"{'='*60}")
        
        # 查找所有demo_*_data.npz文件
        if demo_idx is not None:
            npz_files = list(task_dir.glob(f"demo_{demo_idx}_data.npz"))
        else:
            npz_files = sorted(task_dir.glob("demo_*_data.npz"), 
                             key=lambda x: int(x.stem.split('_')[1]))
        
        if not npz_files:
            print(f"  未找到npz文件")
            continue
        
        print(f"找到 {len(npz_files)} 个npz文件")
        
        # 处理每个npz文件
        for npz_file in tqdm(npz_files, desc=f"处理 {task_dir.name}"):
            try:
                restored_coords = process_single_npz(
                    npz_file, 
                    video_dir=video_dir,
                    verify_frame_count=verify_frame_count,
                    output_dir=output_dir
                )
                if restored_coords is not None:
                    pass  # 数据已在process_single_npz中保存（如果指定了output_dir）
                        
            except Exception as e:
                print(f"  错误: 处理 {npz_file.name} 时出错: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="恢复跳帧：根据frame_stride将coords中跳过的帧补上"
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/mnt/homes/jialin-ldap/UMAP/workspace/output/libero_3dtraj/libero_90_test",
        help="输入目录路径"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="任务名称（可选，如果指定则只处理该任务）"
    )
    parser.add_argument(
        "--demo_idx",
        type=int,
        default=None,
        help="demo索引（可选，如果指定则只处理该demo）"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/mnt/homes/jialin-ldap/UMAP/data/LIBERO_mod/libero_video/libero_90_test",
        help="视频文件所在目录（用于验证帧数）"
    )
    parser.add_argument(
        "--verify_frame_count",
        action="store_true",
        help="是否验证并匹配视频的真实帧数（需要提供--video_dir）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（保存恢复后的npz文件，如果不指定则不保存）"
    )
    
    args = parser.parse_args()
    
    # 如果启用了帧数验证，需要video_dir
    video_dir_for_verify = args.video_dir if args.verify_frame_count else None
    
    process_task_directory(
        args.input_dir,
        task_name=args.task_name,
        demo_idx=args.demo_idx,
        video_dir=video_dir_for_verify,
        verify_frame_count=args.verify_frame_count,
        output_dir=args.output_dir
    )
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

