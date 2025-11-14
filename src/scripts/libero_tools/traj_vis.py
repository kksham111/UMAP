"""
轨迹可视化工具：对3D轨迹数据进行2D和3D可视化。

使用方式：
# 基本使用（恢复帧并可视化）
python traj_vis.py \
  --input_dir /mnt/homes/jialin-ldap/UMAP/workspace/output/libero_3dtraj/libero_90_test \
  --task_name KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo \
  --demo_idx 0 \
  --visualize

# 在视频第一帧上标记所有追踪点（每个点上方显示点的索引数字）
python traj_vis.py \
  --input_dir /mnt/homes/jialin-ldap/UMAP/workspace/output/libero_3dtraj/libero_90_test \
  --task_name KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo \
  --demo_idx 0 \
  --mark_video_frame \
  --video_dir /mnt/homes/jialin-ldap/UMAP/data/LIBERO_mod/libero_video/libero_90_test

# 为每个追踪点生成独立的3D轨迹可视化图
python traj_vis.py \
  --input_dir /mnt/homes/jialin-ldap/UMAP/workspace/output/libero_3dtraj/libero_90_test \
  --task_name KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo \
  --demo_idx 0 \
  --visualize_3d_per_point
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from typing import Optional


def _normalize_coords_like_tapip3d(coords: np.ndarray, extrinsics: Optional[np.ndarray]) -> np.ndarray:
    """
    按 tapip3d 方式做首帧相机归一化：
    - extrinsics 为 W2C（世界到相机）
    - first_frame_inv = inv(extrinsics[0]) -> C2W_0
    - X_norm = first_frame_inv @ [X_world; 1]
    """
    if extrinsics is None:
        return coords
    if coords.ndim != 3 or coords.shape[-1] != 3:
        return coords
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        return coords
    try:
        first_frame_inv = np.linalg.inv(extrinsics[0])  # C2W_0
    except Exception:
        return coords

    T, N, _ = coords.shape
    out = np.empty_like(coords)
    ones = np.ones((N, 1), dtype=coords.dtype)
    for t in range(T):
        homo = np.concatenate([coords[t], ones], axis=1)        # (N, 4)
        transformed = (first_frame_inv @ homo.T).T              # (N, 4)
        out[t] = transformed[:, :3]
    return out


def visualize_coords(original_coords, restored_coords, frame_stride, save_path=None, extrinsics: Optional[np.ndarray] = None):
    """
    可视化原始和恢复后的坐标轨迹。
    
    Args:
        original_coords: 原始跳帧后的坐标 (T, N, 3)
        restored_coords: 恢复后的坐标 (T_restored, N, 3)
        frame_stride: 跳帧间隔
        save_path: 保存路径（可选）
        extrinsics: 外参矩阵（用于归一化）
    """
    # 首帧相机归一化
    norm_original = _normalize_coords_like_tapip3d(original_coords, extrinsics)
    norm_restored = _normalize_coords_like_tapip3d(restored_coords, extrinsics)

    # 选择几个点进行可视化（如果点太多，只显示前几个）
    N_points = min(5, norm_original.shape[1])
    
    fig = plt.figure(figsize=(16, 6))
    
    # 子图1：原始跳帧后的轨迹（2D投影，x-y平面）
    ax1 = fig.add_subplot(131)
    for i in range(N_points):
        ax1.plot(norm_original[:, i, 0], norm_original[:, i, 1], 
                'o-', label=f'Point {i}', alpha=0.7, markersize=4)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Original (after stride={frame_stride})\nT={norm_original.shape[0]} frames')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 子图2：恢复后的轨迹（2D投影，x-y平面）
    ax2 = fig.add_subplot(132)
    for i in range(N_points):
        ax2.plot(norm_restored[:, i, 0], norm_restored[:, i, 1], 
                'o-', label=f'Point {i}', alpha=0.7, markersize=2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Restored (interpolated)\nT={norm_restored.shape[0]} frames')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 子图3：3D轨迹可视化（只显示第一个点，带颜色渐变）
    ax3 = fig.add_subplot(133, projection='3d')
    point_idx = 327
    # 获取轨迹数据
    traj = norm_restored[:, point_idx, :]  # (T, 3)
    T_total = traj.shape[0]
    
    # 创建颜色映射（从蓝色到红色，表示时间进程）
    colors = plt.cm.viridis(np.linspace(0, 1, T_total))
    
    # 绘制轨迹线段，每段用不同颜色
    for i in range(T_total - 1):
        ax3.plot([traj[i, 0], traj[i+1, 0]], 
                [traj[i, 1], traj[i+1, 1]], 
                [traj[i, 2], traj[i+1, 2]],
                color=colors[i], alpha=0.7, linewidth=1.5)
    
    # 用散点图显示所有点，颜色表示时间
    scatter = ax3.scatter(traj[:, 0], traj[:, 1], traj[:, 2],
                         c=np.arange(T_total), cmap='viridis', 
                         s=20, alpha=0.6, edgecolors='none')
    
    # 标记第一帧（绿色，小标记）
    ax3.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
               c='green', s=50, marker='o', 
               label='First frame', zorder=10, edgecolors='black', linewidths=2)
    
    # 标记最后一帧（红色，大标记）
    ax3.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
               c='red', s=200, marker='s', 
               label='Last frame', zorder=10, edgecolors='black', linewidths=2)
    
    # 标记原始帧的位置（如果frame_stride > 1）
    if frame_stride > 1:
        orig_indices = np.arange(0, T_total, frame_stride)
        ax3.scatter(traj[orig_indices, 0],
                   traj[orig_indices, 1],
                   traj[orig_indices, 2],
                   c='orange', s=80, marker='^', 
                   label='Original frames', zorder=8, alpha=0.8, edgecolors='black', linewidths=1)
    
    # 添加颜色条显示时间轴
    cbar = plt.colorbar(scatter, ax=ax3, pad=0.1, shrink=0.8)
    cbar.set_label('Frame Index', rotation=270, labelpad=15)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'3D Trajectory (Point {point_idx})\nColor: Time progression')
    ax3.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def visualize_track_point_on_video_frame(video_path, npz_path, point_indices=None, frame_idx=0, save_path=None):
    """
    在视频的第一帧上标记多个追踪点的位置。
    
    Args:
        video_path: 视频文件路径
        npz_path: npz文件路径（包含tracks_2d数据）
        point_indices: 要标记的追踪点索引列表（如果为None，则标记所有点）
        frame_idx: 要显示的帧索引（默认0，即第一帧）
        save_path: 保存路径（可选）
    
    Returns:
        np.ndarray: 标记后的图像
    """
    # 读取npz文件获取tracks_2d和视频帧
    try:
        data = np.load(npz_path)
    except FileNotFoundError:
        raise ValueError(f"无法找到npz文件: {npz_path}")

    if 'tracks_2d' not in data:
        print(f"警告: npz文件中没有找到 'tracks_2d' 键")
        return None
    if 'video' not in data:
        print(f"警告: npz文件中没有找到 'video' 键，无法进行标记")
        return None
        
    tracks_2d = data['tracks_2d']  # shape: (T, N, 2)
    video_frames = data['video'] # shape: (T, C, H, W) or (T, H, W, C)

    if frame_idx >= tracks_2d.shape[0]:
        print(f"警告: 帧索引 {frame_idx} 超出 'tracks_2d' 范围（总共 {tracks_2d.shape[0]} 帧）")
        return None
    if frame_idx >= video_frames.shape[0]:
        print(f"警告: 帧索引 {frame_idx} 超出 'video' 范围（总共 {video_frames.shape[0]} 帧）")
        return None

    # 获取指定帧并进行格式转换
    frame = video_frames[frame_idx]
    # 视频格式可能是 (C, H, W) 或 (H, W, C)
    if frame.shape[0] == 3 and len(frame.shape) == 3: # (C, H, W)
        frame = np.transpose(frame, (1, 2, 0)) # -> (H, W, C)
    
    # 检查是否为灰度图
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        frame_rgb = frame.copy()

    # 确定要标记的点索引
    if point_indices is None:
        # 如果未指定，标记所有点
        point_indices = list(range(tracks_2d.shape[1]))
    elif isinstance(point_indices, int):
        point_indices = [point_indices]
    
    # 创建图像副本用于绘制
    frame_marked = frame_rgb.copy()
    h, w = frame_rgb.shape[:2]
    
    # 定义颜色列表（HSV颜色空间，循环使用）
    def get_color_for_point(idx, total_points):
        """根据点的索引生成颜色"""
        hue = int(180 * idx / max(total_points, 1)) % 180
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        # 转换为RGB（因为frame_marked是RGB格式）
        return tuple(int(c) for c in color_bgr[::-1])  # BGR转RGB
    
    # 标记所有点
    valid_points = []
    for point_idx in point_indices:
        if point_idx >= tracks_2d.shape[1]:
            print(f"警告: 追踪点索引 {point_idx} 超出范围（总共 {tracks_2d.shape[1]} 个点），跳过")
            continue
        
        # 获取追踪点在指定帧的2D坐标
        point_2d = tracks_2d[frame_idx, point_idx, :]  # (2,)
        x, y = int(point_2d[0]), int(point_2d[1])
        print(f'第{frame_idx}帧，点{point_idx}的坐标为({x}, {y})')
                
        # 获取该点的颜色
        color = get_color_for_point(point_idx, tracks_2d.shape[1])
        valid_points.append((point_idx, x, y, color))
    
    # 绘制所有有效的点
    # 根据点的数量动态调整大小
    num_points = len(valid_points)
    circle_radius = 2 # 稍微缩小点
    border_width = 1
    font_scale = 0.25  # 显著缩小字体
    text_thickness_outline = 1 # 描边也变细
    text_thickness_main = 1

    # 绘制所有有效的点
    for point_idx, x, y, color in valid_points:
        # 绘制追踪点（圆圈）
        cv2.circle(frame_marked, (x, y), circle_radius, color, -1)  # 填充圆圈
        if border_width > 0:
            cv2.circle(frame_marked, (x, y), circle_radius, (0, 0, 0), border_width)  # 黑色边框
        
        # 添加文字标签：显示点的索引数字
        label = str(point_idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 文字位置在点的右侧，并进行垂直对齐
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness_main)
        text_x = x + circle_radius + 3
        text_y = y + text_height // 2
        
        # 绘制文字 (白色)
        cv2.putText(frame_marked, label, (text_x, text_y),
                   font, font_scale, (255, 255, 255), text_thickness_main, cv2.LINE_AA)
    
    # 显示或保存
    if point_indices is None or len(point_indices) == tracks_2d.shape[1]:
        title = f'Video Frame {frame_idx} - All Track Points ({len(valid_points)} points)'
    else:
        title = f'Video Frame {frame_idx} - Track Points {point_indices[:min(10, len(valid_points))]}{"..." if len(valid_points) > 10 else ""}'
    
    if save_path:
        # 保存为RGB格式（matplotlib保存需要RGB）
        plt.figure(figsize=(12, 8))
        plt.imshow(frame_marked)
        plt.axis('off')
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"标记后的图像已保存到: {save_path} (标记了 {len(valid_points)} 个点)")
    else:
        plt.figure(figsize=(12, 8))
        plt.imshow(frame_marked)
        plt.axis('off')
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    return frame_marked


def visualize_3d_trajectory_per_point(restored_coords, frame_stride, point_indices=None, save_dir=None, extrinsics: Optional[np.ndarray] = None):
    """
    为每个追踪点生成独立的3D轨迹可视化图。
    
    Args:
        restored_coords: 恢复后的坐标 (T, N, 3)
        frame_stride: 跳帧间隔
        point_indices: 要可视化的点索引列表（如果为None，则使用默认点）
        save_dir: 保存目录（可选）
        extrinsics: 外参矩阵（用于归一化）
    
    Returns:
        list: 保存的文件路径列表
    """
    norm_restored = _normalize_coords_like_tapip3d(restored_coords, extrinsics)
    T_total, N_points, _ = norm_restored.shape

    if point_indices is None:
        point_indices = [297, 327]
    
    saved_paths = []
    
    for point_idx in point_indices:
        if point_idx >= N_points:
            print(f"警告: 点索引 {point_idx} 超出范围（总共 {N_points} 个点），跳过")
            continue
        
        # 获取该点的轨迹数据
        traj = norm_restored[:, point_idx, :]  # (T, 3)
        
        # 创建独立的图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建颜色映射（viridis colormap，从紫色到黄色）
        colors = plt.cm.viridis(np.linspace(0, 1, T_total))
        
        # 绘制轨迹线段，每段用不同颜色
        for i in range(T_total - 1):
            ax.plot([traj[i, 0], traj[i+1, 0]], 
                   [traj[i, 1], traj[i+1, 1]], 
                   [traj[i, 2], traj[i+1, 2]],
                   color=colors[i], alpha=0.7, linewidth=2)
        
        # 用散点图显示所有点，颜色表示时间
        scatter = ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2],
                            c=np.arange(T_total), cmap='viridis', 
                            s=30, alpha=0.7, edgecolors='none')
        
        # 标记第一帧（绿色，小标记）
        ax.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
                  c='green', s=50, marker='o', 
                  label='First frame', zorder=10, edgecolors='black', linewidths=2)
        
        # 标记最后一帧（红色，大标记）
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                  c='red', s=50, marker='s', 
                  label='Last frame', zorder=10, edgecolors='black', linewidths=3)
        
        # 标记原始帧的位置（如果frame_stride > 1）
        if frame_stride > 1:
            orig_indices = np.arange(0, T_total, frame_stride)
            ax.scatter(traj[orig_indices, 0],
                      traj[orig_indices, 1],
                      traj[orig_indices, 2],
                      c='orange', s=100, marker='^', 
                      label='Original frames', zorder=8, alpha=0.9, edgecolors='black', linewidths=2)
        
        # 添加颜色条显示时间轴
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Frame Index', rotation=270, labelpad=20, fontsize=12)
        
        # 设置坐标轴标签
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'3D Trajectory - Point {point_idx}\nTotal Frames: {T_total}, Frame Stride: {frame_stride}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_dir:
            save_path = Path(save_dir) / f"point_{point_idx}_3d_trajectory.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_paths.append(str(save_path))
            print(f"  Point {point_idx} 3D轨迹图已保存: {save_path}")
        else:
            plt.show()
            plt.close()
    
    return saved_paths


def process_single_npz_for_vis(npz_path, visualize=False, visualize_3d_per_point=False, video_dir=None, mark_video_frame=False):
    """
    处理单个npz文件进行可视化。
    
    Args:
        npz_path: npz文件路径
        visualize: 是否进行2D可视化
        visualize_3d_per_point: 是否为每个点生成3D轨迹图
        video_dir: 视频文件所在目录（用于标记视频帧）
        mark_video_frame: 是否标记视频帧
    """
    print(f"\n处理文件: {npz_path}")
    
    # 加载数据
    data = np.load(npz_path)
    
    if 'coords' not in data:
        print(f"  警告: 文件中没有找到 'coords' 键")
        return
    
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
    
    print(f"  coords shape: {coords.shape}")
    print(f"  frame_stride: {frame_stride}")
    
    # 如果需要可视化，先尝试恢复帧（如果需要）
    if visualize or visualize_3d_per_point:
        # 尝试从restore_frames_from_stride导入恢复函数
        try:
            from restore_frames_from_stride import restore_coords_with_stride
            restored_coords = restore_coords_with_stride(coords, frame_stride)
        except ImportError:
            print(f"  警告: 无法导入 restore_coords_with_stride，使用原始coords")
            restored_coords = coords
        
        if visualize:
            # 创建可视化保存路径
            save_dir = Path(npz_path).parent
            save_path = save_dir / f"{Path(npz_path).stem}_restored_vis.png"
            visualize_coords(coords, restored_coords, frame_stride, save_path=str(save_path), extrinsics=extrinsics)
        
        # 为每个点生成独立的3D轨迹图
        if visualize_3d_per_point:
            save_dir = Path(npz_path).parent
            visualize_3d_trajectory_per_point(
                restored_coords, 
                frame_stride, 
                save_dir=save_dir,
                extrinsics=extrinsics,
            )
    
    # 如果启用了视频帧标记
    if mark_video_frame and video_dir:
        video_dir_path = Path(video_dir)
        npz_file = Path(npz_path)
        
        # 从npz文件路径推断任务名称和demo名称
        task_name = npz_file.parent.name
        demo_name = npz_file.stem.replace('_data', '')
        
        # 尝试找到对应的视频文件
        video_file = video_dir_path / task_name / f"{demo_name}.mp4"
        
        if video_file.exists():
            try:
                # 创建保存路径
                save_dir = npz_file.parent
                save_path = save_dir / f"{demo_name}_frame0_all_points_marked.png"
                visualize_track_point_on_video_frame(
                    video_path=str(video_file),
                    npz_path=str(npz_file),
                    point_indices=None,  # None表示标记所有点
                    frame_idx=0,
                    save_path=str(save_path)
                )
            except Exception as e:
                print(f"  警告: 标记视频帧时出错: {e}")
        else:
            print(f"  提示: 未找到对应的视频文件: {video_file}")


def process_task_directory_for_vis(input_dir, task_name=None, demo_idx=None, visualize=False, video_dir=None, mark_video_frame=False, visualize_3d_per_point=False):
    """
    处理任务目录下的所有npz文件进行可视化。
    
    Args:
        input_dir: 输入目录
        task_name: 任务名称（可选，如果指定则只处理该任务）
        demo_idx: demo索引（可选，如果指定则只处理该demo）
        visualize: 是否进行2D可视化
        video_dir: 视频文件所在目录
        mark_video_frame: 是否标记视频帧
        visualize_3d_per_point: 是否为每个点生成3D轨迹图
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
                process_single_npz_for_vis(
                    npz_file, 
                    visualize=visualize, 
                    visualize_3d_per_point=visualize_3d_per_point,
                    video_dir=video_dir,
                    mark_video_frame=mark_video_frame
                )
            except Exception as e:
                print(f"  错误: 处理 {npz_file.name} 时出错: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="轨迹可视化工具：对3D轨迹数据进行2D和3D可视化"
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
        "--visualize",
        action="store_true",
        help="是否进行2D可视化"
    )
    parser.add_argument(
        "--mark_video_frame",
        action="store_true",
        help="是否在视频第一帧上标记所有追踪点"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/mnt/homes/jialin-ldap/UMAP/data/LIBERO_mod/libero_video/libero_90_test",
        help="视频文件所在目录（用于标记视频帧）"
    )
    parser.add_argument(
        "--visualize_3d_per_point",
        action="store_true",
        help="是否为每个追踪点生成独立的3D轨迹可视化图"
    )
    
    args = parser.parse_args()
    
    # 至少需要选择一个可视化选项
    if not (args.visualize or args.mark_video_frame or args.visualize_3d_per_point):
        print("警告: 请至少选择一个可视化选项 (--visualize, --mark_video_frame, 或 --visualize_3d_per_point)")
        return
    
    process_task_directory_for_vis(
        args.input_dir,
        task_name=args.task_name,
        demo_idx=args.demo_idx,
        visualize=args.visualize,
        video_dir=args.video_dir if args.mark_video_frame else None,
        mark_video_frame=args.mark_video_frame,
        visualize_3d_per_point=args.visualize_3d_per_point
    )
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

