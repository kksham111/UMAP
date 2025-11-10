import argparse
import h5py
import imageio
import numpy as np
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="将 LIBERO HDF5 演示转换为 MP4")
    parser.add_argument("--hdf5_root", type=str, default=None, help="LIBERO 数据目录（与 --hdf5_file 互斥）")
    parser.add_argument("--hdf5_file", type=str, default=None, help="单个 HDF5 文件路径（与 --hdf5_root 互斥）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 MP4 目录")
    parser.add_argument("--fps", type=int, default=10, help="输出视频帧率")
    parser.add_argument(
        "--obs_keys",
        type=str,
        nargs="+",
        default=["agentview_rgb"],
        help="需要导出的视频观测键（agentview_rgb 和 eye_in_hand_rgb）",
    )
    parser.add_argument("--max_files", type=int, default=None,
                        help="最多转换的 HDF5 文件数量（默认全部转换，仅对 --hdf5_root 模式有效）")
    return parser.parse_args()

def save_video(frames: np.ndarray, out_path: Path, fps: int):
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{out_path} 的帧数据形状异常: {frames.shape}")

    # 确保帧数据是uint8类型（0-255范围）
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用imageio保存视频，FPS设置更可靠
    # imageio可以直接处理RGB格式的numpy数组
    with imageio.get_writer(str(out_path), fps=fps, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)

def process_hdf5(hdf5_path: Path, output_root: Path, fps: int, obs_keys=None):
    with h5py.File(hdf5_path, "r") as f:
        demos_group = f["data"]
        demo_names = sorted(demos_group.keys())
        for demo_name in tqdm(demo_names, desc=f"Processing {hdf5_path.name}"):
            demo = demos_group[demo_name]
            obs_group = demo["obs"]
            keys_to_dump = obs_keys or []
            for key in keys_to_dump:
                if key not in obs_group:
                    continue
                frames = obs_group[key][:]  # [T, H, W, C]
                
                # Create a directory for the task (named after the HDF5 file)
                task_dir = output_root / f"{hdf5_path.stem}"
                task_dir.mkdir(parents=True, exist_ok=True)
                
                # Save the entire demo as a single video file, e.g., demo_0.mp4
                filename = f"{demo_name}.mp4"
                out_path = task_dir / filename
                save_video(frames, out_path, fps)

def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    
    # 互斥检查：不能同时指定 hdf5_root 和 hdf5_file
    if args.hdf5_root is not None and args.hdf5_file is not None:
        raise ValueError("错误：不能同时指定 --hdf5_root 和 --hdf5_file，请只使用其中一个参数")
    
    # 至少需要指定一个输入源
    if args.hdf5_root is None and args.hdf5_file is None:
        raise ValueError("错误：必须指定 --hdf5_root 或 --hdf5_file 其中之一")
    
    # 单文件模式
    if args.hdf5_file is not None:
        hdf5_path = Path(args.hdf5_file)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"错误：找不到文件 {hdf5_path}")
        if not hdf5_path.is_file():
            raise ValueError(f"错误：{hdf5_path} 不是一个文件")
        if args.max_files is not None:
            print("警告：--max_files 参数在单文件模式下无效，已忽略")
        print(f"处理单个 HDF5 文件: {hdf5_path}")
        process_hdf5(hdf5_path, output_root, args.fps, args.obs_keys)
        print("全部完成")
        return
    
    # 目录模式（原有逻辑）
    root = Path(args.hdf5_root)
    if not root.exists():
        raise FileNotFoundError(f"错误：找不到目录 {root}")
    if not root.is_dir():
        raise ValueError(f"错误：{root} 不是一个目录")
    
    hdf5_files = sorted(root.glob("**/*.hdf5"))

    if args.max_files is not None:
        hdf5_files = hdf5_files[:args.max_files]
        print(f"仅转换前 {args.max_files} 个 HDF5 文件")

    if not hdf5_files:
        print(f"未在 {root} 找到 hdf5 文件")
        return

    print(f"共找到 {len(hdf5_files)} 个 hdf5 文件")
    for hdf5_path in hdf5_files:
        process_hdf5(hdf5_path, output_root, args.fps, args.obs_keys)

    print("全部完成")

if __name__ == "__main__":
    main()
