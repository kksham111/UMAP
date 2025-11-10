import pycolmap
from models.SpaTrackV2.models.predictor import Predictor
import yaml
import easydict
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import io
import moviepy as mp
from models.SpaTrackV2.utils.visualizer import Visualizer
import tqdm
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from models.SpaTrackV2.models.tracker3D.spatrack_modules.utils import set_procrustes_context
import glob
from rich import print
import argparse
import decord
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri
from contextlib import nullcontext
import traceback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_type", type=str, default="RGB")
    parser.add_argument("--input_dir", type=str, default="/mnt/homes/jialin-ldap/UMAP/workspace/output/libero_videos/libero_90", help="Base directory for input videos.")
    parser.add_argument("--output_dir", type=str, default="results", help="Base output directory.")
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument("--frame_stride", type=int, default=2, help="抽帧步长")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on, e.g., cuda:4 or cpu")
    parser.add_argument("--enable_viz", action="store_true", help="启用可视化输出MP4视频")
    return parser.parse_args()

def get_depth(video_tensor, vggt4track_model, device, amp_ctx):
    """
    使用VGGT4Track模型预测RGB视频的深度图和相机参数
    
    Args:
        video_tensor: 视频张量 (N, C, H, W)
        vggt4track_model: VGGT4Track模型
        device: 设备
        amp_ctx: 自动混合精度上下文
    
    Returns:
        depth_tensor: 预测的深度图
        extrs: 外参矩阵
        intrs: 内参矩阵
        unc_metric: 深度不确定性度量
        video_tensor: 处理后的视频张量
    """
    video_tensor = preprocess_image(video_tensor)[None]
    with torch.no_grad():
        with amp_ctx:
            predictions = vggt4track_model((video_tensor.to(device))/255)
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
    
    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    video_tensor = video_tensor.squeeze()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    
    return depth_tensor, extrs, intrs, unc_metric, video_tensor

def init_visualizer(out_dir, fps=10, grayscale=True, pad_value=0, tracks_leave_trace=5):
    """
    初始化可视化器
    
    Args:
        out_dir: 输出目录
        fps: 可视化视频的帧率
        grayscale: 是否使用灰度图
        pad_value: 填充值
        tracks_leave_trace: 轨迹保留帧数
    
    Returns:
        visualizer: 可视化器对象
    """
    visualizer = Visualizer(
        save_dir=out_dir, 
        grayscale=grayscale, 
        fps=fps, 
        pad_value=pad_value, 
        tracks_leave_trace=tracks_leave_trace
    )
    return visualizer

def save_visualization(results, viser, video_name):
    """
    保存可视化结果为MP4视频
    
    Args:
        results: 推理结果字典，包含 video, track2d_pred, vis_pred
        viser: 可视化器对象
        video_name: 视频名称（用于文件名）
    
    Returns:
        str: 保存的视频文件路径，如果viser为None则返回None
    """
    if viser is None:
        return None
    
    viser.visualize(
        video=results["video"][None],
        tracks=results["track2d_pred"][None][...,:2],
        visibility=results["vis_pred"][None],
        filename=f"{video_name}_track_viz"
    )
    
    video_path = os.path.join(results["out_dir"], f"{video_name}_track_viz_pred_track.mp4")
    return video_path

def save_npz(results, save_path, frame_stride, video_fps):
    """
    保存结果为NPZ格式
    
    Args:
        results: 推理结果字典
        save_path: 保存路径
        frame_stride: 帧率采样间隔
        video_fps: 原始视频帧率
    
    Returns:
        str: 保存的NPZ文件路径
    """
    coords_world = (torch.einsum("tij,tnj->tni", results["c2w_traj"][:,:3,:3], results["track3d_pred"][:,:,:3].cpu())
                    + results["c2w_traj"][:,:3,3][:,None,:]).numpy()
    
    coords_cam = results["track3d_pred"][:,:,:3].cpu().numpy()
    
    T_frames, N_points, _ = coords_world.shape
    t_idx = np.arange(T_frames, dtype=np.int32)
    
    t_tile = np.tile(t_idx[:, None], (1, N_points))[..., None].astype(coords_world.dtype)
    coords_wt = np.concatenate([coords_world, t_tile], axis=2)
    
    vis = results["vis_pred"].cpu().numpy().astype(bool).squeeze(-1)
    coords_wt_masked = coords_wt.copy()
    coords_wt_masked[~vis] = np.nan
    
    extrinsics = torch.inverse(results["c2w_traj"]).cpu().numpy()
    intrinsics = results["intrs"].cpu().numpy()
    
    depth_save = results["point_map"][:,2,...].clone()
    depth_save[results["conf_depth"]<0.5] = 0
    depths = depth_save.cpu().numpy()
    unc_metric_np = results["conf_depth"].cpu().numpy()
    
    video_np = results["video_tensor"].cpu().numpy() / 255
    tracks_2d = results["track2d_pred"].cpu().numpy()
    visibs = results["vis_pred"].cpu().numpy()
    confs = results["conf_pred"].cpu().numpy()
    
    np.savez_compressed(
        save_path,
        coords=coords_world,
        coords_cam=coords_cam,
        coords_wt=coords_wt_masked,
        tracks_2d=tracks_2d,
        visibs=visibs,
        confs=confs,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depths=depths,
        unc_metric=unc_metric_np,
        video=video_np,
        frame_stride=np.array([frame_stride], dtype=np.int32),
        video_fps=np.array([video_fps], dtype=np.float32),
        t_idx=t_idx,
    )
    
    return save_path

def model_inference(video_path, data_dir, video_name, args, model, vggt4track_model, device, amp_ctx, output_directory_for_video):
    """
    对单个视频进行模型推理
    
    Args:
        video_path: 视频文件路径
        data_dir: 数据目录
        video_name: 视频名称
        args: 命令行参数
        model: 主跟踪模型
        vggt4track_model: VGGT4Track模型
        device: 设备
        amp_ctx: 自动混合精度上下文
        output_directory_for_video: 输出目录
    
    Returns:
        dict: 推理结果字典，如果失败返回None
    """
    frame_stride = int(args.frame_stride)
    mask_dir = os.path.join(data_dir, f"{video_name}.png")
    video_fps = float("nan")
    
    # 加载数据
    if args.data_type == "RGBD":
        npz_dir = os.path.join(data_dir, f"{video_name}.npz")
        if not os.path.exists(npz_dir):
            return None
        data_npz_load = dict(np.load(npz_dir, allow_pickle=True))
        video_tensor = data_npz_load["video"] * 255
        video_tensor = torch.from_numpy(video_tensor)
        video_tensor = video_tensor[::frame_stride]
        depth_tensor = data_npz_load["depths"]
        depth_tensor = depth_tensor[::frame_stride]
        intrs = data_npz_load["intrinsics"]
        intrs = intrs[::frame_stride]
        extrs = np.linalg.inv(data_npz_load["extrinsics"])
        extrs = extrs[::frame_stride]
        unc_metric = None
    elif args.data_type == "RGB":
        video_reader = decord.VideoReader(video_path)
        try:
            video_fps = float(video_reader.get_avg_fps())
        except Exception:
            video_fps = float("nan")
        
        video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)
        video_tensor = video_tensor[::frame_stride].float()
        
        # 使用get_depth函数获取深度图和相机参数
        depth_tensor, extrs, intrs, unc_metric, video_tensor = get_depth(video_tensor, vggt4track_model, device, amp_ctx)
        data_npz_load = {}
    else:
        return None
    
    # Mask处理
    if os.path.exists(mask_dir):
        mask = cv2.imread(mask_dir)
        mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
        mask = mask.sum(axis=-1)>0
    else:
        mask = np.ones_like(video_tensor[0,0].numpy())>0
    
    # 生成网格点
    grid_size = args.grid_size
    frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    
    if os.path.exists(mask_dir):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]
    
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    
    # 设置Procrustes上下文（用于记录不满足条件的旋转矩阵）
    log_file_path = os.path.join(output_directory_for_video, "procrustes_problematic_rotations.txt")
    set_procrustes_context(log_file=log_file_path, video_name=video_name)
    
    # 模型推理
    with amp_ctx:
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 
        
        # 缩放结果
        max_size = 336
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs[:,:2,:] = intrs[:,:2,:] * scale
            if depth_tensor is not None:
                if isinstance(depth_tensor, torch.Tensor):
                    depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
                else:
                    depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))
    
    # 准备返回结果字典
    results = {
        "c2w_traj": c2w_traj,
        "intrs": intrs,
        "point_map": point_map,
        "conf_depth": conf_depth,
        "track3d_pred": track3d_pred,
        "track2d_pred": track2d_pred,
        "vis_pred": vis_pred,
        "conf_pred": conf_pred,
        "video": video,
        "video_tensor": video_tensor,
        "depth_tensor": depth_tensor,
        "out_dir": output_directory_for_video,
        "video_fps": video_fps,
    }
    
    return results

if __name__ == "__main__":
    args = parse_args()

    # Ensure the main output directory exists for logging
    os.makedirs(args.output_dir, exist_ok=True)
    error_log_path = os.path.join(args.output_dir, "error_log.txt")

    # resolve device
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
        # set current cuda device for consistency
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        device = torch.device("cpu")
        amp_ctx = nullcontext()
    
    # --- Model Setup (do once) ---
    print("Loading models...")
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to(device)
    
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    model.spatrack.track_num = args.vo_points
    model.eval()
    model.to(device)
    print("Models loaded.")

    # --- Find and Process Videos ---
    video_paths = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.mp4"), recursive=True))
    print(f"Found {len(video_paths)} videos to process in '{args.input_dir}'.")

    for video_path in tqdm.tqdm(video_paths, desc="Processing videos"):
        try:

            data_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Create corresponding output directory by preserving the relative structure
            relative_path_structure = os.path.relpath(os.path.dirname(video_path), args.input_dir)
            output_directory_for_video = os.path.join(args.output_dir, relative_path_structure)
            os.makedirs(output_directory_for_video, exist_ok=True)

            # Define save path, e.g., .../output_dir/TASK_NAME/demo_0/video_0_data.npz
            output_filename = f"{video_name}_data.npz"
            save_path = os.path.join(output_directory_for_video, output_filename)
            if os.path.exists(save_path):
                # print(f"Output exists, skipping: {save_path}")
                continue

            # 运行模型推理
            results = model_inference(
                video_path=video_path,
                data_dir=data_dir,
                video_name=video_name,
                args=args,
                model=model,
                vggt4track_model=vggt4track_model,
                device=device,
                amp_ctx=amp_ctx,
                output_directory_for_video=output_directory_for_video
            )
            
            if results is None:
                tqdm.tqdm.write(f"Warning: Failed to process {video_path}, skipping.")
                continue
            
            # 初始化可视化器（如果启用）
            viser = None
            if args.enable_viz:
                viser = init_visualizer(
                    out_dir=output_directory_for_video,
                    fps=10,
                    grayscale=True,
                    pad_value=0,
                    tracks_leave_trace=5
                )
            
            # 保存可视化MP4（如果启用）
            if args.enable_viz:
                save_visualization(results, viser, video_name)
            
            # 保存NPZ文件
            frame_stride = int(args.frame_stride)
            if args.save_format == "zarr":
                save_zarr(results, save_path, frame_stride, results["video_fps"])
            else:
                save_npz(results, save_path, frame_stride, results["video_fps"])
            
            # 汇总有问题的视频名（如果有日志文件）
            log_file_path = os.path.join(output_directory_for_video, "procrustes_problematic_rotations.txt")
            if os.path.exists(log_file_path):
                # 提取视频名并保存到汇总文件
                summary_file = os.path.join(args.output_dir, "procrustes_problematic_videos_summary.txt")
                try:
                    with open(log_file_path, 'r') as f:
                        content = f.read()
                        # 提取所有视频名
                        import re
                        video_names = re.findall(r'=== Video: (.+?) ===', content)
                        if video_names:
                            # 读取已存在的汇总文件内容（如果存在）
                            existing_videos = set()
                            if os.path.exists(summary_file):
                                with open(summary_file, 'r') as summary_f:
                                    existing_videos = set(line.strip() for line in summary_f if line.strip())
                            
                            # 追加新的视频名
                            with open(summary_file, 'a') as summary_f:
                                for vname in video_names:
                                    if vname not in existing_videos:
                                        summary_f.write(f"{vname}\n")
                                        existing_videos.add(vname)
                except Exception as e:
                    pass  # 忽略汇总错误，不影响主流程
            
            # 关键：删除results中的GPU tensor引用，释放GPU内存
            # 注意：save_npz()中已经将数据转换为numpy，所以可以安全删除tensor
            del results
            if device.type == 'cuda':
                torch.cuda.empty_cache()  # 清空PyTorch的内存池，将内存返回给操作系统

        except Exception as e:
            # Format the error message with full traceback
            error_details = traceback.format_exc()
            log_message = (
                f"--- Error processing video: {video_path} ---\n"
                f"{error_details}"
                f"--------------------------------------------------\n\n"
            )

            # Highlight and print a summary to the console
            print(f"\n[bold red]Failed to process:[/bold red] {video_path}")
            print(f"[red]Error:[/red] {e}")
            print(f"[yellow]Full traceback saved to {error_log_path}[/yellow]")

            # Write the detailed log to the error file
            with open(error_log_path, "a") as f:
                f.write(log_message)
            
            # 即使出错也要清理GPU内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print("Processing finished.")
