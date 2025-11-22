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
    parser.add_argument("--fps", type=int, default=2) # 抽帧步长
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on, e.g., cuda:4 or cpu")
    return parser.parse_args()

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
    video_paths = sorted(glob.glob(os.path.join(args.input_dir, "**", "video_*.mp4"), recursive=True))
    print(f"Found {len(video_paths)} videos to process in '{args.input_dir}'.")

    for video_path in tqdm.tqdm(video_paths, desc="Processing videos"):
        try:
            if video_path == "/mnt/homes/jialin-ldap/UMAP/workspace/output/libero_3dtraj/libero_90/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo":
                continue
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

            # --- Per-Video Processing ---
            fps = int(args.fps)
            mask_dir = os.path.join(data_dir, f"{video_name}.png")
            video_fps = float("nan")

            if args.data_type == "RGBD":
                npz_dir = os.path.join(data_dir, f"{video_name}.npz")
                if not os.path.exists(npz_dir):
                    tqdm.tqdm.write(f"Warning: RGBD mode, but NPZ file not found for {video_path}, skipping.")
                    continue
                data_npz_load = dict(np.load(npz_dir, allow_pickle=True))
                video_tensor = data_npz_load["video"] * 255
                video_tensor = torch.from_numpy(video_tensor)
                video_tensor = video_tensor[::fps]
                depth_tensor = data_npz_load["depths"]
                depth_tensor = depth_tensor[::fps]
                intrs = data_npz_load["intrinsics"]
                intrs = intrs[::fps]
                extrs = np.linalg.inv(data_npz_load["extrinsics"])
                extrs = extrs[::fps]
                unc_metric = None
            elif args.data_type == "RGB":
                video_reader = decord.VideoReader(video_path)
                try:
                    video_fps = float(video_reader.get_avg_fps())
                except Exception:
                    video_fps = float("nan")
                
                video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)
                video_tensor = video_tensor[::fps].float()

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
                data_npz_load = {}
            else:
                tqdm.tqdm.write(f"Unsupported data_type '{args.data_type}', skipping {video_path}")
                continue
            
            if os.path.exists(mask_dir):
                mask = cv2.imread(mask_dir)
                mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
                mask = mask.sum(axis=-1)>0
            else:
                mask = np.ones_like(video_tensor[0,0].numpy())>0
                
            viz = True
        
            viser = Visualizer(save_dir=output_directory_for_video, grayscale=True, 
                             fps=10, pad_value=0, tracks_leave_trace=5)
            
            grid_size = args.grid_size

            frame_H, frame_W = video_tensor.shape[2:]
            grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
            
            if os.path.exists(mask_dir):
                grid_pts_int = grid_pts[0].long()
                mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
                grid_pts = grid_pts[:, mask_values]
            
            query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

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

                if viz:
                    viser.visualize(video=video[None],
                                        tracks=track2d_pred[None][...,:2],
                                        visibility=vis_pred[None],filename=f"{video_name}_track_viz")

                coords_world = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu())
                                + c2w_traj[:,:3,3][:,None,:]).numpy()

                coords_cam = track3d_pred[:,:,:3].cpu().numpy()

                T_frames, N_points, _ = coords_world.shape
                t_idx = np.arange(T_frames, dtype=np.int32)

                t_tile = np.tile(t_idx[:, None], (1, N_points))[..., None].astype(coords_world.dtype)
                coords_wt = np.concatenate([coords_world, t_tile], axis=2)

                vis = vis_pred.cpu().numpy().astype(bool).squeeze(-1)
                coords_wt_masked = coords_wt.copy()
                coords_wt_masked[~vis] = np.nan

                extrinsics = torch.inverse(c2w_traj).cpu().numpy()

                intrinsics = intrs.cpu().numpy()

                depth_save = point_map[:,2,...]
                depth_save[conf_depth<0.5] = 0
                depths = depth_save.cpu().numpy()
                unc_metric_np = conf_depth.cpu().numpy()

                video_np = (video_tensor).cpu().numpy()/255

                tracks_2d = track2d_pred.cpu().numpy()
                visibs = vis_pred.cpu().numpy()
                confs = conf_pred.cpu().numpy()

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
                    frame_stride=np.array([fps], dtype=np.int32),
                    video_fps=np.array([video_fps], dtype=np.float32),
                    t_idx=t_idx,
                )

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

    print("Processing finished.")
