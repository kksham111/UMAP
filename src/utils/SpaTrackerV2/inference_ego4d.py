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
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_type", type=str, default="RGBD")
    parser.add_argument("--data_dir", type=str, default="assets/example1")
    parser.add_argument("--out_dir", type=str, default="assets/example1/results")
    parser.add_argument("--video_name", type=str, default="snowboard")
    parser.add_argument("--grid_size", type=int, default=30)
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument("--fps", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # out_dir = args.data_dir + "/results"
    out_dir = args.out_dir
    # fps
    fps = int(args.fps)
    mask_dir = args.data_dir + f"/{args.video_name}.png"
    
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")

    if args.data_type == "RGBD":
        npz_dir = args.data_dir + f"/{args.video_name}.npz"
        data_npz_load = dict(np.load(npz_dir, allow_pickle=True))
        #TODO: tapip format
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
        vid_dir = os.path.join(args.data_dir, f"{args.video_name}.mp4")
        video_reader = decord.VideoReader(vid_dir)
        video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)  # Convert to tensor and permute to (N, C, H, W)
        video_tensor = video_tensor[::fps].float()

        # process the image tensor
        video_tensor = preprocess_image(video_tensor)[None]
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = vggt4track_model(video_tensor.cuda()/255)
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
        
        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
        extrs = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_tensor.squeeze()
        #NOTE: 20% of the depth is not reliable
        # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

        data_npz_load = {}
    
    if os.path.exists(mask_dir):
        mask_files = mask_dir
        mask = cv2.imread(mask_files)
        mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
        mask = mask.sum(axis=-1)>0
    else:
        mask = np.ones_like(video_tensor[0,0].numpy())>0
        
    # get all data pieces
    viz = True
    os.makedirs(out_dir, exist_ok=True)
        
    # with open(cfg_dir, "r") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = easydict.EasyDict(cfg)
    # cfg.out_dir = out_dir
    # cfg.model.track_num = args.vo_points
    # print(f"Downloading model from HuggingFace: {cfg.ckpts}")
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    # config the model; the track_num is the number of points in the grid
    model.spatrack.track_num = args.vo_points
    
    model.eval()
    model.to("cuda")
    viser = Visualizer(save_dir=out_dir, grayscale=True, 
                     fps=10, pad_value=0, tracks_leave_trace=5)
    
    grid_size = args.grid_size

    # get frame H W
    if video_tensor is  None:
        cap = cv2.VideoCapture(video_path)
        frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        frame_H, frame_W = video_tensor.shape[2:]

    info_clips = "/mnt/homes/jialin-ldap/ego4d/data/jialin_test/info_clips_10.json"
    with open(info_clips, "r") as f:
        info_clips = json.load(f)
    
    clip_name = os.path.basename(args.data_dir)
    action_num = int(args.video_name.split("_")[1])
    pre_frame_boxes = info_clips[clip_name][action_num]["pre_frame"]["boxes"]
    for i in range(len(pre_frame_boxes)):
        if pre_frame_boxes[i]["object_type"] == "object_of_change":
            bbox = pre_frame_boxes[i]["bbox"]
            break
    
    center_x = bbox["x"] + bbox["width"] / 2
    center_y = bbox["y"] + bbox["height"] / 2
    interp_shape_for_bbox = (bbox["height"], bbox["width"]) 
    grid_center_for_bbox = (center_y, center_x)
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    # grid_pts = get_points_on_a_grid(grid_size, interp_shape_for_bbox, grid_center_for_bbox, device="cpu")

    def bbox_to_mask(bbox, canvas_shape):
        """
        将一个边界框（bbox）转换为一个二维的布尔掩码（mask）。

        参数:
        bbox (dict): 包含 'x', 'y', 'width', 'height' 的字典。
        canvas_shape (tuple): 一个元组 (Height, Width)，表示最终掩码的尺寸。

        返回:
        torch.Tensor: 一个二维的掩码 Tensor，矩形区域内为 1，其他区域为 0。
        """
        # 1. 获取画布的高度和宽度
        canvas_height, canvas_width = canvas_shape
        
        # 2. 创建一个全为 0 的画布
        mask = torch.zeros(canvas_shape, dtype=torch.uint8) # 使用 uint8 适合存储 0-255 的值

        # 3. 计算 bbox 的四个整数坐标
        # 使用 int() 确保坐标是整数，因为 Tensor 切片需要整数索引
        x_start = int(bbox['x'])
        y_start = int(bbox['y'])
        x_end = int(x_start + bbox['width'])
        y_end = int(y_start + bbox['height'])

        # 4. (安全措施) 确保坐标不会超出画布边界
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(canvas_width, x_end)
        y_end = min(canvas_height, y_end)

        # 5. 使用切片操作，将矩形区域内的值设置为 1
        # 这是最关键的一步！
        mask[y_start:y_end, x_start:x_end] = 1
        
        return mask

    canvas_size = (frame_H, frame_W) # (高度=H, 宽度=W)
    # 将原始标注分辨率(假设 2560x1920)的 bbox 缩放到当前帧尺寸
    src_w, src_h = 1440, 1080
    scaled_bbox = {
        'x': bbox['x'] / src_w * frame_W,
        'y': bbox['y'] / src_h * frame_H,
        'width': bbox['width'] / src_w * frame_W,
        'height': bbox['height'] / src_h * frame_H,
    }
    bbox_mask = bbox_to_mask(scaled_bbox, canvas_size)

    print("边界框信息:", bbox)
    print(f'scaled_bbox: {scaled_bbox}')
    print(f"生成的 {canvas_size} 掩码:")
    print(bbox_mask)

    # 记录原始网格点，方便兜底回退
    grid_pts_all = grid_pts.clone()
    grid_pts_int = grid_pts[0].long()
    mask_values = bbox_mask[grid_pts_int[...,1], grid_pts_int[...,0]]
    grid_pts = grid_pts[:, mask_values]

    # Sample mask values at grid points and filter out points where mask=0
    if os.path.exists(mask_dir):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]


    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 
        
        # ==================== DEBUG CODE START ====================
        # # Draw points_inside_bbox on frame 0 (green) and save to out_dir
        # raw_frame_rgb = video[0].detach().permute(1, 2, 0).cpu().numpy().copy()
        # min_val, max_val = raw_frame_rgb.min(), raw_frame_rgb.max()
        # if max_val > min_val:
        #     normalized_frame = (raw_frame_rgb - min_val) / (max_val - min_val)
        #     uint8_frame_rgb = (normalized_frame * 255).clip(0, 255).astype(np.uint8)
        # else:
        #     uint8_frame_rgb = np.zeros_like(raw_frame_rgb, dtype=np.uint8)
        # debug_img_bgr = cv2.cvtColor(uint8_frame_rgb, cv2.COLOR_RGB2BGR)

        # pts = grid_pts.squeeze(0)
        # if isinstance(pts, torch.Tensor):
        #     pts_np = pts.detach().cpu().numpy()
        # else:
        #     pts_np = np.asarray(pts)
        # if pts_np.size > 0:
        #     for (x, y) in pts_np.astype(np.int32):
        #         cv2.circle(debug_img_bgr, (int(x), int(y)), 6, (0, 255, 0), -1)
        # else:
        #     print("[DEBUG] points_inside_bbox is empty, nothing to draw.")

        # debug_img_path = os.path.join(out_dir, "DEBUG_points_inside_bbox_frame0.png")
        # cv2.imwrite(debug_img_path, debug_img_bgr)
        # print(f"[DEBUG] Saved bbox points overlay: {debug_img_path}")
        # ===================== DEBUG CODE END =====================

        # resize the results to avoid too large I/O Burden
        # depth and image, the maximum side is 336
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

        # basename = os.path.basename(args.data_dir)
        # output_name = basename[-5:]
        # video_name = args.video_name
        # parts = video_name.split("_")
        # output_name = output_name + "_" + "_".join(parts[:2])
        output_name = f"pred_{args.video_name}_2d"
        if viz:
            # viser.visualize(video=video[None],
            #                     tracks=track2d_pred[None][...,:2],
            #                     visibility=vis_pred[None],filename=output_name)
            viser.visualize(video=video[None],
                            tracks=track2d_pred[None][...,:2],
                            visibility=vis_pred[None],filename=output_name)

        # save as the tapip3d format   
        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        depth_save = point_map[:,2,...]
        depth_save[conf_depth<0.5] = 0
        data_npz_load["depths"] = depth_save.cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
        np.savez(os.path.join(out_dir, f'result_{output_name}.npz'), **data_npz_load)

        print(f"Results saved to {out_dir}.\nTo visualize them with tapip3d, run: [bold yellow]python tapip3d_viz.py {out_dir}/result_{output_name}.npz[/bold yellow]")
