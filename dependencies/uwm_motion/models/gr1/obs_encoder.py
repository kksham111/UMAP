from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from models.common.transforms import VideoTransform
from models.common.vision import get_vit
from models.gr1.flamingo import PerceiverResampler


class MultiViewViTImageEncoder(nn.Module):
    def __init__(
        self, num_views: int, num_frames: int, embed_dim: int, resampler_params: dict
    ):
        super().__init__()
        self.num_views = num_views
        self.model = get_vit("vit_b_32", embed_dim, weights="IMAGENET1K_V1")

        # Perceiver resampler
        self.perceiver_resampler = PerceiverResampler(**resampler_params)

        # Learnable embeddings
        self.pos_shift = nn.Parameter(
            torch.zeros(1, num_views * num_frames, 1, embed_dim),
            requires_grad=True,
        )
        self.pos_scale = nn.Parameter(
            torch.zeros(1, num_views * num_frames, 1, embed_dim),
            requires_grad=True,
        )

        self.feat_head = nn.Linear(embed_dim, embed_dim)
        self.patch_head = nn.Linear(embed_dim, embed_dim)

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, "b v c t h w -> (b v t) c h w")

        # Reshape and permute the input tensor
        x = self.model._process_input(imgs)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Get raw tokens
        x = self.model.encoder(x)
        feats, patch_embeds = x[:, :1], x[:, 1:]

        # Pass through perceiver resampler
        feats = self.feat_head(feats)
        patch_embeds = self.patch_head(
            self.perceiver_resampler(x.unsqueeze(1)).squeeze(1)
        )

        # Add learned positional embeddings
        x = torch.cat([feats, patch_embeds], dim=1)
        x = rearrange(x, "(b v t) n c -> b (v t) n c", b=B, v=V)
        x = x * (1 + self.pos_scale) + self.pos_shift
        return x.flatten(1, 2)  # (b, v*t*n, c)


class GR1ObservationEncoder(nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        num_frames: int,
        embed_dim: int,
        resize_shape: Tuple[int, int] = None,
        crop_shape: Tuple[int, int] = None,
        random_crop: bool = True,
        color_jitter: Optional[Dict] = None,
        imagenet_norm: bool = False,
        resampler_params: dict = None,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.num_frames = num_frames
        self.rgb_keys = sorted(
            [k for k, v in shape_meta["obs"].items() if v["type"] == "rgb"]
        )
        self.low_dim_keys = sorted(
            [k for k, v in shape_meta["obs"].items() if v["type"] == "low_dim"]
        )
        self.num_views = len(self.rgb_keys)

        # Image augmentation
        self.obs_transform = VideoTransform(
            resize_shape=resize_shape,
            crop_shape=crop_shape,
            random_crop=random_crop,
            color_jitter=color_jitter,
            imagenet_norm=imagenet_norm,
        )

        # Image encoder
        self.img_encoder = MultiViewViTImageEncoder(
            num_views=self.num_views,
            num_frames=self.num_frames,
            embed_dim=embed_dim,
            resampler_params=resampler_params,
        )

    def apply_transform(self, obs_dicts: Union[dict, list[dict]]):
        """
        Accept a list of observation dictionaries and apply the same transform to each.
        """
        if isinstance(obs_dicts, dict):
            obs_dicts = [obs_dicts]
            is_singleton = True
        else:
            is_singleton = False
        assert isinstance(obs_dicts, list)

        # Apply the same transform to each observation
        num_obs = len(obs_dicts)
        transformed_imgs = [[] for _ in range(num_obs)]
        for key in self.rgb_keys:
            combined_imgs = torch.cat([obs_dict[key] for obs_dict in obs_dicts], dim=0)
            combined_imgs = self.obs_transform(combined_imgs)
            chunked_imgs = combined_imgs.chunk(num_obs, dim=0)
            for i, img in enumerate(chunked_imgs):
                transformed_imgs[i].append(img)

        # Stack transformed images
        # Each image has shape (B, V, C, T, H, W)
        transformed_imgs = [torch.stack(imgs, dim=1) for imgs in transformed_imgs]
        if is_singleton:
            transformed_imgs = transformed_imgs[0]
        return transformed_imgs

    def encode_obs(self, obs_dict: dict):
        imgs = self.apply_transform(obs_dict)
        feats = self.img_encoder(imgs)
        return feats

    def encode_curr_and_next_obs(self, curr_obs_dict: dict, next_obs_dict: dict):
        # Apply the same transform to obs and next obs
        curr_imgs, next_imgs = self.apply_transform([curr_obs_dict, next_obs_dict])
        curr_feats = self.img_encoder(curr_imgs)
        return curr_feats, next_imgs

    @property
    def num_latents(self):
        return (
            (self.img_encoder.perceiver_resampler.num_latents + 1)
            * self.num_views
            * self.num_frames
        )
