from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from models.common.language import CLIPTextEncoder
from models.common.transforms import VideoTransform, VAEDownsample
from models.common.vision import ResNetImageEncoder, ViTImageEncoder


class UWMObservationEncoder(nn.Module):
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
        vision_backbone: str = "vit",
        use_low_dim: bool = True,
        use_language: bool = True,
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
        self.embed_dim = embed_dim

        # Image augmentation
        self.obs_transform = VideoTransform(
            resize_shape=resize_shape,
            crop_shape=crop_shape,
            random_crop=random_crop,
            color_jitter=color_jitter,
            imagenet_norm=imagenet_norm,
        )

        # Image encoder
        if vision_backbone == "vit":
            self.img_encoder = ViTImageEncoder(
                num_views=self.num_views,
                embed_dim=embed_dim,
            )
        elif vision_backbone == "resnet":
            self.img_encoder = ResNetImageEncoder(
                num_views=self.num_views,
                embed_dim=embed_dim,
            )
        else:
            raise NotImplementedError(f"Unsupported vision backbone: {vision_backbone}")

        # Low-dim observations
        self.use_low_dim = use_low_dim

        # Language encoder
        self.use_language = use_language
        self.text_encoder = (
            CLIPTextEncoder(embed_dim=embed_dim) if use_language else None
        )

        # VAE downsampling
        self.vae = VAEDownsample()

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

    def apply_vae(
        self,
        imgs_list: Union[torch.Tensor, list[torch.Tensor]],
        inverse: bool = False,
        microbatch_size: int = 72,  # Tuned for 40GB VRAM
    ):
        """
        Accept a list of images and apply VAE to downsample or upsample images.
        If inverse is False, downsample images. Otherwise, upsample images.
        Process images in microbatches to reduce memory usage.
        """
        if isinstance(imgs_list, torch.Tensor):
            imgs_list = [imgs_list]
            is_singleton = True
        else:
            is_singleton = False
        assert isinstance(imgs_list, list)
        imgs = torch.cat(imgs_list, dim=0)

        # Flatten multiview videos to images
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, "b v c t h w -> (b v t) c h w")

        # Process images in microbatches
        transformed_imgs = []
        for i in range(0, imgs.shape[0], microbatch_size):
            batch_imgs = imgs[i : i + microbatch_size]
            if inverse:
                batch_transformed_imgs = self.vae.inverse(batch_imgs)
            else:
                batch_transformed_imgs = self.vae(batch_imgs)
            transformed_imgs.append(batch_transformed_imgs)
        transformed_imgs = torch.cat(transformed_imgs, dim=0)

        # Unflatten images to multiview videos
        transformed_imgs = rearrange(
            transformed_imgs, "(b v t) c h w -> b v c t h w", b=B, v=V
        )
        if not is_singleton:
            chunk_sizes = [img.shape[0] for img in imgs_list]
            transformed_imgs = list(transformed_imgs.split(chunk_sizes, dim=0))
        return transformed_imgs

    def encode_curr_obs(self, curr_obs_dict: dict):
        # Encoder current observations to features
        curr_imgs = self.apply_transform(curr_obs_dict)
        curr_feats = self.img_encoder(curr_imgs)  # (B, V*T*D)

        if self.use_low_dim:
            low_dims = [curr_obs_dict[key] for key in self.low_dim_keys]
            low_dims = torch.cat(low_dims, dim=-1).flatten(1)
            curr_feats = torch.cat([curr_feats, low_dims], dim=-1)

        if self.use_language:
            lang_feats = self.text_encoder(
                input_ids=curr_obs_dict["input_ids"],
                attention_mask=curr_obs_dict["attention_mask"],
            )
            curr_feats = torch.cat([curr_feats, lang_feats], dim=-1)
        return curr_feats

    def encode_next_obs(self, next_obs_dict: dict):
        # Encoder next observations to latents
        next_imgs = self.apply_transform(next_obs_dict)
        next_latents = self.apply_vae(next_imgs)
        return next_latents

    def encode_curr_and_next_obs(self, curr_obs_dict: dict, next_obs_dict: dict):
        # Apply the same transform to obs and next obs
        curr_imgs, next_imgs = self.apply_transform([curr_obs_dict, next_obs_dict])

        # Encode current obs to features
        curr_feats = self.img_encoder(curr_imgs)  # (B, V*T*D)

        if self.use_low_dim:
            low_dims = [curr_obs_dict[key] for key in self.low_dim_keys]
            low_dims = torch.cat(low_dims, dim=-1).flatten(1)
            curr_feats = torch.cat([curr_feats, low_dims], dim=-1)

        if self.use_language:
            lang_feats = self.text_encoder(
                input_ids=curr_obs_dict["input_ids"],
                attention_mask=curr_obs_dict["attention_mask"],
            )
            curr_feats = torch.cat([curr_feats, lang_feats], dim=-1)

        # Encode next obs to latents
        next_latents = self.apply_vae(next_imgs)
        return curr_feats, next_latents

    def feat_dim(self):
        # Return the dimension of encoded features
        low_dim_size = sum(
            self.shape_meta["obs"][key]["shape"][-1] for key in self.low_dim_keys
        )
        return (
            self.num_views * self.num_frames * self.embed_dim
            + int(self.use_low_dim) * self.num_frames * low_dim_size
            + int(self.use_language) * self.embed_dim
        )

    def latent_img_shape(self):
        # Construct dummy image and forward pass to get latent image shape
        dummy_obs = {}
        for k in self.rgb_keys:
            img_shape = self.shape_meta["obs"][k]["shape"]
            dummy_obs[k] = torch.zeros(
                1, self.num_frames, *img_shape, dtype=torch.uint8
            )
        with torch.no_grad():
            latent = self.encode_next_obs(dummy_obs)
        return tuple(latent.shape[1:])
