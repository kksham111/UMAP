from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.transforms import Normalize

from models.common.language import CLIPTextEncoder
from models.common.transforms import ImageTransform
from models.common.vision import get_resnet, get_vit, get_clip


class ImageObservationEncoder(nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        num_frames: int,
        embed_dim: int,
        resize_shape: Tuple[int, int] = None,
        crop_shape: Tuple[int, int] = None,
        random_crop: bool = True,
        color_jitter: Optional[Dict] = None,
        imagenet_norm: bool = True,
        pretrained_weights: Optional[str] = None,
        use_low_dim: bool = True,
        use_language: bool = True,
    ):
        """
        Assumes rgb input: (B, T, H, W, C) uint8 image
        Assumes low_dim input: (B, T, D)
        """
        super().__init__()
        rgb_keys = list()
        low_dim_keys = list()
        key_shape_map = dict()
        key_transform_map = nn.ModuleDict()

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_shape = tuple(attr["shape"])
            key_shape_map[key] = obs_shape

            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
                key_transform_map[key] = ImageTransform(
                    resize_shape=resize_shape,
                    crop_shape=crop_shape,
                    random_crop=random_crop,
                    color_jitter=color_jitter,
                    imagenet_norm=imagenet_norm,
                )
            elif obs_type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        self.shape_meta = shape_meta
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_shape_map = key_shape_map
        self.key_transform_map = key_transform_map

        # RGB model
        if pretrained_weights == "clip":
            assert not imagenet_norm, "imagenet_norm must be False for CLIP encoder"
            norm = Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
                inplace=True,
            )
            model = get_clip(embed_dim)
            self.rgb_encoder = nn.Sequential(norm, model)
        elif pretrained_weights == "vit":
            self.rgb_encoder = get_vit(
                "vit_b_32", embed_dim, weights=pretrained_weights
            )
        else:
            self.rgb_encoder = get_resnet(
                "resnet18", embed_dim, weights=pretrained_weights
            )

        # Low dim model
        self.use_low_dim = use_low_dim
        self.low_dim_size = sum([key_shape_map[key][-1] for key in low_dim_keys])

        # Language model
        self.use_language = use_language
        self.text_encoder = (
            CLIPTextEncoder(embed_dim=embed_dim) if use_language else None
        )

    def __call__(self, obs_dict):
        # Process rgb observations
        imgs = list()
        for key in self.rgb_keys:
            img = obs_dict[key].flatten(0, 1)
            assert img.shape[1:] == self.key_shape_map[key]
            img = self.key_transform_map[key](img)  # (B*T, C, H, W)
            imgs.append(img)

        # Concatenate along batch dimension
        imgs = torch.cat(imgs, dim=0)  # (N*B*T, C, H, W)
        feats = self.rgb_encoder(imgs)  # (N*B*T, D)
        feats = rearrange(
            feats, "(n b t) d -> b (t n d)", n=len(self.rgb_keys), t=self.num_frames
        )

        if self.use_low_dim:
            # Process low dim observations
            low_dims = list()
            for key in self.low_dim_keys:
                low_dim = obs_dict[key].flatten(0, 1)
                assert low_dim.shape[1:] == self.key_shape_map[key]
                low_dims.append(low_dim)
            low_dims = torch.cat(low_dims, dim=-1)  # (B*T, D_low_dim)
            low_dims = rearrange(low_dims, "(b t) d -> b (t d)", t=self.num_frames)

            # Concatenate image and lowdim features
            feats = torch.cat([feats, low_dims], dim=-1)

        # Encode language
        if self.use_language:
            lang_feats = self.text_encoder(
                input_ids=obs_dict["input_ids"],
                attention_mask=obs_dict["attention_mask"],
            )
            feats = torch.cat([feats, lang_feats], dim=-1)

        return feats

    @property
    def output_len(self):
        return (
            len(self.rgb_keys) * self.embed_dim
            + int(self.use_low_dim) * self.low_dim_size
        ) * self.num_frames + int(self.use_language) * self.embed_dim
