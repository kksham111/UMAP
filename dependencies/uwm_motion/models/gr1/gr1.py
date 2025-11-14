# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GR-1 model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.common.attention import AttentionBlock
from models.common.utils import get_nd_sinusoidal_embed
from models.gr1.obs_encoder import GR1ObservationEncoder
from models.gr1.vision_transformer import Block


class GR1(nn.Module):
    def __init__(
        self,
        action_len: int,
        action_dim: int,
        obs_encoder: GR1ObservationEncoder,
        embed_dim: int,
        image_size: tuple[int, ...],
        patch_size: tuple[int, ...],
        num_chans: int = 3,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        decoder_depth: int = 3,
        decoder_num_heads: int = 16,
        decoder_mlp_ratio: int = 4,
        decoder_qkv_bias: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_len = action_len
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_chans = num_chans

        # Observation encoder (with perceiver resampler)
        self.obs_encoder = obs_encoder
        self.obs_len = obs_encoder.num_latents

        # Action query token
        self.action_queries = nn.Parameter(
            torch.empty(1, self.action_len, embed_dim).normal_(mean=0, std=0.02)
        )

        # Observation query token
        self.obs_queries = nn.Parameter(
            torch.empty(1, self.obs_len, embed_dim).normal_(mean=0, std=0.02)
        )

        # Main transformer
        self.transformer = nn.ModuleList(
            [
                AttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, self.action_dim),
        )

        # Image decoder
        self.decoder_query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        decoder_pos_embed = get_nd_sinusoidal_embed(
            embed_dim,
            (self.image_size // self.patch_size, self.image_size // self.patch_size),
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.from_numpy(decoder_pos_embed).float()[None], requires_grad=False
        )
        self.decoder_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=decoder_mlp_ratio,
                    qkv_bias=decoder_qkv_bias,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.patch_size**2 * 3, bias=True),
        )

    def predict_next_obs(self, next_obs_embed):
        next_obs_embed = rearrange(
            next_obs_embed,
            "b (v t n) d -> (b v t) n d",
            v=self.obs_encoder.num_views,
            t=self.obs_encoder.num_frames,
        )
        next_obs_embed = self.decoder_proj(next_obs_embed)
        next_obs_queries = self.decoder_query_token + self.decoder_pos_embed
        next_obs_pred = torch.cat(
            [next_obs_embed, next_obs_queries.repeat(next_obs_embed.shape[0], 1, 1)],
            dim=1,
        )
        for block in self.decoder_blocks:
            next_obs_pred = block(next_obs_pred)
        next_obs_pred = self.decoder_head(
            next_obs_pred[:, -self.decoder_pos_embed.shape[1] :]
        )
        next_obs_pred = rearrange(
            next_obs_pred,
            "(b v t) (h w) (c ph pw) -> b v c t (h ph) (w pw)",
            v=self.obs_encoder.num_views,
            t=self.obs_encoder.num_frames,
            h=self.image_size // self.patch_size,
            w=self.image_size // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return next_obs_pred

    def forward(self, obs_dict, next_obs_dict, action, action_mask=None):
        # Get obs patches and prediction targets
        curr_embeds, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict
        )

        # Transformer inputs
        action_queries = self.action_queries.expand(action.shape[0], -1, -1)
        obs_queries = self.obs_queries.expand(next_obs.shape[0], -1, -1)
        x = torch.cat([curr_embeds, action_queries, obs_queries], dim=1)

        # Attention mask
        attn_mask = None
        if action_mask is not None:
            # Action mask has shape (B,)
            B = action_mask.shape[0]
            N = self.action_len + self.obs_len * 2
            attn_mask = torch.ones((B, N, N), device=action.device, dtype=torch.bool)
            attn_mask[
                ~action_mask, :, self.obs_len : self.obs_len + self.action_len
            ] = 0

        # Transformer forward pass
        for block in self.transformer:
            x = block(x, attn_mask=attn_mask)

        # Action prediction
        action_embed = x[:, self.obs_len : self.obs_len + self.action_len]
        action_pred = self.action_head(action_embed)

        # Image prediction
        next_obs_embed = x[:, -self.obs_len :]
        next_obs_pred = self.predict_next_obs(next_obs_embed)

        # Compute losses
        if action_mask is None:
            action_loss = F.mse_loss(action_pred, action)
        else:
            action_loss = F.mse_loss(action_pred[action_mask], action[action_mask])
        dynamics_loss = F.mse_loss(next_obs_pred, next_obs)
        loss = action_loss + dynamics_loss
        info = {
            "loss": loss.item(),
            "action_loss": action_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
        }
        return loss, info

    @torch.no_grad()
    def sample(self, obs_dict):
        _, action_sample = self.sample_joint(obs_dict)
        return action_sample

    @torch.no_grad()
    def sample_joint(self, obs_dict):
        # Get obs patches and prediction targets
        curr_embeds = self.obs_encoder.encode_obs(obs_dict)

        # Transformer inputs
        action_queries = self.action_queries.expand(curr_embeds.shape[0], -1, -1)
        obs_queries = self.obs_queries.expand(curr_embeds.shape[0], -1, -1)
        x = torch.cat([curr_embeds, action_queries, obs_queries], dim=1)

        # Transformer forward pass
        for block in self.transformer:
            x = block(x)

        # Action prediction
        action_embed = x[:, self.obs_len : self.obs_len + self.action_len]
        action_pred = self.action_head(action_embed)

        # Image prediction
        next_obs_embed = x[:, -self.obs_len :]
        next_obs_pred = self.predict_next_obs(next_obs_embed)
        return next_obs_pred, action_pred
