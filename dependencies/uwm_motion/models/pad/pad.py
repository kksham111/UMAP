from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import rearrange


from models.common.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from models.common.utils import SinusoidalPosEmb, init_weights
from .obs_encoder import PADObservationEncoder


class MultiViewVideoPatchifier(nn.Module):
    def __init__(
        self,
        num_views: int,
        input_shape: tuple[int, ...] = (8, 224, 224),
        patch_shape: tuple[int, ...] = (2, 8, 8),
        num_chans: int = 3,
        embed_dim: int = 768,
        cond_chans: bool = True,
    ):
        super().__init__()
        self.num_views = num_views
        iT, iH, iW = input_shape
        pT, pH, pW = patch_shape
        self.T, self.H, self.W = iT // pT, iH // pH, iW // pW

        self.patch_encoder = nn.Conv3d(
            in_channels=num_chans * 2 if cond_chans else num_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.patch_decoder = nn.ConvTranspose3d(
            in_channels=embed_dim,
            out_channels=num_chans,
            kernel_size=patch_shape,
            stride=patch_shape,
        )

    def forward(self, imgs):
        return self.patchify(imgs)

    def patchify(self, imgs):
        imgs = rearrange(imgs, "b v c t h w -> (b v) c t h w")
        feats = self.patch_encoder(imgs)
        feats = rearrange(feats, "(b v) c t h w -> b (v t h w) c", v=self.num_views)
        return feats

    def unpatchify(self, feats):
        feats = rearrange(
            feats, "b (v t h w) c -> (b v) c t h w", t=self.T, h=self.H, w=self.W
        )
        imgs = self.patch_decoder(feats)
        imgs = rearrange(imgs, "(b v) c t h w -> b v c t h w", v=self.num_views)
        return imgs

    @property
    def num_patches(self):
        return self.num_views * self.T * self.H * self.W


class TimestepEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, mlp_ratio: float = 4.0):
        super().__init__()
        self.sinusoidal_pos_emb = SinusoidalPosEmb(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t):
        temb = self.sinusoidal_pos_emb(t)
        return self.proj(temb)


class NoisePredictionNet(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, ...],
        patch_shape: Tuple[int, ...],
        num_chans: int,
        num_views: int,
        action_len: int,
        action_dim: int,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        # Observation encoder and decoder
        self.obs_patchifier = MultiViewVideoPatchifier(
            num_views=num_views,
            input_shape=image_shape,
            patch_shape=patch_shape,
            num_chans=num_chans,
            embed_dim=embed_dim,
            cond_chans=True,
        )
        obs_len = self.obs_patchifier.num_patches
        self.obs_len = obs_len
        self.action_len = action_len

        # Action encoder and decoder
        hidden_dim = int(max(action_dim, embed_dim) * mlp_ratio)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Timestep embedding
        self.timestep_embedding = TimestepEncoder(timestep_embed_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.empty(1, action_len + obs_len, embed_dim).normal_(std=0.02)
        )

        # Action padding
        self.pad_tokens = nn.Parameter(
            torch.empty(1, action_len, embed_dim).normal_(std=0.02)
        )

        # DiT blocks
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=timestep_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=timestep_embed_dim)
        self.action_inds = (0, action_len)
        self.next_obs_inds = (action_len, action_len + obs_len)

        # AdaLN-specific weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Base initialization
        self.apply(init_weights)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.obs_patchifier.patch_encoder.weight.data
        nn.init.normal_(w.view([w.shape[0], -1]), mean=0.0, std=0.02)
        nn.init.constant_(self.obs_patchifier.patch_encoder.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(self, obs, action, next_obs, t, action_mask=None):
        # Encode inputs
        action_embed = self.action_encoder(action)
        next_obs_embed = self.obs_patchifier(torch.cat([obs, next_obs], dim=2))

        # Expand and encode timesteps
        if len(t.shape) == 0:
            t = t.expand(action.shape[0]).to(dtype=torch.long, device=action.device)
        temb = self.timestep_embedding(t)

        # Construct attention mask
        attn_mask = None
        if action_mask is not None:
            # Action mask has shape (B,)
            action_embed[~action_mask] = self.pad_tokens.expand(
                sum(~action_mask), -1, -1
            )
            B = action_mask.shape[0]
            N = self.action_len + self.obs_len
            attn_mask = torch.ones((B, N, N), device=action.device, dtype=torch.bool)
            attn_mask[~action_mask, :, : self.action_len] = 0

        # Forward through model
        x = torch.cat((action_embed, next_obs_embed), dim=1)
        x = x + self.pos_embed
        cond = temb
        for block in self.blocks:
            x = block(x, cond, attn_mask=attn_mask)
        x = self.head(x, cond)

        # Extract action and next observation noise predictions
        action_noise_pred = x[:, self.action_inds[0] : self.action_inds[1]]
        next_obs_noise_pred = x[:, self.next_obs_inds[0] : self.next_obs_inds[1]]

        # Decode outputs
        action_noise_pred = self.action_decoder(action_noise_pred)
        next_obs_noise_pred = self.obs_patchifier.unpatchify(next_obs_noise_pred)
        return action_noise_pred, next_obs_noise_pred


class PAD(nn.Module):
    def __init__(
        self,
        action_len: int,
        action_dim: int,
        obs_encoder: PADObservationEncoder,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        latent_patch_shape: tuple[int, ...] = (2, 4, 4),
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
    ):
        """
        Assumes rgb input: (B, T, H, W, C) uint8 image
        Assumes low_dim input: (B, T, D)
        """

        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.action_shape = (action_len, action_dim)

        # Image augmentation
        self.obs_encoder = obs_encoder
        self.latent_img_shape = self.obs_encoder.latent_img_shape()

        # Diffusion noise prediction network
        num_views, num_chans = self.latent_img_shape[:2]
        image_shape = self.latent_img_shape[2:]
        self.noise_pred_net = NoisePredictionNet(
            image_shape=image_shape,
            patch_shape=latent_patch_shape,
            num_chans=num_chans,
            num_views=num_views,
            action_len=action_len,
            action_dim=action_dim,
            embed_dim=embed_dim,
            timestep_embed_dim=timestep_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )

        # Diffusion scheduler
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )

    def forward(self, obs_dict, next_obs_dict, action, action_mask=None):
        batch_size, device = action.shape[0], action.device

        # Encode observations
        obs, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict
        )

        # Forward diffusion process
        t = torch.randint(
            low=0, high=self.num_train_steps, size=(batch_size,), device=device
        ).long()
        action_noise = torch.randn_like(action)
        noisy_action = self.noise_scheduler.add_noise(action, action_noise, t)
        next_obs_noise = torch.randn_like(next_obs)
        noisy_next_obs = self.noise_scheduler.add_noise(next_obs, next_obs_noise, t)

        # Diffusion loss
        action_noise_pred, next_obs_noise_pred = self.noise_pred_net(
            obs, noisy_action, noisy_next_obs, t, action_mask
        )
        if action_mask is not None:
            action_loss = F.mse_loss(
                action_noise_pred[action_mask], action_noise[action_mask]
            )
        else:
            action_loss = F.mse_loss(action_noise_pred, action_noise)
        dynamics_loss = F.mse_loss(next_obs_noise_pred, next_obs_noise)
        loss = action_loss + dynamics_loss

        # Logging
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
        obs = self.obs_encoder.encode_obs(obs_dict)

        # Initialize action and next_obs
        action_sample = torch.randn(
            (obs.shape[0],) + self.action_shape, device=obs.device
        )
        next_obs_sample = torch.randn(
            (obs.shape[0],) + self.latent_img_shape, device=obs.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            action_noise_pred, next_obs_noise_pred = self.noise_pred_net(
                obs, action_sample, next_obs_sample, t
            )
            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample
            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample
        return next_obs_sample, action_sample
