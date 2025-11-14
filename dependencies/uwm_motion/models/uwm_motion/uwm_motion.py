import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import rearrange


from models.common.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from models.common.utils import SinusoidalPosEmb, init_weights
from .obs_encoder import UWMObservationEncoder


class MultiViewVideoPatchifier(nn.Module):
    def __init__(
        self,
        num_views: int,
        input_shape: tuple[int, ...] = (8, 224, 224),
        patch_shape: tuple[int, ...] = (2, 8, 8),
        num_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_views = num_views
        iT, iH, iW = input_shape
        pT, pH, pW = patch_shape
        self.T, self.H, self.W = iT // pT, iH // pH, iW // pW
        self.pT, self.pH, self.pW = pT, pH, pW

        self.patch_encoder = nn.Conv3d(
            in_channels=num_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.patch_decoder = nn.Linear(embed_dim, num_chans * pT * pH * pW)

    def forward(self, imgs):
        return self.patchify(imgs)

    def patchify(self, imgs):
        imgs = rearrange(imgs, "b v c t h w -> (b v) c t h w")
        feats = self.patch_encoder(imgs)
        feats = rearrange(feats, "(b v) c t h w -> b (v t h w) c", v=self.num_views)
        return feats

    def unpatchify(self, feats):
        imgs = self.patch_decoder(feats)
        imgs = rearrange(
            imgs,
            "b (v t h w) (c pt ph pw) -> b v c (t pt) (h ph) (w pw)",
            v=self.num_views,
            t=self.T,
            h=self.H,
            w=self.W,
            pt=self.pT,
            ph=self.pH,
            pw=self.pW,
        )
        return imgs

    @property
    def num_patches(self):
        return self.num_views * self.T * self.H * self.W


class DualTimestepEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, mlp_ratio: float = 4.0):
        super().__init__()
        self.sinusoidal_pos_emb = SinusoidalPosEmb(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t1, t2):
        temb1 = self.sinusoidal_pos_emb(t1)
        temb2 = self.sinusoidal_pos_emb(t2)
        temb = torch.cat([temb1, temb2], dim=-1)
        return self.proj(temb)


class DualNoisePredictionNet(nn.Module):
    def __init__(
        self,
        global_cond_dim: int,
        image_shape: tuple[int, ...],
        patch_shape: tuple[int, ...],
        num_chans: int,
        num_views: int,
        action_len: int,
        action_dim: int,
        traj3d_len: int = 0,
        traj3d_points: int = 0,
        traj3d_dim: int = 3,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        num_registers: int = 8,
    ):
        super().__init__()
        # Observation encoder and decoder
        self.obs_patchifier = MultiViewVideoPatchifier(
            num_views=num_views,
            input_shape=image_shape,
            patch_shape=patch_shape,
            num_chans=num_chans,
            embed_dim=embed_dim,
        )
        obs_len = self.obs_patchifier.num_patches

        # Action encoder and decoder
        hidden_dim_action = int(max(action_dim, embed_dim) * mlp_ratio)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim_action),
            nn.Mish(),
            nn.Linear(hidden_dim_action, embed_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim_action),
            nn.Mish(),
            nn.Linear(hidden_dim_action, action_dim),
        )

        # 3D Trajectory encoder and decoder (照搬 action 的处理方式)
        self.use_traj3d = traj3d_len > 0 and traj3d_points > 0
        if self.use_traj3d:
            # Flatten points: (traj3d_points, traj3d_dim) -> (traj3d_points * traj3d_dim)
            traj3d_total_dim = traj3d_points * traj3d_dim
            hidden_dim_traj = int(max(traj3d_total_dim, embed_dim) * mlp_ratio)
            self.traj3d_encoder = nn.Sequential(
                nn.Linear(traj3d_total_dim, hidden_dim_traj),
                nn.Mish(),
                nn.Linear(hidden_dim_traj, embed_dim),
            )
            self.traj3d_decoder = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim_traj),
                nn.Mish(),
                nn.Linear(hidden_dim_traj, traj3d_total_dim),
            )
            self.traj3d_len = traj3d_len
            self.traj3d_points = traj3d_points
            self.traj3d_dim = traj3d_dim
        else:
            self.traj3d_len = 0

        # Timestep embedding
        self.timestep_embedding = DualTimestepEncoder(timestep_embed_dim)

        # Registers
        self.registers = nn.Parameter(
            torch.empty(1, num_registers, embed_dim).normal_(std=0.02)
        )

        # Positional embedding (更新长度以包含 traj3d)
        if self.use_traj3d:
            total_len = action_len + self.traj3d_len + obs_len + num_registers
        else:
            total_len = action_len + obs_len + num_registers
        self.pos_embed = nn.Parameter(
            torch.empty(1, total_len, embed_dim).normal_(std=0.02)
        )

        # DiT blocks
        cond_dim = global_cond_dim + timestep_embed_dim
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)
        
        # 更新序列索引 (action, traj3d, next_obs)
        self.action_inds = (0, action_len)
        if self.use_traj3d:
            self.traj3d_inds = (action_len, action_len + self.traj3d_len)
            self.next_obs_inds = (action_len + self.traj3d_len, action_len + self.traj3d_len + obs_len)
        else:
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

    def forward(self, global_cond, action, action_t, next_obs, next_obs_t, traj3d=None, traj3d_t=None):
        """
        Forward pass with optional traj3d support (照搬 action 的处理方式)
        
        Args:
            global_cond: Global conditioning (B, cond_dim)
            action: Action sequence (B, action_len, action_dim)
            action_t: Action diffusion timestep
            next_obs: Next observation (B, num_views, num_chans, T, H, W)
            next_obs_t: Observation diffusion timestep
            traj3d: 3D trajectory (B, traj3d_len, traj3d_points, traj3d_dim), optional
            traj3d_t: Traj3d diffusion timestep, optional
        
        Returns:
            action_noise_pred: (B, action_len, action_dim)
            traj3d_noise_pred: (B, traj3d_len, traj3d_points, traj3d_dim) or None
            next_obs_noise_pred: (B, num_views, num_chans, T, H, W)
        """
        # Encode action (original)
        action_embed = self.action_encoder(action)
        
        # Encode traj3d if provided (照搬 action 的编码方式)
        if self.use_traj3d and traj3d is not None:
            # traj3d shape: (B, traj3d_len, traj3d_points, traj3d_dim)
            B, T, N, D = traj3d.shape
            # Flatten points: (B, T, N*D) 以便 encoder 处理
            traj3d_flat = traj3d.reshape(B, T, -1)
            traj3d_embed = self.traj3d_encoder(traj3d_flat)  # (B, T, embed_dim)
        else:
            traj3d_embed = None
        
        # Encode next observation (original)
        next_obs_embed = self.obs_patchifier(next_obs)

        # Expand and encode timesteps
        if len(action_t.shape) == 0:
            action_t = action_t.expand(action.shape[0]).to(
                dtype=torch.long, device=action.device
            )
        if len(next_obs_t.shape) == 0:
            next_obs_t = next_obs_t.expand(next_obs.shape[0]).to(
                dtype=torch.long, device=next_obs.device
            )
        
        # If traj3d is used, use its timestep for the first position of dual timestep
        # (照搬 action 的时间步处理方式)
        if self.use_traj3d and traj3d is not None and traj3d_t is not None:
            if len(traj3d_t.shape) == 0:
                traj3d_t = traj3d_t.expand(traj3d.shape[0]).to(
                    dtype=torch.long, device=traj3d.device
                )
            # Use traj3d_t and next_obs_t as dual timesteps
            temb = self.timestep_embedding(traj3d_t, next_obs_t)
        else:
            # Original: use action_t and next_obs_t
            temb = self.timestep_embedding(action_t, next_obs_t)

        # Forward through model (拼接 action, traj3d, next_obs)
        registers = self.registers.expand(next_obs.shape[0], -1, -1)
        if self.use_traj3d and traj3d_embed is not None:
            x = torch.cat((action_embed, traj3d_embed, next_obs_embed, registers), dim=1)
        else:
            x = torch.cat((action_embed, next_obs_embed, registers), dim=1)
        x = x + self.pos_embed
        cond = torch.cat((global_cond, temb), dim=-1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.head(x, cond)

        # Extract predictions (从拼接的序列中提取各部分)
        action_noise_pred = x[:, self.action_inds[0] : self.action_inds[1]]
        if self.use_traj3d:
            traj3d_noise_pred = x[:, self.traj3d_inds[0] : self.traj3d_inds[1]]
        next_obs_noise_pred = x[:, self.next_obs_inds[0] : self.next_obs_inds[1]]

        # Decode outputs
        action_noise_pred = self.action_decoder(action_noise_pred)
        
        # Decode traj3d if applicable (照搬 action 的解码方式)
        if self.use_traj3d and traj3d is not None:
            traj3d_noise_pred = self.traj3d_decoder(traj3d_noise_pred)
            # Reshape back: (B, T, N*D) -> (B, T, N, D)
            B_out = traj3d_noise_pred.shape[0]
            traj3d_noise_pred = traj3d_noise_pred.reshape(
                B_out, self.traj3d_len, self.traj3d_points, self.traj3d_dim
            )
        else:
            traj3d_noise_pred = None
        
        next_obs_noise_pred = self.obs_patchifier.unpatchify(next_obs_noise_pred)
        
        return action_noise_pred, traj3d_noise_pred, next_obs_noise_pred


class UnifiedWorldModelWithMotion(nn.Module):
    def __init__(
        self,
        action_len: int,
        action_dim: int,
        obs_encoder: UWMObservationEncoder,
        traj3d_len: int = 0,
        traj3d_points: int = 0,
        traj3d_dim: int = 3,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        latent_patch_shape: tuple[int, ...] = (2, 4, 4),
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        num_registers: int = 8,
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

        # 3D Trajectory parameters (照搬 action 的处理方式)
        self.use_traj3d = traj3d_len > 0 and traj3d_points > 0
        self.traj3d_len = traj3d_len
        self.traj3d_points = traj3d_points
        self.traj3d_dim = traj3d_dim
        if self.use_traj3d:
            self.traj3d_shape = (traj3d_len, traj3d_points, traj3d_dim)

        # Image augmentation
        self.obs_encoder = obs_encoder
        self.latent_img_shape = self.obs_encoder.latent_img_shape()

        # Diffusion noise prediction network
        global_cond_dim = self.obs_encoder.feat_dim()
        image_shape = self.latent_img_shape[2:]
        num_views, num_chans = self.latent_img_shape[:2]
        self.noise_pred_net = DualNoisePredictionNet(
            global_cond_dim=global_cond_dim,
            image_shape=image_shape,
            patch_shape=latent_patch_shape,
            num_chans=num_chans,
            num_views=num_views,
            action_len=action_len,
            action_dim=action_dim,
            traj3d_len=traj3d_len,
            traj3d_points=traj3d_points,
            traj3d_dim=traj3d_dim,
            embed_dim=embed_dim,
            timestep_embed_dim=timestep_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            num_registers=num_registers,
        )

        # Diffusion scheduler
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )

    def forward(self, obs_dict, next_obs_dict, action, traj3d=None, action_mask=None, traj3d_mask=None):
        batch_size, device = action.shape[0], action.device

        # Encode observations
        obs, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict
        )

        # Sample diffusion timestep for action (照搬原有 action 的处理方式)
        action_noise = torch.randn_like(action)
        action_t = torch.randint(
            low=0, high=self.num_train_steps, size=(batch_size,), device=device
        ).long()
        if action_mask is not None:
            action_t[~action_mask] = self.num_train_steps - 1
        noisy_action = self.noise_scheduler.add_noise(action, action_noise, action_t)

        # Sample diffusion timestep for traj3d (照搬 action 的处理方式)
        if self.use_traj3d and traj3d is not None:
            traj3d_noise = torch.randn_like(traj3d)
            traj3d_t = torch.randint(
                low=0, high=self.num_train_steps, size=(batch_size,), device=device
            ).long()
            if traj3d_mask is not None:
                traj3d_t[~traj3d_mask] = self.num_train_steps - 1
            noisy_traj3d = self.noise_scheduler.add_noise(traj3d, traj3d_noise, traj3d_t)
        else:
            traj3d_noise = None
            traj3d_t = None
            noisy_traj3d = None

        # Sample diffusion timestep for next observation
        next_obs_noise = torch.randn_like(next_obs)
        next_obs_t = torch.randint(
            low=0, high=self.num_train_steps, size=(batch_size,), device=device
        ).long()
        noisy_next_obs = self.noise_scheduler.add_noise(
            next_obs, next_obs_noise, next_obs_t
        )

        # Diffusion loss
        action_noise_pred, traj3d_noise_pred, next_obs_noise_pred = self.noise_pred_net(
            obs, noisy_action, action_t, noisy_next_obs, next_obs_t, noisy_traj3d, traj3d_t
        )
        action_loss = F.mse_loss(action_noise_pred, action_noise)
        dynamics_loss = F.mse_loss(next_obs_noise_pred, next_obs_noise)
        
        # 计算 traj3d 损失 (照搬 action 的损失计算方式)
        if self.use_traj3d and traj3d is not None and traj3d_noise_pred is not None:
            traj3d_loss = F.mse_loss(traj3d_noise_pred, traj3d_noise)
            loss = action_loss + traj3d_loss + dynamics_loss
        else:
            traj3d_loss = None
            loss = action_loss + dynamics_loss

        # Logging
        info = {
            "loss": loss.item(),
            "action_loss": action_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
        }
        if traj3d_loss is not None:
            info["traj3d_loss"] = traj3d_loss.item()
        
        return loss, info

    @torch.no_grad()
    def sample(self, obs_dict):
        return self.sample_marginal_action(obs_dict)

    @torch.no_grad()
    def sample_forward_dynamics(self, obs_dict, action):
        # Encode observations
        obs = self.obs_encoder.encode_curr_obs(obs_dict)

        # Initialize next observation sample
        next_obs_sample = torch.randn(
            (obs.shape[0],) + self.latent_img_shape, device=obs.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        action_t = self.noise_scheduler.timesteps[-1]
        for next_obs_t in self.noise_scheduler.timesteps:
            _, next_obs_noise_pred = self.noise_pred_net(
                obs, action, action_t, next_obs_sample, next_obs_t
            )
            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, next_obs_t, next_obs_sample
            ).prev_sample
        return next_obs_sample

    @torch.no_grad()
    def sample_inverse_dynamics(self, obs_dict, next_obs_dict):
        # Encode observations
        obs_feat, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict
        )

        # Initialize action sample
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_obs_t = self.noise_scheduler.timesteps[-1]
        for action_t in self.noise_scheduler.timesteps:
            action_noise_pred, _ = self.noise_pred_net(
                obs_feat, action_sample, action_t, next_obs, next_obs_t
            )
            action_sample = self.noise_scheduler.step(
                action_noise_pred, action_t, action_sample
            ).prev_sample
        return action_sample

    @torch.no_grad()
    def sample_marginal_next_obs(self, obs_dict):
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)

        # Initialize action and next_obs
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        action_t = self.noise_scheduler.timesteps[0]
        for t in self.noise_scheduler.timesteps:
            _, next_obs_noise_pred = self.noise_pred_net(
                obs_feat, action_sample, action_t, next_obs_sample, t
            )
            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample
        return next_obs_sample

    @torch.no_grad()
    def sample_marginal_action(self, obs_dict):
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)

        # Initialize action and next_obs
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_obs_t = self.noise_scheduler.timesteps[0]
        for t in self.noise_scheduler.timesteps:
            action_noise_pred, _ = self.noise_pred_net(
                obs_feat, action_sample, t, next_obs_sample, next_obs_t
            )
            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample
        return action_sample

    @torch.no_grad()
    def sample_joint(self, obs_dict):
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)

        # Initialize action and next_obs
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            action_noise_pred, next_obs_noise_pred = self.noise_pred_net(
                obs_feat, action_sample, t, next_obs_sample, t
            )
            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample
            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample
        return next_obs_sample, action_sample
