from functools import partial

from .obs_encoder import ImageObservationEncoder
from .base_policy import NoisePredictionNet, DiffusionPolicy, FlowPolicy


class ImageDiffusionPolicy(DiffusionPolicy):
    def __init__(
        self,
        action_len: int,
        action_dim: int,
        obs_encoder: ImageObservationEncoder,
        noise_pred_net: partial[NoisePredictionNet],
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        num_train_noise_samples: int = 1,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
    ):
        """
        Assumes rgb input: (B, T, H, W, C) uint8 image
        Assumes low_dim input: (B, T, D)
        """
        super().__init__(
            action_len=action_len,
            action_dim=action_dim,
            noise_pred_net=noise_pred_net(global_cond_dim=obs_encoder.output_len),
            num_train_steps=num_train_steps,
            num_inference_steps=num_inference_steps,
            num_train_noise_samples=num_train_noise_samples,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )

        # Observation encoder
        self.obs_encoder = obs_encoder

    def sample(self, obs_dict):
        obs = self.obs_encoder(obs_dict)
        action = super().sample(obs)
        return action

    def forward(self, obs_dict, action):
        obs = self.obs_encoder(obs_dict)
        loss = super().forward(obs, action)
        return loss


class ImageFlowPolicy(FlowPolicy):
    def __init__(
        self,
        action_len: int,
        action_dim: int,
        obs_encoder: ImageObservationEncoder,
        noise_pred_net: partial[NoisePredictionNet],
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        timeshift: float = 1.0,
    ):
        """
        Assumes rgb input: (B, T, H, W, C) uint8 image
        Assumes low_dim input: (B, T, D)
        """
        super().__init__(
            action_len=action_len,
            action_dim=action_dim,
            noise_pred_net=noise_pred_net(global_cond_dim=obs_encoder.output_len),
            num_train_steps=num_train_steps,
            num_inference_steps=num_inference_steps,
            timeshift=timeshift,
        )

        # Observation encoder
        self.obs_encoder = obs_encoder

    def sample(self, obs_dict):
        obs = self.obs_encoder(obs_dict)
        action = super().sample(obs)
        return action

    def forward(self, obs_dict, action):
        obs = self.obs_encoder(obs_dict)
        loss = super().forward(obs, action)
        return loss
