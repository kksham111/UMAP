from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class NoisePredictionNet(nn.Module, ABC):

    @abstractmethod
    def forward(self, sample, timestep, global_cond):
        raise NotImplementedError


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_len,
        action_dim,
        noise_pred_net,
        num_train_steps=100,
        num_inference_steps=10,
        num_train_noise_samples=1,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
    ):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.num_train_noise_samples = num_train_noise_samples

        # Noise prediction net
        assert isinstance(noise_pred_net, NoisePredictionNet)
        self.noise_pred_net = noise_pred_net

        # Noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )

    @torch.no_grad()
    def sample(self, obs):
        # Initialize sample
        action = torch.randn(
            (obs.shape[0], self.action_len, self.action_dim), device=obs.device
        )

        # Initialize scheduler
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Reverse diffusion process
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.noise_pred_net(action, t, global_cond=obs)

            # Diffusion step
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        return action

    def forward(self, obs, action):
        # Repeat observations and actions for multiple noise samples
        if self.num_train_noise_samples > 1:
            obs = obs.repeat_interleave(self.num_train_noise_samples, dim=0)
            action = action.repeat_interleave(self.num_train_noise_samples, dim=0)

        # Sample random noise
        noise = torch.randn_like(action)

        # Sample a random timestep
        t = torch.randint(
            low=0,
            high=self.num_train_steps,
            size=(action.shape[0],),
            device=action.device,
        ).long()

        # Forward diffusion step
        noisy_action = self.noise_scheduler.add_noise(action, noise, t)

        # Diffusion loss
        noise_pred = self.noise_pred_net(noisy_action, t, global_cond=obs)
        loss = F.mse_loss(noise_pred, noise)
        return loss


class FlowPolicy(nn.Module):
    def __init__(
        self,
        action_len,
        action_dim,
        noise_pred_net,
        num_train_steps=100,
        num_inference_steps=10,
        timeshift=1.0,
    ):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim

        # Noise prediction net
        assert isinstance(noise_pred_net, NoisePredictionNet)
        self.noise_pred_net = noise_pred_net

        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        timesteps = torch.linspace(1, 0, self.num_inference_steps + 1)
        self.timesteps = (timeshift * timesteps) / (1 + (timeshift - 1) * timesteps)

    @torch.no_grad()
    def sample(self, obs):
        # Initialize sample
        action = torch.randn(
            (obs.shape[0], self.action_len, self.action_dim), device=obs.device
        )

        for tcont, tcont_next in zip(self.timesteps[:-1], self.timesteps[1:]):
            # Predict noise
            t = (tcont * self.num_train_steps).long()
            noise_pred = self.noise_pred_net(action, t, global_cond=obs)

            # Flow step
            action = action + (tcont_next - tcont) * noise_pred

        return action

    def forward(self, obs, action):
        # Sample random noise
        noise = torch.randn_like(action)

        # Sample random timestep
        tcont = torch.rand((action.shape[0],), device=action.device)

        # Forward flow step
        direction = noise - action
        noisy_action = (
            action + tcont.view(-1, *[1 for _ in range(action.dim() - 1)]) * direction
        )

        # Flow matching loss
        t = (tcont * self.num_train_steps).long()
        noise_pred = self.noise_pred_net(noisy_action, t, global_cond=obs)
        loss = F.mse_loss(noise_pred, direction)
        return loss
