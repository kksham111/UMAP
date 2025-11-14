import os
from collections import deque

import hydra
import torch
import numpy as np

from droid.controllers.oculus_controller import VRPolicy
from droid.data_processing.timestep_processing import TimestepProcesser
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI

from datasets.droid.utils import rot_6d_to_euler_angles


class DROIDAgent:
    def __init__(
        self,
        model,
        device,
        img_keys,
        action_normalizer,
        lowdim_normalizer,
        obs_horizon=2,
        action_horizon=8,
    ):
        self.model = model
        self.device = device
        self.img_keys = img_keys
        self.action_normalizer = action_normalizer
        self.lowdim_normalizer = lowdim_normalizer
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        assert obs_horizon == model.obs_encoder.num_frames
        assert action_horizon <= model.action_len

        self.action_scale = action_normalizer.scale[None]
        self.action_offset = action_normalizer.offset[None]

        # Observation buffer
        self.obs_buffer = deque(maxlen=obs_horizon)
        self.act_buffer = deque(maxlen=action_horizon)

        # Timestep processor
        self.timestep_processor = TimestepProcesser(
            ignore_action=True,
            action_space="cartesian_position",
            gripper_action_space="position",
            robot_state_keys=[
                "cartesian_position",
                "gripper_position",
                "joint_positions",
            ],
            image_transform_kwargs=dict(
                remove_alpha=True,
                bgr_to_rgb=True,
                to_tensor=False,
                augment=False,
            ),
        )

    def _convert_obs(self, observation):
        # Process camera images
        timestep = {"observation": observation}
        processed_timestep = self.timestep_processor.forward(timestep)
        camera_images = processed_timestep["observation"]["camera"]["image"]

        # Extract observations
        obs = {
            "exterior_image_1_left": camera_images["varied_camera"][0],
            "exterior_image_2_left": camera_images["varied_camera"][2],
            "wrist_image_left": camera_images["hand_camera"][0],
            "cartesian_position": observation["robot_state"]["cartesian_position"],
            "gripper_position": np.array(
                [observation["robot_state"]["gripper_position"]]
            ),
        }

        # Convert image observations to torch tensors
        for key in self.img_keys:
            obs[key] = torch.from_numpy(obs[key][None]).to(self.device)

        # Normalize low-dimensional observations and convert to torch tensors
        for key in self.lowdim_normalizer.keys():
            lowdim_obs = self.lowdim_normalizer[key](obs[key])
            obs[key] = torch.from_numpy(lowdim_obs[None]).float().to(self.device)

        return obs

    def _convert_action(self, action):
        xyz, rot6d, grippers = action[:3], action[3:9], action[9:]
        euler = rot_6d_to_euler_angles(torch.tensor(rot6d)).numpy()
        return np.concatenate([xyz, euler, grippers])

    @torch.no_grad()
    def forward(self, observation):
        # Encode observations
        obs = self._convert_obs(observation)

        # Update observation buffer
        if len(self.obs_buffer) == 0:
            # Pad observation buffer if empty (only after reset)
            for _ in range(self.obs_horizon):
                self.obs_buffer.append(obs)
        else:
            self.obs_buffer.append(obs)

        # Update action buffer
        if len(self.act_buffer) == 0:
            # Stack observations by key
            obs_seq = {}
            for key in obs.keys():
                obs_seq[key] = torch.stack([obs[key] for obs in self.obs_buffer], dim=1)

            # Sample actions
            act_seq = self.model.sample(obs_seq)
            act_seq = act_seq[0].cpu().numpy()
            act_seq = act_seq * self.action_scale + self.action_offset

            # Store new actions in buffer
            for t in range(self.action_horizon):
                self.act_buffer.append(self._convert_action(act_seq[t]))

        # Return next action
        action = self.act_buffer.popleft()
        # Clip action
        action = np.clip(action, -1, 1)
        return action

    def reset(self):
        self.obs_buffer.clear()
        self.act_buffer.clear()


@hydra.main(
    version_base=None, config_path="../../configs", config_name="train_uwm.yaml"
)
def main(config):
    device = torch.device(f"cuda:0")

    # Create model
    model = hydra.utils.instantiate(config.model).to(device)
    model.eval()

    # Load models
    ckpt = torch.load(os.path.join(config.logdir, "models.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"Loaded models from pretraining checkpoint, step: {ckpt['step']}")

    # Create agent
    img_keys = [
        k for k, v in config.dataset.shape_meta["obs"].items() if v["type"] == "rgb"
    ]
    agent = DROIDAgent(
        model=model,
        device=device,
        img_keys=img_keys,
        action_normalizer=ckpt["action_normalizer"],
        lowdim_normalizer=ckpt["lowdim_normalizer"],
        obs_horizon=config.model.obs_encoder.num_frames,
        action_horizon=config.model.action_len // 2,
    )

    # Create evaluation environment
    h, w = tuple(config.dataset.shape_meta["obs"][img_keys[0]]["shape"][:2])
    img_size = (w, h)  # flip width and height
    env = RobotEnv(
        action_space="cartesian_velocity",
        gripper_action_space="position",
        camera_kwargs=dict(
            hand_camera=dict(
                image=True,
                concatenate_images=False,
                resolution=img_size,
                resize_func="cv2",
            ),
            varied_camera=dict(
                image=True,
                concatenate_images=False,
                resolution=img_size,
                resize_func="cv2",
            ),
        ),
    )
    controller = VRPolicy()

    # Launch GUI
    data_collector = DataCollecter(
        env=env,
        controller=controller,
        policy=agent,
        save_traj_dir=os.path.join(config.logdir, "videos"),
        save_data=True,
    )
    RobotGUI(robot=data_collector)


if __name__ == "__main__":
    main()
