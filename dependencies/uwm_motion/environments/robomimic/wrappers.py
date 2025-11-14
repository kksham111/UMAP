from collections import deque

import numpy as np


class RoboMimicEnvWrapper:
    def __init__(
        self,
        env,
        obs_keys,
        obs_horizon,
        max_episode_length,
        record=False,
        render_size=(224, 224),
    ):
        self.env = env
        self.obs_keys = obs_keys
        self.obs_buffer = deque(maxlen=obs_horizon)

        self._max_episode_length = max_episode_length
        self._elapsed_steps = None

        self.record = record
        self.render_size = render_size
        if record:
            self.video_buffer = deque()

    def _is_success(self):
        return self.env.is_success()["task"]

    def _get_obs(self):
        # Return a dictionary of stacked observations
        stacked_obs = {}
        for key in self.obs_keys:
            stacked_obs[key] = np.stack([obs[key] for obs in self.obs_buffer])
        return stacked_obs

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        # Clear buffers
        self.obs_buffer.clear()
        if self.record:
            self.video_buffer.clear()

        # Reset environment
        obs = self.env.reset()
        self._elapsed_steps = 0

        # Pad observation buffer
        for _ in range(self.obs_buffer.maxlen):
            self.obs_buffer.append(obs)

        return self._get_obs()

    def step(self, actions):
        # Roll out a sequence of actions in the environment
        total_reward = 0
        for action in actions:
            # Step environment
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self.obs_buffer.append(obs)
            if self.record:
                self.video_buffer.append(self.render())

            # Store success info
            info["success"] = self._is_success()

            # Terminate on success
            done = done or info["success"]

            # Terminate if max episode length is reached
            self._elapsed_steps += 1
            if self._elapsed_steps >= self._max_episode_length:
                info["truncated"] = not done
                done = True

            if done:
                break

        return self._get_obs(), total_reward, done, info

    def render(self):
        return self.env.render(
            mode="rgb_array",
            width=self.render_size[0],
            height=self.render_size[1],
        )

    def get_video(self):
        if not self.record:
            raise ValueError("Video recording is disabled.")
        return np.stack(self.video_buffer)

    def close(self):
        self.env.close()


class LIBEROEnvWrapper(RoboMimicEnvWrapper):
    def __init__(
        self,
        env,
        obs_keys,
        obs_horizon,
        max_episode_length,
        record=False,
        render_size=(224, 224),
    ):
        super().__init__(
            env,
            obs_keys,
            obs_horizon,
            max_episode_length,
            record,
            render_size,
        )
        self.source_key_map = {
            "agentview_rgb": "agentview_image",
            "eye_in_hand_rgb": "robot0_eye_in_hand_image",
        }

    def reset_to_state(self, state):
        self.seed(0)
        self.reset()
        self.env.set_init_state(state)

        # Refresh obs buffer
        self.obs_buffer.clear()
        obs = self.env.env._get_observations()
        for _ in range(self.obs_buffer.maxlen):
            self.obs_buffer.append(obs)
        return self._get_obs()

    def _is_success(self):
        return self.env.check_success()

    def _get_obs(self):
        # Return a dictionary of stacked observations
        stacked_obs = {}
        for key in self.obs_keys:
            source_key = self.source_key_map.get(key, key)
            stacked_obs[key] = np.stack([obs[source_key] for obs in self.obs_buffer])

        # Flip all image observations
        for key in self.obs_keys:
            if len(stacked_obs[key].shape) == 4:
                stacked_obs[key] = stacked_obs[key][:, ::-1].copy()
        return stacked_obs

    def render(self):
        img = self.env.env.sim.render(
            height=self.render_size[1],
            width=self.render_size[0],
            camera_name="frontview",
        )
        return img[::-1]
