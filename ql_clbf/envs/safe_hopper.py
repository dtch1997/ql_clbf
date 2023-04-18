import numpy as np

from gym.envs.mujoco.hopper_v3 import HopperEnv
from ql_clbf.envs.env_interface import SafetyEnv, EnvHParams
from ql_clbf.utils.gym_utils import get_flattened_dim

class SafeHopperEnv(HopperEnv, SafetyEnv):

    def get_hparams(self):
        return EnvHParams(
            observation_dim=get_flattened_dim(self.observation_space),
            action_dim=get_flattened_dim(self.action_space),
            r_max=3,
        )
    
    def step(self, action):
        obs, base_reward, done, info = super().step(action)
        # Cap reward at 3
        reward = min(base_reward, 3)
        return obs, reward, done, info

    def sample_states(self, n_samples: int):
        high = 3 * np.ones_like(self.observation_space.high)
        low = -high
        return np.random.uniform(
            low=low,
            high=high,
            size=(n_samples, self.observation_space.shape[0]),
        ).astype(self.observation_space.dtype)

    def is_unsafe(self, states: np.ndarray) -> np.ndarray:

        # 0-th dimension is z-position of head
        z = states[:, 0]
        # 1-th dimension is angle of head
        angle = states[:, 1]
        state = states[:, 1:]
        
        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = ((min_state < state) * (state < max_state)).all(axis=1)
        healthy_z = (min_z < z) * (z < max_z)
        healthy_angle = (min_angle < angle) * (angle < max_angle)

        is_healthy = healthy_state * healthy_z * healthy_angle
        return ~is_healthy
