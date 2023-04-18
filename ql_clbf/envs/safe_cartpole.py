import numpy as np

from gym.envs.classic_control import CartPoleEnv
from ql_clbf.envs.env_interface import SafetyEnv, EnvHParams
from ql_clbf.utils.gym_utils import get_flattened_dim

class SafeCartPoleEnv(CartPoleEnv, SafetyEnv):

    def get_hparams(self):
        return EnvHParams(
            observation_dim=get_flattened_dim(self.observation_space),
            action_dim=get_flattened_dim(self.action_space),
            r_max=1.0,
        )

    def sample_states(self, n_samples: int):
        x_high = 2 * self.x_threshold
        theta_high = 2 * self.theta_threshold_radians
        high = np.array([x_high, 2, theta_high, 2])
        low = -high
        return np.random.uniform(
            low=low, 
            high=high, 
            size=(n_samples, self.observation_space.shape[0]),
        ).astype(self.observation_space.dtype)
    
    def is_unsafe(self, states: np.ndarray) -> np.ndarray:
        is_safe = (
            (states[:, 0] > -self.x_threshold) *
            (states[:, 0] < self.x_threshold) *
            (states[:, 2] > -self.theta_threshold_radians) *
            (states[:, 2] < self.theta_threshold_radians)
        )
        return np.logical_not(is_safe).reshape(-1, 1)
