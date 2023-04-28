import numpy as np

from ql_clbf.envs.safe_cartpole import SafeCartPoleEnv

class CartPoleAEnv(SafeCartPoleEnv):

    def is_done(self, states: np.ndarray) -> np.ndarray:
        is_safe = (
            (states[..., 0] > -self.x_threshold) *
            (states[..., 0] < self.x_threshold)
        )
        return ~is_safe
    
class CartPoleBEnv(SafeCartPoleEnv):
    
    def is_done(self, states: np.ndarray) -> np.ndarray:
        is_safe = (
            (states[..., 2] > -self.theta_threshold_radians) *
            (states[..., 2] < self.theta_threshold_radians)
        )
        return ~is_safe
    
class CartPoleCEnv(SafeCartPoleEnv):

    def is_done(self, states: np.ndarray) -> np.ndarray:
        is_safe = (
            (states[..., 0] > -self.x_threshold) *
            (states[..., 2] > -self.theta_threshold_radians)
        )
        return ~is_safe
    
class CartPoleDEnv(SafeCartPoleEnv):
    
    def is_done(self, states: np.ndarray) -> np.ndarray:
        is_safe = (
            (states[..., 0] < self.x_threshold) *
            (states[..., 2] < self.theta_threshold_radians)
        )
        return ~is_safe