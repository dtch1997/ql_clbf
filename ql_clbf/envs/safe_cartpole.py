import math
import numpy as np

from gym import logger
from gym.envs.classic_control import CartPoleEnv
from ql_clbf.envs.env_interface import SafetyEnv, EnvHParams
from ql_clbf.utils.gym_utils import get_flattened_dim

class ModifiableCartPoleEnv(CartPoleEnv):
    """ CartPoleEnv
    
    Termination condition can be easily overwritten by subclasses 
    Otherwise the same as original CartPoleEnv
    """

    def is_done(self, states: np.ndarray) -> np.ndarray:
        is_safe = (
            (states[..., 0] > -self.x_threshold) *
            (states[..., 0] < self.x_threshold) *
            (states[..., 2] > -self.theta_threshold_radians) *
            (states[..., 2] < self.theta_threshold_radians)
        )
        return ~is_safe

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = self.is_done(np.array(self.state))

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

class SafeCartPoleEnv(ModifiableCartPoleEnv, SafetyEnv):

    def get_hparams(self):
        return EnvHParams(
            observation_dim=get_flattened_dim(self.observation_space),
            action_dim=get_flattened_dim(self.action_space),
            r_max=1.0,
            r_min=1.0,
            r_term=0.0,
            gamma=0.99,
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
        return self.is_done(states)
    
class NegSafeCartPoleEnv(SafeCartPoleEnv):

    def get_hparams(self):
        return EnvHParams(
            observation_dim=get_flattened_dim(self.observation_space),
            action_dim=get_flattened_dim(self.action_space),
            r_max=0.0,
            r_min=0.0,
            r_term=0.0,
            gamma=0.99,
        )
    
    def step(self, action):
        """ 
        Modified step function to return negative reward 
        Also do not terminate early.
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        is_unsafe = self.is_unsafe(np.array(self.state))
        if is_unsafe: reward = 0
        else: reward = -1

        return np.array(self.state, dtype=np.float32), reward, False, {}
        