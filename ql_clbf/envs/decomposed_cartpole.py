import numpy as np
import math

from gym import spaces, logger
from gym.envs.classic_control import CartPoleEnv

class DecomposableCartPoleEnv(CartPoleEnv):

    def is_done(self, state):
        x, x_dot, theta, theta_dot = state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        return done

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

        done = self.is_done(self.state)

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

class CartPoleAEnv(DecomposableCartPoleEnv):

    def is_done(self, state):
        x, x_dot, theta, theta_dot = state
        done = x < -self.x_threshold \
            or x > self.x_threshold
        return done
    
class CartPoleBEnv(DecomposableCartPoleEnv):
    
    def is_done(self, state):
        x, x_dot, theta, theta_dot = state
        done = theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        return done
    
class CartPoleCEnv(DecomposableCartPoleEnv):

    def is_done(self, state):
        x, x_dot, theta, theta_dot = state
        done = x < -self.x_threshold \
            or theta < -self.theta_threshold_radians
        return done
    
class CartPoleDEnv(DecomposableCartPoleEnv):
    
    def is_done(self, state):
        x, x_dot, theta, theta_dot = state
        done = x > self.x_threshold \
            or theta > self.theta_threshold_radians
        return done