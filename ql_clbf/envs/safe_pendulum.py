import numpy as np

from gym import spaces
from gym.envs.classic_control import PendulumEnv

class SafePendulumEnv(PendulumEnv):

    def __init__(self, g=10.0):
        super().__init__(g=g)
        self.action_space = spaces.Discrete(3)
        
    def is_goal(self, states: np.ndarray) -> np.ndarray:
        xs = states[..., 0]
        ys = states[..., 1]
        thetas = np.arctan2(ys, xs)
        return np.abs(thetas) < 0.1
    
    def is_unsafe(self, states: np.ndarray) -> np.ndarray:
        """ Unsafe if in the top right quadrant """
        xs = states[..., 0]
        ys = states[..., 1]
        return np.logical_and(ys > 0, xs < 0) 

    def step(self, u: int):
        if u == 0:
            force = -self.max_torque
        elif u == 1:
            force = 0
        elif u == 2:
            force = self.max_torque
        else:
            raise ValueError(f"Invalid action {u}")
        
        observation, _, _, info = super().step([force])
        is_goal = self.is_goal(observation) 
        is_terminated = self.is_unsafe(observation)

        if is_goal:
            reward = 1.0
        elif is_terminated:
            reward = -1.0
        else: 
            reward = 0.0

        done = np.logical_or(is_goal, is_terminated)
        return observation, reward, done, info
        
        