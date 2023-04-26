import numpy as np

from gym import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv

class GridWorld(MiniGridEnv):
    """ A simple NxN grid world with a single agent and a single goal."""

    def __init__(
        self, size, obstacle_type=Lava, max_steps: int = None, **kwargs
    ):
        self.obstacle_type = obstacle_type
        self.size = size

        if obstacle_type == Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
            mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

        self.observation_space = spaces.MultiDiscrete((size, size))
        self.action_space = spaces.Discrete(3)

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "find the opening and get to the green goal square"

    def _gen_grid(self, width, height):
        self.grid = Grid(7, 7)
        self.grid.wall_rect(0, 0, 7, 7)

        self.put_obj(Goal(), 5, 3)

        self.grid.set(2, 2, Lava())
        self.grid.set(2, 3, Lava())

        self.grid.set(4, 3, Lava())
        self.grid.set(4, 4, Lava())

        self.agent_pos = np.array([1, 3])
        self.agent_dir = 0

    def gen_obs(self):
        return self.agent_pos

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        return obs, reward, done, info

if __name__ == "__main__":
    env = GridWorld(7, render_mode='human')
    
    obs = env.reset()
    print(env.observation_space)
    print(env.action_space)
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print(action, obs, reward, done, info)
        if done:
            break