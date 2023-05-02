import numpy as np

import math
import numpy as np
import xml.etree.ElementTree as ET
import tempfile
from typing import List, Union, Tuple, Optional
from os import path

from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.maze.maze import MazeEnv, Maze, U_MAZE, RESET, GOAL, COMBINED
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
from gymnasium_robotics.envs.maze.maps import U_MAZE
from gymnasium_robotics.envs.maze.maze import MazeEnv
from gymnasium_robotics.envs.maze.point import PointEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

UNSAFE = "u"  # Add a constant for unsafe tiles

class UnsafeMaze(Maze):

    """ Maze environment with unsafe tiles.
    
    Add property for unique unsafe locations
    Modified make_maze to handle unsafe tiles
    """

    def __init__(
        self,
        maze_map: List[List[Union[str, int]]],
        maze_size_scaling: float,
        maze_height: float,
    ):
        super().__init__(maze_map, maze_size_scaling, maze_height)
        self._unique_unsafe_locations = []

    @property
    def unique_unsafe_locations(self) -> List[np.ndarray]:
        """Returns all the possible unsafe locations in discrete cell
        coordinates (i,j)
        """
        return self._unique_unsafe_locations

    @classmethod
    def make_maze(
        cls,
        agent_xml_path: str,
        maze_map: list,
        maze_size_scaling: float,
        maze_height: float,
    ):
        tree = ET.parse(agent_xml_path)
        worldbody = tree.find(".//worldbody")

        maze = cls(maze_map, maze_size_scaling, maze_height)
        empty_locations = []
        for i in range(maze.map_length):
            for j in range(maze.map_width):
                struct = maze_map[i][j]
                # Store cell locations in simulation global Cartesian coordinates
                x = (j + 0.5) * maze_size_scaling - maze.x_map_center
                y = maze.y_map_center - (i + 0.5) * maze_size_scaling
                if struct == 1:  # Unmovable block.
                    # Offset all coordinates so that maze is centered.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {maze_height / 2 * maze_size_scaling}",
                        size=f"{0.5 * maze_size_scaling} {0.5 * maze_size_scaling} {maze_height / 2 * maze_size_scaling}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.7 0.5 0.3 1.0",
                    )
                elif struct == RESET:
                    maze._unique_reset_locations.append(np.array([x, y]))
                elif struct == GOAL:
                    maze._unique_goal_locations.append(np.array([x, y]))
                elif struct == COMBINED:
                    maze._combined_locations.append(np.array([x, y]))
                elif struct == 0:
                    empty_locations.append(np.array([x, y]))
                elif struct == UNSAFE:  # Handle the new unsafe tile type
                    maze._unique_unsafe_locations.append(np.array([x, y]))

        # Add target site for visualization
        ET.SubElement(
            worldbody,
            "site",
            name="target",
            pos=f"0 0 {maze_height / 2 * maze_size_scaling}",
            size=f"{0.2 * maze_size_scaling}",
            rgba="1 0 0 0.7",
            type="sphere",
        )

        # Add the combined cell locations (goal/reset) to goal and reset
        if (
            not maze._unique_goal_locations
            and not maze._unique_reset_locations
            and not maze._combined_locations
        ):
            # If there are no given "r", "g" or "c" cells in the maze data structure,
            # any empty cell can be a reset or goal location at initialization.
            maze._combined_locations = empty_locations
        maze._unique_goal_locations += maze._combined_locations
        maze._unique_reset_locations += maze._combined_locations

        # Save new xml with maze to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_xml_path = path.join(path.dirname(tmp_dir), "ant_maze.xml")
            tree.write(temp_xml_path)

        return maze, temp_xml_path

class SafetyMazeEnv(MazeEnv):

    """ 
    
    Use UnsafeMaze instead of Maze 
    Modified termination condition
    Modified reward 
    """
    def __init__(
        self,
        agent_xml_path: str,
        reward_type: str = "dense",
        continuing_task: bool = True,
        maze_map: List[List[Union[int, str]]] = U_MAZE,
        maze_size_scaling: float = 1.0,
        maze_height: float = 0.5,
        position_noise_range: float = 0.25,
        **kwargs,
    ):

        self.reward_type = reward_type
        self.continuing_task = continuing_task
        self.maze, self.tmp_xml_file_path = UnsafeMaze.make_maze(
            agent_xml_path, maze_map, maze_size_scaling, maze_height
        )

        self.position_noise_range = position_noise_range

    def is_unsafe(self):
        """Returns True if the agent is in an unsafe location"""
        return self.maze.current_cell in self.maze.unique_unsafe_locations

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        reward = super().compute_reward(achieved_goal, desired_goal, info)
        if self.is_unsafe():
            reward = -1
        return reward

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        terminated = super().compute_terminated(achieved_goal, desired_goal, info)
        if self.is_unsafe():
            terminated = True
        return terminated
    

class SafetyPointMazeEnv(SafetyMazeEnv, PointMazeEnv):
    """ PointMazeEnv with unsafe tiles

    Use UnsafeMaze instead of Maze
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        maze_map: List[List[Union[str, int]]] = U_MAZE,
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        **kwargs,
    ):
        point_xml_file_path = path.join(
            path.dirname(path.realpath(__file__)), "../assets/point/point.xml"
        )
        SafetyMazeEnv.__init__(
            self,
            agent_xml_path=point_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=1,
            maze_height=0.4,
            reward_type=reward_type,
            continuing_task=continuing_task,
            **kwargs,
        )

        maze_length = len(maze_map)
        default_camera_config = {"distance": 12.5 if maze_length > 8 else 8.8}

        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        obs_shape: tuple = self.point_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs_shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            maze_map,
            render_mode,
            reward_type,
            continuing_task,
            **kwargs,
        )