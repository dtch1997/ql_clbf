from gym.envs.registration import register


from gymnasium.envs.registration import register as register_gymnasium
from gymnasium_robotics.envs.maze import maps
from ql_clbf.envs.safety_point_maze import SAFETY_OPEN

register(
    'SafeCartPole-v1',
    entry_point='ql_clbf.envs.safe_cartpole:SafeCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'DiverseCArtPole-v1',
    entry_point='ql_clbf.envs.safe_cartpole:DiverseCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'NegSafeCartPole-v1',
    entry_point='ql_clbf.envs.safe_cartpole:NegSafeCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=0,
)

register(
    'SafePendulum-v1',
    entry_point='ql_clbf.envs.safe_pendulum:SafePendulumEnv',
    max_episode_steps=200,
    reward_threshold=1.0,
)

register(
    'SafeHopper-v3',
    entry_point='ql_clbf.envs.safe_hopper:SafeHopperEnv',
    max_episode_steps=1000,
    reward_threshold=3000.0,
)

register(
    'CartPoleA-v1',
    entry_point='ql_clbf.envs.decomposed_cartpole:CartPoleAEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'CartPoleB-v1',
    entry_point='ql_clbf.envs.decomposed_cartpole:CartPoleBEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'CartPoleC-v1',
    entry_point='ql_clbf.envs.decomposed_cartpole:CartPoleCEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'CartPoleD-v1',
    entry_point='ql_clbf.envs.decomposed_cartpole:CartPoleDEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register_gymnasium(
    id='SafetyPointMaze_Open-v3',
    entry_point='ql_clbf.envs.safety_point_maze:SafetyPointMazeEnv',
    max_episode_steps=500,
    reward_threshold=0.0,
    kwargs={
        "reward_type": 'sparse',
        'maze_map': SAFETY_OPEN
    },
)