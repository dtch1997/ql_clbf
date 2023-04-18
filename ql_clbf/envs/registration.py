from gym.envs.registration import register

register(
    'SafeCartPole-v1',
    entry_point='ql_clbf.envs.safe_cartpole:SafeCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)