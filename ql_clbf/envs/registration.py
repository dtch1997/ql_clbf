from gym.envs.registration import register

register(
    'SafeCartPole-v1',
    entry_point='ql_clbf.envs.safe_cartpole:SafeCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'SafeHopper-v3',
    entry_point='ql_clbf.envs.safe_hopper:SafeHopperEnv',
    max_episode_steps=1000,
    reward_threshold=3000.0,
)