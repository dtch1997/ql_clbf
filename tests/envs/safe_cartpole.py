import numpy as np

import gym 
import ql_clbf.envs

def test_env_registration():
    env = gym.make('SafeCartPole-v1')

def test_sample_states():
    env = gym.make('SafeCartPole-v1')
    states = env.sample_states(100)
    assert states.shape == (100, 4)

    for i, state in enumerate(states):
        assert env.observation_space.contains(state), f"State {i} is out of bounds: {state}"

def test_is_unsafe():
    env = gym.make('SafeCartPole-v1')
    states = env.sample_states(100)
    assert states.shape == (100, 4)
    is_unsafe = env.is_unsafe(states)
    assert is_unsafe.shape == (100,)