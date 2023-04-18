import numpy as np

import gym 
import ql_clbf.envs

def test_env_registration():
    env = gym.make('SafeHopper-v3')

def test_sample_states():
    env = gym.make('SafeHopper-v3')
    states = env.sample_states(100)
    assert states.shape == (100, 11)

    for i, state in enumerate(states):
        assert env.observation_space.contains(state), f"State {i} is out of bounds: {state}"

def test_is_unsafe():
    env = gym.make('SafeHopper-v3')
    states = env.sample_states(100)
    is_unsafe = env.is_unsafe(states)
    assert is_unsafe.shape == (100,)

    safe_state = np.zeros((1, 11))
    safe_state[0, 0] = 1.0
    assert not env.is_unsafe(safe_state)