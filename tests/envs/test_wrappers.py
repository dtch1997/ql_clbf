import gym
import numpy as np

from ql_clbf.envs.wrappers import RecordObservationActionHistory

def test_record_observation_action_history():
    env = gym.make("CartPole-v0")
    wrapped_env = RecordObservationActionHistory(env)

    # Test for 5 episodes
    num_episodes = 5
    for episode in range(num_episodes):
        obs = wrapped_env.reset()
        done = False

        # Variables to store the observation and action history
        obs_history = []
        action_history = []
        obs_history.append(obs)

        while not done:
            action = wrapped_env.action_space.sample()
            action_history.append(action)

            obs, reward, done, info = wrapped_env.step(action)
            obs_history.append(obs)

            if done:                
                print(info['observation_history'].shape)
                # Check if the observation history matches the info provided by the wrapper
                assert np.array_equal(info["observation_history"], np.array(obs_history))

                # Check if the action history matches the info provided by the wrapper
                assert np.array_equal(info["action_history"], np.array(action_history))

    wrapped_env.close()

# Run the test
test_record_observation_action_history()
