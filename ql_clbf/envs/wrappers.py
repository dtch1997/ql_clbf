import gym
import numpy as np

class RecordObservationActionHistory(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.observation_history = [[] for _ in range(self.num_envs)]
        self.action_history = [[] for _ in range(self.num_envs)]

    def add_observation(self, idx, observation):
        self.observation_history[idx].append(observation)

    def add_action(self, idx, action):
        self.action_history[idx].append(action)

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        observations_orig = observations
        self.observation_history = [[] for i in range(self.num_envs)]
        self.action_history = [[] for _ in range(self.num_envs)]

        if not self.is_vector_env:
            observations = [observations]

        for i in range(len(observations)):
            self.add_observation(i, observations[i])

        return observations_orig

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        observations_orig = observations

        if not self.is_vector_env:
            action = [action]
            infos = [infos]
            dones = [dones]
            observations = [observations]

        for i in range(len(dones)):
            self.add_observation(i, observations[i])
            self.add_action(i, action[i])
            if dones[i]:
                infos[i] = infos[i].copy()
                infos[i]["observation_history"] = np.array(self.observation_history[i])
                infos[i]["action_history"] = np.array(self.action_history[i])

                # Reset the history for the next episode
                self.observation_history[i] = []
                self.action_history[i] = []

        if self.is_vector_env:
            infos = tuple(infos)
            observations = np.array(observations)
            dones = np.array(dones)

        return (
            observations_orig,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )
