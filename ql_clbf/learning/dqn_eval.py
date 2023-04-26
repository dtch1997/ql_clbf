import random
from typing import Callable

import gym
import numpy as np

import torch
import torch.nn as nn
from typing import List

from ql_clbf.learning.env_utils import make_env
from ql_clbf.net.q_network import QNetwork

class QNetworkEnsemble(nn.Module):
    def __init__(self, envs, models: List[nn.Module]):
        super(QNetworkEnsemble, self).__init__()
        self.envs = envs
        self.models = nn.ModuleList(models)

    def forward(self, x, reduction: str ='min'):
        assert reduction in ['min', 'max', 'mean']
        q_values = torch.stack([model(x) for model in self.models], dim=0)
        if reduction == 'min':
            return torch.min(q_values, dim=0)[0]
        elif reduction == 'max':
            return torch.max(q_values, dim=0)[0]
        elif reduction == 'mean':
            return torch.mean(q_values, dim=0)

def load_env(
    env_id: str,
    run_name: str,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    return envs

def load_model(    
    envs: gym.vector.SyncVectorEnv,
    model_path: str,
    device: torch.device = torch.device("cpu"),
):
    model = QNetwork(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model        

def load_ensemble_model(
    envs: gym.vector.SyncVectorEnv,
    model_paths: List[str],
    device: torch.device = torch.device("cpu"),
):
    models = []
    for model_path in model_paths:
        _model = QNetwork(envs).to(device)
        _model.load_state_dict(torch.load(model_path, map_location=device))
        models.append(_model)
    model = QNetworkEnsemble(envs, models).to(device)
    model.eval()
    return model

def evaluate(
    model: nn.Module, 
    envs: gym.vector.SyncVectorEnv,
    eval_episodes: int,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
):
    obs = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, infos = envs.step(actions)
        for info in infos:
            if "episode" in info.keys():
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

if __name__ == "__main__":

    model_paths = [
        'downloads/llxgy0q2/dqn.pth',
        'downloads/u2vzgk9i/dqn.pth',
    ]

    envs = load_env('CartPole-v1', 'eval', capture_video=False)
    model_A = load_model(envs, model_paths[0])
    model_B = load_model(envs, model_paths[1])
    model_AB = load_ensemble_model(envs, model_paths)

    eval_episodes = 10
    returns_A = evaluate(model_A, envs, eval_episodes)
    returns_B = evaluate(model_B, envs, eval_episodes)
    returns_AB = evaluate(model_AB, envs, eval_episodes)

    print(f"returns_A={returns_A}")
    print(f"returns_B={returns_B}")
    print(f"returns_AB={returns_AB}")