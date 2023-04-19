import random
from typing import Callable

import gym
import numpy as np

import torch
import torch.nn as nn
from typing import List

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

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs = envs.reset()
    episodic_returns = []
    print(obs)
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

def evaluate_ensemble(
    model_paths: List[str],
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    
    models = []
    for model_path in model_paths:
        _model = Model(envs).to(device)
        _model.load_state_dict(torch.load(model_path, map_location=device))
        models.append(_model)
    model = QNetworkEnsemble(envs, models).to(device)
    model.eval()

    obs = envs.reset()
    print(obs)
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
    from ql_clbf.learning.dqn import QNetwork, make_env

    model_paths = [
        'downloads/llxgy0q2/dqn.pth',
        'downloads/u2vzgk9i/dqn.pth',
    ]
    
    evaluate(
        model_paths[0],
        make_env,
        "CartPole-v1",
        eval_episodes=1,
        run_name=f"eval_A",
        Model=QNetwork,
        device="cpu",
        capture_video=True, 
    )

    evaluate(
        model_paths[1],
        make_env,
        "CartPole-v1",
        eval_episodes=1,
        run_name=f"eval_B",
        Model=QNetwork,
        device="cpu",
        capture_video=True,
    )
    
    evaluate_ensemble(
        model_paths,
        make_env,
        "CartPole-v1",
        eval_episodes=1,
        run_name=f"eval_AB",
        Model=QNetwork,
        device="cpu",
        capture_video=True,
    )