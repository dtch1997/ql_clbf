import random
from typing import Callable

import gym
import numpy as np

import torch
import torch.nn as nn
from typing import List

from ql_clbf.learning.env_utils import make_env
from ql_clbf.net.q_network import QNetwork, QNetworkEnsemble

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
    envs = gym.make('CartPole-v1')
    obs = envs.reset()
    episodic_returns = []

    # Roll out model
    state = envs.reset()
    done = False
    states = []
    actions = []
    values = []
    for i in range(500):
        envs.render()
        states.append(state)
        action = model.predict([state]).item()
        value = model.predict_value([state], [action]).item()
        values.append(value)
        actions.append(action)
        state, _, done, _ = envs.step(action)
    return episodic_returns

if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-paths', nargs='+', type=str, required=True)
    args = parser.parse_args()

    envs = load_env('CartPole-v1', 'eval', capture_video=False)
    model_ensemble = load_ensemble_model(envs, args.model_paths)

    eval_episodes = 10

    if model_ensemble.get_num_models() > 1:
        for i, model in enumerate(model_ensemble.models):
            print(f"evaluating model {i}")
            # Reload envs to change video recording directory
            envs = load_env('CartPole-v1', f'eval_primitive_{i}', capture_video=True)
            returns = evaluate(model, envs, eval_episodes)
            print(f"returns={returns}")

    print("evaluating ensemble model")
    # Reload envs to change video recording directory
    envs = load_env('CartPole-v1', f'eval_ensemble', capture_video=True)
    ensemble_return = evaluate(model_ensemble, envs, eval_episodes)
    print(f"ensemble_return={ensemble_return}")