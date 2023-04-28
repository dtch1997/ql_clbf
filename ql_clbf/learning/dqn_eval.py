import random
from typing import Callable

import gym
import numpy as np
import pandas as pd

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

    def get_num_models(self):
        return len(self.models)

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
    envs = gym.vector.SyncVectorEnv([
        make_env(env_id, 0, 0, capture_video, run_name, 
                 record_obs_act_hist=True)
    ])
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

def create_results_df() -> pd.DataFrame:
    # Define the column names
    column_names = [
        'episode_index', 
        'episode_return', 
        'episode_length', 
        'observations', 
        'actions',
        'rewards',
        'next_observations',
        'r_min',
        'r_max',
        'r_term'    
    ]
    # Initialize an empty DataFrame with the specified columns
    dataframe = pd.DataFrame(columns=column_names)
    return dataframe

def postprocess_results(
        results: pd.DataFrame,
        model: nn.Module,
        envs: gym.vector.SyncVectorEnv,
        device: torch.device = torch.device("cpu"),
    ) -> pd.DataFrame:

    run_model_inference = lambda obs: model(torch.Tensor(obs).to(device)).cpu().detach().numpy()
    get_state_values = lambda q: np.max(q, axis=1)
    get_optimal_actions = lambda q: np.argmax(q, axis=1)

    results['q_values'] = results['observations'].apply(run_model_inference)
    results['next_q_values'] = results['next_observations'].apply(run_model_inference)
    results['optimal_actions'] = results['q_values'].apply(get_optimal_actions)
    results['state_values'] = results['q_values'].apply(get_state_values)
    results['next_state_values'] = results['next_q_values'].apply(get_state_values)

    results['alpha'] = 1.0
    results['R'] = results['r_max'] / (1 - results['gamma'])
    results['h_values'] = - results['state_values'] + results['R']
    results['h_next_values'] = - results['next_state_values'] + results['R']

    return results

def calculate_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """ Add relevant metrics to the results dataframe """
    def calculate_validity(row):
        h_curr = row['h_values']
        h_next = row['h_next_values']
        alpha = row['alpha']
        validity = h_next <= (1-alpha) * h_curr 
        return validity.astype(float)

    validity = results.apply(calculate_validity, axis=1)
    results['validity'] = validity
    results['validity_rate'] = results['validity'].apply(np.mean)
    return results

def evaluate(
    model: nn.Module, 
    envs: gym.vector.SyncVectorEnv,
    eval_episodes: int,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
):
    obs = envs.reset()

    results = create_results_df()
    hparams = envs.call('get_hparams')[0]

    while len(results) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, infos = envs.step(actions)
        for info in infos:
            if "episode" in info.keys():
                print(f"eval_episode={len(results)}, episodic_return={info['episode']['r']}")
                
                observations = np.array(info['observation_history'][:-1])
                is_observation_unsafe = np.array([envs.call('is_unsafe', obs) for obs in observations])
                episode_stats = pd.DataFrame({
                    'episode_index': [len(results)],
                    'episode_return': [info["episode"]["r"]],
                    'episode_length': [info["episode"]["l"]],
                    'observations': [info['observation_history'][:-1]],
                    'actions': [info['action_history']],
                    'rewards': [info['reward_history']],
                    'next_observations': [info['observation_history'][1:]],
                    'r_min': [hparams.r_min],
                    'r_max': [hparams.r_max],
                    'r_term': [hparams.r_term],
                    'is_observation_safe': [~is_observation_unsafe],
                    'gamma': [hparams.gamma],
                })
                results = pd.concat([results, episode_stats], ignore_index=True)

        obs = next_obs

    results = postprocess_results(results, model, envs, device)
    results = calculate_metrics(results)
    return results

if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, required=True)
    parser.add_argument('--model-paths', nargs='+', type=str, required=True)
    parser.add_argument('--capture-video', action='store_true')
    args = parser.parse_args()

    envs = load_env(args.env_id, 'eval', capture_video=False)
    model_ensemble = load_ensemble_model(envs, args.model_paths)

    eval_episodes = 100

    if len(args.model_paths) > 1:
        for i, model in enumerate(model_ensemble.models):
            print(f"evaluating model {i}")
            # Reload envs to change video recording directory
            envs = load_env(args.env_id, f'eval_primitive_{i}', capture_video=args.capture_video)
            returns = evaluate(model, envs, eval_episodes)
            print(f"returns: {returns}")

        print("evaluating ensemble model")
        # Reload envs to change video recording directory
        envs = load_env(args.env_id, f'eval_ensemble', capture_video=args.capture_video)
        ensemble_results = evaluate(model_ensemble, envs, eval_episodes, epsilon=0.0)
        print(f"ensemble_return={ensemble_results}")


    if len(args.model_paths) == 1:
        envs = load_env(args.env_id, f'eval_ensemble', capture_video=args.capture_video)
        model = load_model(envs, args.model_paths[0])
        ensemble_results = evaluate(model_ensemble, envs, eval_episodes, epsilon=0.0)
        print(f"return={ensemble_results}")        

    import matplotlib.pyplot as plt
    # Plot state-value trajectories
    fig, ax = plt.subplots()
    time = np.arange(0, len(ensemble_results['h_values'].iloc[0]))
    for i in range(eval_episodes):
        ax.plot(time, ensemble_results['h_values'].iloc[i], label=f'episode {i}')
    fig.show()

    input("Press Enter to continue...")
