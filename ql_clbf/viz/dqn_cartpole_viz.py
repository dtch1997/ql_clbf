import argparse
import gym 
import torch 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from typing import Callable, List
from ql_clbf.learning.dqn_train import make_env, QNetwork
from ql_clbf.learning.dqn_eval import QNetworkEnsemble

def load_model(
    model_path: str,
    make_env: Callable,
    env_id: str,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, envs

def load_ensemble_model(
    model_paths: List[str],
    make_env: Callable,
    env_id: str,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
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
    return model, envs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str)
    parser.add_argument('--model-paths', type=str, nargs="+")
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('-R', '--reward-upper-bound', type=float, default=1.0, 
                        help="Upper bound on reward for environment")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    model, env = load_ensemble_model(
        args.model_paths,
        make_env,
        args.env_id,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False
    )
    
    def plot_heatmap_theta(fig, ax):

        # Visualize Q values
        obs_space = env.single_observation_space
        x_space = torch.zeros(1)
        x_dot_space = torch.zeros(1)
        theta_space = torch.linspace(-np.pi / 4, np.pi / 4, steps=100)
        theta_dot_space = torch.linspace(-2, 2, steps=100)

        grid_x, grid_x_dot, grid_theta, grid_theta_dot = torch.meshgrid(
            x_space, x_dot_space, theta_space, theta_dot_space, indexing='xy'
        )
        state_space = torch.stack([grid_x, grid_x_dot, grid_theta, grid_theta_dot], dim=-1)
        qsa_values = model(state_space)
        qs_values, _ = torch.max(qsa_values, dim=-1)

        X = theta_space
        Y = theta_dot_space
        Z = qs_values[0,0].detach().cpu().numpy()
        Z = - Z + args.reward_upper_bound / (1- args.gamma)
        levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('coolwarm')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        image = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
        fig.colorbar(image, ax=ax)
        theta_threshold = 12 * 2 * np.pi / 360
        ax.axvline(-theta_threshold, color='k', linestyle='--', linewidth=3)
        ax.axvline(theta_threshold, color='k', linestyle='--', linewidth=3)
        ax.set_title('Barrier function value')        
        ax.set_xlabel('theta')
        ax.set_ylabel('theta_dot')
        
        return ax
    
    def plot_heatmap_x(fig, ax):

        obs_space = env.single_observation_space
        x_space = torch.linspace(-4, 4, steps=100)
        x_dot_space = torch.linspace(-4, 4, steps=100)
        theta_space = torch.zeros(1)
        theta_dot_space = torch.zeros(1)

        grid_x, grid_x_dot, grid_theta, grid_theta_dot = torch.meshgrid(
            x_space, x_dot_space, theta_space, theta_dot_space, indexing='xy'
        )
        state_space = torch.stack([grid_x, grid_x_dot, grid_theta, grid_theta_dot], dim=-1)
        qsa_values = model(state_space)
        qs_values, _ = torch.max(qsa_values, dim=-1)

        X = x_space
        Y = x_dot_space
        Z = qs_values[:,:,0,0].detach().cpu().numpy()
        Z = - Z + args.reward_upper_bound / (1- args.gamma)
        levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('coolwarm')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        image = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
        fig.colorbar(image, ax=ax)
        x_threshold = -2.4
        ax.axvline(-x_threshold, color='k', linestyle='--', linewidth=3)
        ax.axvline(x_threshold, color='k', linestyle='--', linewidth=3)
        ax.set_title('Barrier function value')        
        ax.set_xlabel('x')
        ax.set_ylabel('x_dot')
        
        return ax

    fig, ax = plt.subplots(ncols=2, figsize=(25, 10))
    plot_heatmap_theta(fig, ax[0])
    plot_heatmap_x(fig, ax[1])

    fig.show()
    input("Press Enter to exit...")