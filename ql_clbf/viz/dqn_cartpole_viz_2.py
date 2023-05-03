import argparse
import gym 
import torch 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from typing import Callable, List
from ql_clbf.net.q_network import QNetwork, QNetworkEnsemble
from ql_clbf.learning.env_utils import make_env

def get_zero_spaces():
    """ Get zero spaces for plotting

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of x and theta space
    """
    x_space = torch.zeros(1)
    x_dot_space = torch.zeros(1)
    theta_space = torch.zeros(1)
    theta_dot_space = torch.zeros(1)
    return x_space, x_dot_space, theta_space, theta_dot_space

def get_default_spaces():
    """ Get default spaces for plotting

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of x and theta space
    """
    x_space = torch.linspace(-8, 8, steps=100)
    x_dot_space = torch.linspace(-8, 8, steps=100)
    theta_space = torch.linspace(-np.pi, np.pi, steps=100)
    theta_dot_space = torch.linspace(-4, 4, steps=100)
    return x_space, x_dot_space, theta_space, theta_dot_space

def plot_heatmap_theta(fig: plt.Figure, 
                        ax: plt.Axes, 
                        theta_space: torch.Tensor, 
                        theta_dot_space: torch.Tensor, 
                        barrier_values: torch.Tensor):
    """ Plot heatmap of state values for theta and theta_dot

    Args:
        fig (matplotlib.figure.Figure): Figure to plot on
        ax (matplotlib.axes.Axes): Axes to plot on
        theta_values (torch.Tensor): Theta values
            Shape [N]
        theta_dot_values (torch.Tensor): Theta_dot values
            Shape [N]
        state_values (torch.Tensor): State values to plot
            Shape [1, 1, N, N]
    """
    X = theta_space
    Y = theta_dot_space
    Z = barrier_values[0,0].detach().cpu().numpy()
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

def plot_heatmap_x(fig: plt.Figure, 
                    ax: plt.Axes, 
                    x_space: torch.Tensor, 
                    x_dot_space: torch.Tensor, 
                    barrier_values: torch.Tensor):

    X = x_space
    Y = x_dot_space
    Z = barrier_values[:,:,0,0].detach().cpu().numpy()
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str)
    parser.add_argument('--model-paths', type=str, nargs='+')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    _, env = load_model(
        args.model_paths[0],
        make_env,
        args.env_id,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False
    )
    model = load_ensemble_model(
        env,
        args.model_paths,
        device="cpu",
    )
    
    env = gym.make('CartPole-v1')

    # Roll out model
    state = env.reset()
    done = False
    states = []
    actions = []
    values = []
    while not done:
        states.append(state)
        action = model.predict([state]).item()
        value = model.predict_value([state], [action]).item()
        values.append(value)
        actions.append(action)
        state, _, done, _ = env.step(action)
    state_history = np.array(states)
    action_history = np.array(actions)
    value_history = np.array(values)

    # Plot value history
    time = np.arange(len(state_history))
    fig, ax = plt.subplots()
    ax.plot(time, value_history)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Value history')
    fig.show()
    input('Press enter to continue...')

    # Evaluate model on theta and theta_dot
    _, _, theta_space, theta_dot_space = get_default_spaces()
    x_space, x_dot_space, _, _ = get_zero_spaces()
    grid_x, grid_x_dot, grid_theta, grid_theta_dot = torch.meshgrid(
        x_space, x_dot_space, theta_space, theta_dot_space, indexing='xy'
    )
    states = torch.stack([grid_x, grid_x_dot, grid_theta, grid_theta_dot], dim=-1)
    states = states.reshape(-1, 4)
    optimal_actions = model.predict(states)
    state_values = model.predict_value(states, optimal_actions)
    state_values = state_values.reshape(1, 1, 100, 100)
    state_values = torch.Tensor(state_values)
    barrier_values = - state_values + 75

    fig, ax = plt.subplots(ncols=2, figsize=(25, 10))
    plot_heatmap_theta(fig, ax[0], theta_space, theta_dot_space, barrier_values)
    ax[0].scatter(state_history[:,2], state_history[:,3], c='k', s=1)

    # Evaluate model on x and x_dot
    x_space, x_dot_space, _, _ = get_default_spaces()
    _, _, theta_space, theta_dot_space = get_zero_spaces()
    grid_x, grid_x_dot, grid_theta, grid_theta_dot = torch.meshgrid(
        x_space, x_dot_space, theta_space, theta_dot_space, indexing='xy'
    )
    states = torch.stack([grid_x, grid_x_dot, grid_theta, grid_theta_dot], dim=-1)
    states = states.reshape(-1, 4)
    optimal_actions = model.predict(states)
    state_values = model.predict_value(states, optimal_actions)
    state_values = state_values.reshape(100, 100, 1, 1)
    state_values = torch.Tensor(state_values)
    barrier_values = - state_values + 75
    plot_heatmap_x(fig, ax[1], x_space, x_dot_space, barrier_values)
    ax[1].scatter(state_history[:,0], state_history[:,1], c='k', s=1)

    fig.show()
    input("Press Enter to exit...")