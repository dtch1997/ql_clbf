import argparse
import gym 
import torch 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from typing import Callable, List

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
    theta_space = torch.linspace(-np.pi / 4, np.pi / 4, steps=100)
    theta_dot_space = torch.linspace(-2, 2, steps=100)
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

if __name__ == '__main__':

    from ql_clbf.offline.double_dqn_eval import load_model
    model = load_model('d3rlpy_logs/DoubleDQN_20230501120606/model_50140.pt')

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
    barrier_values = - state_values

    fig, ax = plt.subplots(ncols=2, figsize=(25, 10))
    plot_heatmap_theta(fig, ax[0], theta_space, theta_dot_space, barrier_values)

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
    barrier_values = - state_values
    plot_heatmap_x(fig, ax[1], x_space, x_dot_space, barrier_values)

    fig.show()
    input("Press Enter to exit...")