import random
from typing import Callable

import argparse
import gym
import numpy as np
import torch

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt 
import ql_clbf.envs.gym # noqa: F401

def pad_to(x: np.ndarray, n: int) -> np.ndarray:
    """ Pad x to length n with np.NaN """
    n_shape = len(x.shape)
    pad_widths = [(0, n - len(x))]
    for i in range(1, n_shape):
        pad_widths.append((0, 0))
    return np.pad(x, pad_widths, mode="constant", constant_values=np.nan)

def plot_heatmap_theta(fig, ax, model, env, reward_upper_bound, gamma, beta):
    """ Plot Q-values for CartPole-v1 dynamics """
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
    Z = - Z + reward_upper_bound / (1- gamma) + beta
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

def plot_heatmap_x(fig, ax, model, env, reward_upper_bound, gamma, beta):
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
    Z = - Z + reward_upper_bound / (1- gamma) + beta
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('-R', '--reward-upper-bound', type=float, default=1.0, 
                        help="Upper bound on reward for environment")
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--n-episodes', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--max-episode-length', type=int, default=500)
    args = parser.parse_args()
    return args

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
    max_episode_length: int = 500,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs = envs.reset(seed=0)
    infos = [{'violation': False}]

    # Set up episodic statistics buffers
    episodic_returns = []
    episodic_q_values = []
    episodic_state_trajectories = []
    episodic_violations = []
    q_values_hist = []
    x_hist = []
    violations_hist = []

    while len(episodic_returns) < eval_episodes:

        q_values = model(torch.Tensor(obs).to(device))
        q_values_hist.append(q_values.detach().cpu().numpy())
        x_hist.append(obs.copy())
        violations_hist.append([info['violation'] for info in infos])

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, _, _, infos = envs.step(actions)
        for info in infos:
            if "episode" in info.keys():
                # Episode is done
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                q_values_hist = np.stack(q_values_hist, axis=0)
                q_values_hist = pad_to(q_values_hist, max_episode_length)
                x_hist = np.stack(x_hist, axis=0)
                x_hist = pad_to(x_hist, max_episode_length)
                violations_hist = np.stack(violations_hist, axis=0)
                violations_hist = pad_to(violations_hist, max_episode_length)
                
                # Record episode statistics
                episodic_returns += [info["episode"]["r"]]
                episodic_q_values += [q_values_hist]
                episodic_state_trajectories += [x_hist]
                episodic_violations += [violations_hist]
                
                # Handle reset
                q_values_hist = []
                x_hist = []
                violations_hist = []
                next_obs = envs.reset(seed=len(episodic_returns))
        obs = next_obs
        
    episodic_returns = np.stack(episodic_returns, axis=0)
    episodic_q_values = np.stack(episodic_q_values, axis=0)
    episodic_state_trajectories = np.stack(episodic_state_trajectories, axis=0)
    episodic_violations = np.stack(episodic_violations, axis=0)
    episodic_safety_success = episodic_violations.sum(axis=1) == 0

    return {
        'model': model,
        'envs': envs,
        'episodic_returns': episodic_returns, 
        'episodic_q_values': episodic_q_values, 
        'episodic_state_trajectories': episodic_state_trajectories,
        'episodic_violations': episodic_violations,        
        'episodic_safety_success': episodic_safety_success
    }

def plot_h_values_trajectory(episodic_q_values, reward_upper_bound, gamma, beta = 0):
    fig, ax = plt.subplots()
    state_values = episodic_q_values.max(axis=-1)
    h_values = - state_values + reward_upper_bound / (1- gamma) + beta

    h_mean = np.nanmean(h_values, axis=0)[:, 0]
    h_stdev = np.nanstd(h_values, axis=0)[:, 0]
    t = np.arange(h_mean.shape[0])

    # Plot Q-values over time
    ax.plot(t, h_mean)
    ax.fill_between(t, h_mean - h_stdev, h_mean + h_stdev, alpha=0.5)
    ax.set_title("h(x) trajectories")
    ax.set_ylim(np.nanmin(h_mean - h_stdev), np.nanmax(h_mean + h_stdev))
    return fig, ax

def plot_heatmap(model, envs, reward_upper_bound, gamma, beta = 0):
    fig, ax = plt.subplots(ncols=2, figsize=(25, 10))
    plot_heatmap_theta(fig, ax[0], model, envs, reward_upper_bound, gamma, beta)
    plot_heatmap_x(fig, ax[1], model, envs, reward_upper_bound, gamma, beta)
    return fig, ax

if __name__ == "__main__":
    from ql_clbf.learning.dqn import make_env, QNetwork
    
    args = get_args()
    eval_info = evaluate(
        args.model_path,
        make_env,
        args.env_id,
        eval_episodes=args.n_episodes,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False,
        epsilon=args.epsilon
    )

    model = eval_info['model']
    envs = eval_info['envs']
    episode_state_trajectories = eval_info['episodic_state_trajectories']

    # Plot heatmap with state trajectories
    fig, ax = plot_heatmap(model, envs, args.reward_upper_bound, args.gamma, args.beta)
    ax[0].scatter(episode_state_trajectories[:, :, 0, 2], episode_state_trajectories[:, :, 0, 3], s=4, c='k')
    ax[1].scatter(episode_state_trajectories[:, :, 0, 0], episode_state_trajectories[:, :, 0, 1], s=4, c='k')
    fig.show()
    input("Press Enter to exit...")