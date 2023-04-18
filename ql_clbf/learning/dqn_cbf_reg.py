# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import ql_clbf.envs.gym # register envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument('-R', '--reward-upper-bound', type=float, default=1.0, 
                        help="Upper bound on reward for environment")
    parser.add_argument('--cbf-reg-coef', type=float, default=0.0,
        help="the coefficient of the CBF regularization term")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument('--eval-frequency', type=int, default=100000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument("--supervised-frequency", type=int, default=-1,
        help="the frequency of supervised learning step")
    parser.add_argument("--supervised-batch-size", type=int, default=256,
        help="the frequency of supervised learning step")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    total_safety_violations = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                episode_length = info["episode"]["l"]
                if episode_length < 500:
                    total_safety_violations += 1
                    writer.add_scalar("charts/safety_violations", total_safety_violations, global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            # If enabled, run CBF supervised learning step
            if (args.supervised_frequency > 0) and global_step % args.supervised_frequency == 0:
                unsafe_states = envs.call('sample_unsafe_states', args.supervised_batch_size)
                unsafe_states = np.stack(unsafe_states, axis=0)
                unsafe_states = unsafe_states.reshape(-1, np.array(envs.single_observation_space.shape).prod())
                unsafe_states = torch.Tensor(unsafe_states).to(device)

                qsa_val = q_network(unsafe_states)
                qs_val, _ = qsa_val.max(dim=1)
                h_val = - qs_val + args.reward_upper_bound / (1- args.gamma)
                supervised_loss = torch.maximum(-h_val, torch.zeros_like(h_val))
                supervised_loss = supervised_loss.mean()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/cbf_supervised_loss", supervised_loss, global_step)

                # optimize the model
                optimizer.zero_grad()
                supervised_loss.backward()
                optimizer.step()

                del unsafe_states, qsa_val, qs_val, supervised_loss

            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val_all = q_network(data.observations)
                old_val_act = old_val_all.gather(1, data.actions).squeeze()
                old_val = old_val_act
                loss = F.mse_loss(td_target, old_val)

                # TRY NOT TO MODIFY: CBF regularization term
                # Hack: Assume reward = 0 when unsafe and reward > 0 when not unsafe
                is_unsafe = (data.rewards < -1 + 1e-6).float()
                q_val, _ = old_val_all.max(dim=-1)
                h_val = - q_val + args.reward_upper_bound / (1- args.gamma)
                cbf_unsafe_loss = torch.maximum(-h_val, torch.zeros_like(h_val)) * is_unsafe
                cbf_loss = cbf_unsafe_loss.mean()

                # Compute total loss
                reg_coef = args.cbf_reg_coef
                loss += reg_coef * cbf_loss

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("losses/cbf_reg_loss", cbf_loss, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

            # Evaluate the agent periodically
            if global_step % args.eval_frequency == 0:
                model_path = f"runs/{run_name}/{args.exp_name}_step{global_step}.pth"
                base_path = f"runs/{run_name}"
                torch.save(q_network.state_dict(), model_path)
                print(f"model saved to {model_path}")

                if args.track:
                    wandb.save(model_path, base_path=base_path)
                    print(f"model saved to wandb cloud")

                from ql_clbf.learning.dqn_eval import evaluate, plot_h_values_trajectory
                eval_info = evaluate(
                    model_path,
                    make_env,
                    args.env_id,
                    eval_episodes=args.eval_episodes,
                    run_name=f"{run_name}-eval",
                    Model=QNetwork,
                    device=device,
                    epsilon=0.00, # Disable exploration in evaluation
                    capture_video=args.capture_video,
                )

                model = eval_info['model']
                envs = eval_info['envs']
                episodic_returns = eval_info['episodic_returns']
                episodic_q_values = eval_info['episodic_q_values']
                episodic_state_trajectories = eval_info['episodic_state_trajectories']
                episodic_violations = eval_info['episodic_violations']
                episodic_safety_success = eval_info['episodic_safety_success']

                # Compute empirical safety accuracy
                unsafe_states = envs.call('sample_unsafe_states', 10000)
                unsafe_states = np.stack(unsafe_states, axis=0)
                unsafe_states = unsafe_states.reshape(-1, np.array(envs.single_observation_space.shape).prod())
                unsafe_states = torch.Tensor(unsafe_states).to(device)
                state_values = model(unsafe_states).max(dim=-1)[0]
                h_values = - state_values + args.reward_upper_bound / (1- args.gamma)
                is_unsafe_empirical = torch.ones_like(h_values) # True unsafe labels are all ones
                is_unsafe_pred_empirical = (h_values > 0).float()
                is_unsafe_accuracy_empirical = (is_unsafe_pred_empirical == is_unsafe_empirical).float().mean()
                
                # Compute rollout safety accuracy 
                states = episodic_state_trajectories # (episodes, steps, 1, state_dim)
                states = torch.Tensor(states).to(device)
                state_values = model(states).max(dim=-1)[0]
                h_values = - state_values + args.reward_upper_bound / (1- args.gamma)
                is_unsafe_rollout = torch.Tensor(episodic_violations).to(device)
                is_unsafe_pred_rollout = (h_values > 0).float()
                is_unsafe_accuracy_rollout = (is_unsafe_pred_rollout == is_unsafe_rollout).float().mean()

                writer.add_scalar("eval/episodic_return", episodic_returns.mean(), global_step)
                writer.add_scalar("eval/safety_success_rate", episodic_safety_success.mean(), global_step)
                writer.add_scalar("eval/safety_accuracy_empirical", is_unsafe_accuracy_empirical, global_step)
                writer.add_scalar("eval/safety_accuracy_rollout", is_unsafe_accuracy_rollout, global_step)

                fig, ax = plot_h_values_trajectory(episodic_q_values, args.reward_upper_bound, args.gamma)
                writer.add_figure("eval/h_values_trajectory", fig, global_step)

    envs.close()
    writer.close()