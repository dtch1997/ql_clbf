import argparse
import os

from distutils.util import strtobool

from d3rlpy.datasets import get_cartpole # CartPole-v0 dataset
from sklearn.model_selection import train_test_split
from d3rlpy.algos import DoubleDQN
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment

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
    parser.add_argument("--wandb-project-name", type=str, default="QL_CLBF",
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
    parser.add_argument("--total-epochs", type=int, default=500,
        help="total epochs of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=1024,
        help="the batch size of samples")
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
    args = parser.parse_args()
    # fmt: on
    return args

def make_dataset():
    replay_dataset, env = get_cartpole('replay')
    random_dataset, _ = get_cartpole('random')
    replay_dataset.extend(random_dataset)
    dataset = replay_dataset
    return dataset, env

if __name__ == "__main__":
    
    args = parse_args()
    if args.track:
        import wandb
        wandb.init(
            name=args.exp_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=args,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True
        )

    dataset, env = make_dataset()
    
    
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    # if you don't use GPU, set use_gpu=False instead.
    dqn = DoubleDQN(
        learning_rate=args.learning_rate,
        use_gpu=True,
        batch_size=args.batch_size,
        n_frames=1,
        target_update_interval=args.target_network_frequency,
    )

    # initialize neural networks with the given observation shape and action size.
    # this is not necessary when you directly call fit or fit_online method.
    dqn.build_with_dataset(dataset)


    # calculate metrics with test dataset
    td_error = td_error_scorer(dqn, test_episodes)

    # set environment in scorer function
    evaluate_scorer = evaluate_on_environment(env)

    # evaluate algorithm on the environment
    rewards = evaluate_scorer(dqn)

    dqn.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=args.total_epochs,
            scorers={
                'td_error': td_error_scorer,
                'value_scale': average_value_estimation_scorer,
                'environment': evaluate_scorer
            }, 
            tensorboard_dir='./tensorboard_logs',
        )
    
    # save full parameters
    dqn.save_model('double_dqn.pth')