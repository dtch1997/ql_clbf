import gym
import argparse

from d3rlpy.algos import DoubleDQN
from d3rlpy.datasets import get_cartpole # CartPole-v0 dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-paths', type=str, nargs="+")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)

    dataset, env = get_cartpole()
    dqn = DoubleDQN(use_gpu=True)

    # initialize neural networks with the given observation shape and action size.
    # this is not necessary when you directly call fit or fit_online method.
    dqn.build_with_dataset(dataset)
    dqn.load_model(args.model_paths[0])

    # evaluate algorithm on the environment
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = dqn.predict([obs])[0]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

