from d3rlpy.datasets import get_cartpole # CartPole-v0 dataset
from sklearn.model_selection import train_test_split
from d3rlpy.algos import DoubleDQN
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment

if __name__ == "__main__":
    
    
    dataset, env = get_cartpole()
    
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    # if you don't use GPU, set use_gpu=False instead.
    dqn = DoubleDQN(use_gpu=True)

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
            n_epochs=20,
            scorers={
                'td_error': td_error_scorer,
                'value_scale': average_value_estimation_scorer,
                'environment': evaluate_scorer
            })
    
    # save full parameters
    dqn.save_model('double_dqn.pth')