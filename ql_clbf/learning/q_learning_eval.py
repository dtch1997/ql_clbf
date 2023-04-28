import argparse
import numpy as np
import gym
import pickle

from q_learning_train import desc_from_string, desc_to_string, TabularQLearning

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc-name', type=str)
    args = parser.parse_args()
    return args

def generate_plan(T: int):
    plan = np.random.randint(4, size = T)
    return plan

def simulate_with_safety(env, plan, agent):
    state = env.reset()
    state_trajectory = [state]
    action_trajectory = []
    for t in range(len(plan)):
        action = plan[t]
        if agent.get_q_value(state, action) == 0:
            # Resample action from safe set
            safe_actions = agent.get_safe_actions(state)
            action = np.random.choice(safe_actions)

        next_state, reward, done, info = env.step(action)
        state = next_state
        state_trajectory.append(state)
        action_trajectory.append(action)
        if done: break

    was_successful = (reward == 1)
    state_trajectory = np.array(state_trajectory)
    action_trajectory = np.array(action_trajectory)
    return state_trajectory, action_trajectory, was_successful

def simulate(env, plan):
    state = env.reset()
    state_trajectory = [state]
    action_trajectory = []
    for t in range(len(plan)):
        action = plan[t]
        next_state, reward, done, info = env.step(action)
        state = next_state
        state_trajectory.append(state)  
        action_trajectory.append(action)
        if done: break
    was_successful = (t == len(plan)-1)
    state_trajectory = np.array(state_trajectory)
    action_trajectory = np.array(action_trajectory)
    return state_trajectory, action_trajectory, was_successful

if __name__ == "__main__":

    args = get_args()
    desc_name = args.desc_name
    desc = desc_from_string(desc_name)
    env = gym.make("FrozenLake-v1", is_slippery=False, desc=desc)

    state = env.reset()
    env.render(mode='human')
    input('Press enter to continue...')
    desc_filename = f'experiments/{desc_name}/desc.npy'
    state_value_grid = np.load(desc_filename)
    agent_filename = f'experiments/{desc_name}/agent.pkl'
    with open(agent_filename, 'rb') as f:
        agent = pickle.load(f)

    plan = generate_plan(100)
    state_trajectory, action_trajectory, success = simulate(env, plan)
    safe_state_trajectory, safe_action_trajectory, safe_success = simulate_with_safety(env, plan, agent)
    
    print("Plan: ", plan)
    print("State Trajectory: ", len(state_trajectory))
    print("Success: ", success)
    print("Safe Success: ", safe_success)
    print("Safe State Trajectory: ", len(safe_state_trajectory))
    