import gym
import numpy as np

from gym import spaces
from collections import defaultdict

class TabularQLearning:

    def __init__(self, 
                 env: gym.Env, 
                 gamma=0.9, 
                 alpha=0.1):
        assert isinstance(env.observation_space, spaces.Discrete)
        assert isinstance(env.action_space, spaces.Discrete)

        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.gamma = gamma
        self.alpha = alpha

    def get_q_value(self, state, action = None):
        if action is None:
            return self.q_table[state]
        else: 
            return self.q_table[state, action]
        
    def get_safe_actions(self, state):
        return np.nonzero(self.q_table[state] > 0)[0]
        
    def update_q_value(self, state, action, reward, next_state):
        q_value_curr = self.get_q_value(state, action)
        q_value_target = reward + self.gamma * self.get_q_value(next_state).max()
        self.q_table[state, action] = q_value_curr + self.alpha * (q_value_target - q_value_curr)

    def get_action(self, state, epsilon=0.1):
        if np.random.uniform() < epsilon:
            return np.random.choice(self.q_table.shape[1])
        else:
            return self.get_q_value(state).argmax()

def get_state_value_grid(q_table, env):
    state_values_grid = np.zeros(env.unwrapped.desc.shape, dtype=float)
    nrow, ncol = env.unwrapped.desc.shape
    for i in range(env.observation_space.n):        
        row, col = i // ncol, i % ncol
        state_values_grid[row, col] = q_table[i].max()
    return state_values_grid

def get_default_desc():
    pass

def desc_to_string(desc):
    return "|".join(desc)

def desc_from_string(desc_str):
    return desc_str.split("|")

def get_desc_A():
    return ["FFSFF", "FHHFF", "FFFFF", "FFFFF", "FFGFF"]

def get_desc_B():
    return ["FFSFF", "FFFFF", "FFFFF", "FFHHF", "FFGFF"]

def get_desc_AB():
    return ["FFSFF", "FHHFF", "FFFFF", "FFHHF", "FFGFF"]

def generate_fixed_experiment():
    return {
        "A": get_desc_A(),
        "B": get_desc_B(),
        "AB": get_desc_AB(),
    }

def generate_default_experiment():
    return {
        "default": ["SFFF", "FHFH", "FFFH", "HFFG"]
    }

def generate_base_description():
    desc = []
    for i in range(5):
        desc.append("FFFFF")
    return desc

def generate_random_positions(n: int):
    """ Sample n random positions on the grid without replacement """
    positions = []
    while len(positions) < n:
        pos = np.random.randint(5, size = 2)
        pos = (pos[0], pos[1])
        if pos not in positions:
            positions.append(pos)
    return positions

def put_object(desc, positions, obj):
    for pos in positions:
        row, col = pos
        desc[row] = desc[row][:col] + obj + desc[row][col+1:]
    return desc

def generate_random_experiment():
    desc = generate_base_description()
    positions = generate_random_positions(1 + 1 + 4)
    put_object(desc, positions[0:1], "S")
    put_object(desc, positions[1:2], "G")
    put_object(desc, positions[2:6], "H")

    desc_str = desc_to_string(desc)    
    return {
        desc_str: desc
    }

def generate_random_composition_experiment():
    desc = generate_base_description()
    positions = generate_random_positions(1 + 1 + 4 + 4)
    put_object(desc, positions[0:1], "S")
    put_object(desc, positions[1:2], "G")
    
    desc_A = desc.copy()
    put_object(desc_A, positions[2:6], "H")
    
    desc_B = desc.copy()
    put_object(desc_B, positions[6:10], "H")

    desc_AB = desc.copy()
    put_object(desc_AB, positions[2:10], "H")
    
    return {
        'A': desc_A, 
        'B': desc_B, 
        'AB': desc_AB
    }

def linear_schedule(initial_value, final_value, n_steps):
    slope = (final_value - initial_value) / n_steps
    def func(step):
        return initial_value + slope * step
    return func

if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default = 'default')
    args = parser.parse_args()

    if args.exp == 'default':
        descs = generate_default_experiment()
    elif args.exp == 'fixed':
        descs = generate_fixed_experiment()
    elif args.exp == 'random':
        descs = generate_random_experiment()
    elif args.exp == 'random_composition':
        descs = generate_random_composition_experiment()
    else:
        raise ValueError("Invalid experiment type")

    for name, desc in descs.items():
        env = gym.make("FrozenLake-v1", is_slippery=False, desc=desc)
        state = env.reset()
        env.render(mode='human')

        input('Press enter to continue...')
        agent = TabularQLearning(env)

        epsilon_sched = linear_schedule(1.0, 0.1, 100000)
        
        prev_state_grid = get_state_value_grid(agent.q_table, env)
        for i in range(100000):
            epsilon = epsilon_sched(i)
            state = env.reset()
            done = False
            while not done:
                action = agent.get_action(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                agent.update_q_value(state, action, reward, next_state)
                state = next_state
            
            if i % 1000 == 0:
                print("Episode: ", i)
                state_value_grid = get_state_value_grid(agent.q_table, env)
                print(state_value_grid)
                
                if np.all(np.isclose(prev_state_grid, state_value_grid)) and i > 40000:
                    print("Converged")
                    break
            
                prev_state_grid = state_value_grid

        import pathlib
        save_dir = pathlib.Path(f'experiments/{name}')
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save state grid
        np.save(f'{save_dir}/desc.npy', state_value_grid)
        # Save agent
        import pickle
        with open(f'{save_dir}/agent.pkl', 'wb') as f:
            pickle.dump(agent, f) 