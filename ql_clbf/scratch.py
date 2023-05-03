import ql_clbf.envs 
import gymnasium as gym 

if __name__ == "__main__":
    env = gym.make('SafetyPointMaze_Open-v3', render_mode='human')
    env.reset()

    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(action, obs, reward, term, trunc, info)
        if term or trunc:
            break


