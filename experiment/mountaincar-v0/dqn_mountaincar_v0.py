import gym
import numpy as np
import tqdm
import os
print(os.getcwd())
os.sys.path.append(os.getcwd())
from algorithm.dqn import DQN

def adjust(next_state):
    position, velocity = next_state
    if position > 0.5:
        return 100
    button_x = -0.5
    distance = abs(position - button_x)
    return (distance + 10*abs(velocity)) * 10 - 1

env = gym.make('MountainCar-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128
batch_size = 128
num_episodes = 1000


agent = DQN(state_dim, action_dim, hidden_dim, batch_size)

metrics = {
    "reward": [],
    "raw_reward": [],
    "done": [],
    "terminal": [],
    "v_loss": [],
    "p_loss": [],
    "max_pos": [],
    "max_speed": [],
}

with tqdm.tqdm(range(num_episodes)) as pbar:
    for episode in pbar:
        state, _ = env.reset()
        total_reward = 0
        total_raw_reward = 0
        max_pos = -1
        max_speed = -1
        while True:
            action = agent.take_action(state)
            next_state, reward, done, terminal, _ = env.step(action)
            total_raw_reward += reward
            reward = adjust(next_state)
            total_reward += reward
            max_pos = max(max_pos, next_state[0]) if max_pos else next_state[0]
            max_speed = max(max_speed, next_state[1]) if max_speed else next_state[1]
            agent.add_experience(state, action, reward, next_state, done)
            state = next_state

            if done or terminal:
                metrics["done"].append(1 if done else 0)
                metrics["terminal"].append(1 if terminal else 0)
                metrics["reward"].append(total_reward)
                metrics["raw_reward"].append(total_raw_reward)
                metrics["max_pos"].append(max_pos)
                metrics["max_speed"].append(max_speed)
                break
        
        agent.update()
        
        if episode > 0 and episode % 100 == 0:
            pbar.set_postfix({
                "Episode": episode,
                "Reward": np.mean(metrics["reward"]),
                "Raw Reward": np.mean(metrics["raw_reward"]),
                "Done": np.mean(metrics["done"]),
                "Terminal": np.mean(metrics["terminal"]),
                "Max Pos": np.max(metrics["max_pos"]),
                "Max Speed": np.max(metrics["max_speed"]),
            })

            for key in metrics.keys():
                metrics[key] = []
        
