import numpy as np
import tqdm
import os
import wandb
print(f"running in {os.getcwd()}")
os.sys.path.append(os.getcwd())
from algorithm import ActorCritic, DQN
from envs import mountain_car_v1 as mountain_car_env

env = mountain_car_env()
state_dim = env.state_dim
action_dim = env.action_dim
hidden_dim = 128
batch_size = 128
num_episodes = 10000

agent = ActorCritic(state_dim, action_dim, hidden_dim, batch_size)

metrics = {
    "reward": [],
    "raw_reward": [],
    "done": [],
    "terminal": [],
    "max_pos": [],
    "max_speed": [],
    "max_steps": []
}

with tqdm.tqdm(range(num_episodes)) as pbar:
    for episode in pbar:
        state, _ = env.reset()
        
        while True:
            action = agent.take_action(state)
            next_state, reward, done, terminal, info = env.step(action)
            agent.add_experience(state, action, reward, next_state, done)
            state = next_state

            if done or terminal:
                metrics["done"].append(1 if done else 0)
                metrics["terminal"].append(1 if terminal else 0)
                
                total_steps = info.get("total_steps")
                metrics["max_steps"].append(total_steps)
                metrics["reward"].append(info.get("total_reward") / total_steps)
                metrics["raw_reward"].append(info.get("total_raw_reward") / total_steps)
                metrics["max_pos"].append(info.get("rightmost_position"))
                metrics["max_speed"].append(info.get("max_speed"))
                break
        
        agent.update()
        
        if episode > 0 and episode % 100 == 0:
            pbar.set_postfix({
                "Episode": episode,
                "Reward": np.mean(metrics["reward"]),
                "Raw Reward": np.mean(metrics["raw_reward"]),
                "Max Steps": np.mean(metrics["max_steps"]),
                "Done": np.mean(metrics["done"]),
                "Terminal": np.mean(metrics["terminal"]),
                "Max Pos": np.max(metrics["max_pos"]),
                "Max Speed": np.max(metrics["max_speed"]),
            })

            for key in metrics.keys():
                metrics[key] = []
        
