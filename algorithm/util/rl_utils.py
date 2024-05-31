import torch
import tqdm
import numpy as np
import wandb

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def train_and_evaluate(env, agent, num_episodes=1000, **kwargs):

    metrics = {
        "reward": [],
        "steps": [],
        "done": [],
        "terminal": [],
    }

    with tqdm.tqdm(range(num_episodes)) as pbar:
        for i_episode in pbar:
            total_reward, total_steps = 0.0, 0

            state, _ = env.reset()
            while True:
                action = agent.take_action(state)
                next_state, reward, done, terminal, info = env.step(action)
                agent.add_experience(state, action, reward, next_state, done)
                state = next_state

                total_reward += reward
                total_steps += 1

                if done or terminal:
                    metrics["reward"].append(total_reward)
                    metrics["steps"].append(total_steps)
                    metrics["done"].append(int(done))
                    metrics["terminal"].append(int(terminal))
                    break
            
            agent.update()
            
            if i_episode > 0 and i_episode % kwargs.get("num_eval_episodes", 100) == 0:
                logdata = {
                    "Episode": i_episode,
                    "Reward": np.mean(metrics["reward"]),
                    "Steps": np.mean(metrics["steps"]),
                    "Done": np.mean(metrics["done"]),
                    "Terminal": np.mean(metrics["terminal"])
                }
                
                for key in metrics.keys():
                    metrics[key] = []

                pbar.set_postfix(logdata)
                try: wandb.log(logdata)
                except: pass

