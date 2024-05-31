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
        "episode_return": [],
        "episode_step": [],
        "episode_done": [],
        "episode_terminal": [],
    }

    with tqdm.tqdm(range(num_episodes)) as pbar:
        for i_episode in pbar:
            episode_return, episode_step = 0.0, 0

            state, _ = env.reset()
            while True:
                action = agent.take_action(state)
                next_state, reward, done, terminal, _ = env.step(action)
                agent.add_experience(state, action, reward, next_state, done)
                state = next_state

                episode_return += reward
                episode_step += 1

                if done or terminal:
                    metrics["episode_return"].append(episode_return)
                    metrics["episode_step"].append(episode_step)
                    metrics["episode_done"].append(int(done))
                    metrics["episode_terminal"].append(int(terminal))
                    break
            
            agent.update()

            if i_episode > 0 and i_episode % kwargs.get("num_eval_episodes", 100) == 0:
                logdata = {
                    "Ev_Index": i_episode,
                    "Ev_Avg_Return": np.mean(metrics["episode_return"]),
                    "Ev_Avg_Step": np.mean(metrics["episode_step"]),
                    "Ev_Avg_Done": np.mean(metrics["episode_done"]),
                    "Ev_Avg_Terminal": np.mean(metrics["episode_terminal"]),
                }
                
                for key in metrics.keys():
                    metrics[key] = []

                pbar.set_postfix(logdata)
                try: wandb.log(logdata)
                except: pass

