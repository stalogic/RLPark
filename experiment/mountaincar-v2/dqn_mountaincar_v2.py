import numpy as np
import os
import sys
import wandb

print(f"running in {os.getcwd()}")
sys.path.append(os.getcwd())
from algorithm import DQN
from algorithm.util import train_and_evaluate
from envs import mountain_car_v2 as make_env

env = make_env()
state_dim = env.state_dim
action_dim = env.action_dim
hidden_dim = 128
batch_size = 128
num_episodes = 10000

wandb.init(project="mountain_car_v2",
            config={
               "state_dim": state_dim,
               "action_dim": action_dim,
               "hidden_dim": hidden_dim,
               "batch_size": batch_size,
               "num_episodes": num_episodes,
               "algorithm": "DQN"
           })

agent = DQN(state_dim, action_dim, hidden_dim, batch_size)

train_and_evaluate(env, agent, num_episodes)
