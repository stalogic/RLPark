import os
import sys
import wandb

print(f"running in {os.getcwd()}")
sys.path.append(os.getcwd())
from algorithm import ActorCritic
from algorithm.util import train_and_evaluate
from envs import poker_game_v0 as make_env

env = make_env()
state_dim = env.state_dim
action_dim = env.action_dim
hidden_dim = 128
batch_size = 128
num_episodes = 10000

wandb.init(project="poker_game_v0",
            config={
               "state_dim": state_dim,
               "action_dim": action_dim,
               "hidden_dim": hidden_dim,
               "batch_size": batch_size,
               "num_episodes": num_episodes,
               "algorithm": "ActorCritic"
           })

agent = ActorCritic(state_dim, action_dim, hidden_dim, batch_size)

train_and_evaluate(env, agent, num_episodes)
