import pathlib
import typer

def generate_code(algo_name, env_name, **kwargs):
    hidden_dim = kwargs.get('hidden_dim', 128)
    batch_size = kwargs.get('batch_size', 128)
    num_episodes = kwargs.get('num_episodes', 10000)

    code_template = f"""\
import os
import sys
import wandb

print(f"running in {{os.getcwd()}}")
sys.path.append(os.getcwd())
from algorithm import {algo_name}
from algorithm.util import train_and_evaluate
from envs import {env_name} as make_env

env = make_env()
state_dim = env.state_dim
action_dim = env.action_dim
hidden_dim = {hidden_dim}
batch_size = {batch_size}
num_episodes = {num_episodes}

wandb.init(project="{env_name}",
            config={{
               "state_dim": state_dim,
               "action_dim": action_dim,
               "hidden_dim": hidden_dim,
               "batch_size": batch_size,
               "num_episodes": num_episodes,
               "algorithm": "{algo_name}"
            }})

agent = {algo_name}(state_dim, action_dim, hidden_dim, batch_size)

train_and_evaluate(env, agent, num_episodes)
"""
    return code_template

def generate_python_script(algo_name: str, env_name: str):
    code = generate_code(algo_name, env_name)
    path = pathlib.Path(f"./experiments/{env_name}".lower())
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"{algo_name}_{env_name}.py".lower()
    path.write_text(code)
    return str(path)

if __name__ == '__main__':
    typer.run(generate_python_script)