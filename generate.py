import os
import pathlib
import typer


class CodeValue(object):

    def __init__(self, value) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return self.value
    
def generate_dict(**kwargs):
    lines = ["{"]
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, str):
            lines.append(f"'{key}': '{value}',")
        else:
            lines.append(f"'{key}': {value},")
    lines.append("}")
    return "\n".join(lines)

def generate_fn(fn_name:str, *args, **kwargs):
    lines = []
    lines.append(f"{fn_name}(")
    for arg in args:
        if isinstance(arg, str):
            lines.append(f"'{arg}', ")
        else:
            lines.append(f"{arg},")
    for key, value in kwargs.items():
        if value is None:
            continue
        elif isinstance(value, dict):
            lines.append(f"{key}={generate_dict(**value)},")
        else:
            lines.append(f"{key}={value},")
    lines.append(f")")
    return "\n".join(lines)
    

def generate_code(algo_name, env_name, **kwargs):
    hidden_dims = kwargs.get('hidden_dims', None)
    conv_layers = kwargs.get('conv_layers', None)
    batch_size = kwargs.get('batch_size', 128)
    num_episodes = kwargs.get('num_episodes', 500)
    action_bound = kwargs.get('action_bound', None)
    device = kwargs.get('device', None)

    import_code = f"""\
import os
import sys
import wandb

print(f"running in {{os.getcwd()}}")
sys.path.append(os.getcwd())
from algorithm import {algo_name}
from algorithm.util import train_and_evaluate
from envs import {env_name} as make_env

env = make_env()
print(env)
"""

    wandb_init = generate_fn("wandb.init", project=f"'{env_name}'", config={
        "state_dim": CodeValue("env.state_dim_or_shape"),
        "action_dim": CodeValue("env.action_dim_or_shape"),
        "hidden_dims": hidden_dims,
        "conv_layers": conv_layers,
        "batch_size": batch_size,
        "num_episodes": num_episodes,
        "action_bound": action_bound,
        "device": device,
        "algorithm": algo_name
    })
    
    agent_init = generate_fn(f"agent={algo_name}", CodeValue("env.state_dim_or_shape"), CodeValue("env.action_dim_or_shape"), 
                             batch_size=batch_size,
                             hidden_dims=hidden_dims,
                             conv_layers=conv_layers,
                             action_bound=action_bound,
                             device=device)

    train_code = generate_fn("train_and_evaluate", env="env", agent="agent", num_episodes=num_episodes,)

    return "\n\n".join([import_code, wandb_init, agent_init, train_code])

def generate_python_script(algo_name: str, env_name: str):
    code = generate_code(algo_name, env_name)
    path = pathlib.Path(f"./experiments/{env_name}".lower())
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"{algo_name}_{env_name}.py".lower()
    path.write_text(code)
    os.system(f"black {path}")
    return str(path)

if __name__ == '__main__':
    typer.run(generate_python_script)