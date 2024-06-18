import os
import pathlib
import typer
import subprocess


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
    return " ".join(lines)

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
        elif isinstance(value, str):
            lines.append(f"{key}='{value}',")
        elif isinstance(value, dict):
            lines.append(f"{key}={generate_dict(**value)},")
        else:
            lines.append(f"{key}={value},")
    lines.append(f")")
    return " ".join(lines)
    

def generate_code(algo_name, env_name, **kwargs):
    num_episodes = kwargs.get('num_episodes', 500)

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

    wandb_init = generate_fn("wandb.init", project=f"{env_name}", config={
        "state_dim": CodeValue("env.state_dim_or_shape"),
        "action_dim": CodeValue("env.action_dim_or_shape"),
        "algorithm": algo_name,
        **kwargs
    })
    
    agent_init = generate_fn(f"agent={algo_name}", CodeValue("env.state_dim_or_shape"), CodeValue("env.action_dim_or_shape"), 
                            **kwargs
                             )

    train_code = generate_fn("train_and_evaluate", env=CodeValue("env"), agent=CodeValue("agent"), **kwargs)

    return "\n\n".join([import_code, wandb_init, agent_init, train_code])

def generate_python_script(algo_name: str, env_name: str, **kwargs):
    code = generate_code(algo_name, env_name, **kwargs)
    path = pathlib.Path(f"./experiments/{env_name}".lower())
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"{algo_name}_{env_name}.py".lower()
    path.write_text(code)
    black_result = subprocess.run(["black", str(path)], capture_output=True)
    if black_result.returncode != 0:
        print(black_result.stderr.decode("utf-8"))
    return str(path)

if __name__ == '__main__':
    typer.run(generate_python_script)