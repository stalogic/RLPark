import os
import envs
import algorithm
from generate import generate_python_script

for env_name in envs.ENV_LIST:
    for algo_name in algorithm.ALGO_LIST:
        path = generate_python_script(algo_name, env_name)
        # print(path)
        cmd = f"python {path}"
        os.system(cmd)