import os
import envs
import algorithm
from generate import generate_python_script

for env_name in envs.DISCRETE_ENV_LIST:
    for algo_name in algorithm.DISCRETE_ALGO_LIST:
        path = generate_python_script(algo_name, env_name)
        cmd = f"python {path}"
        print(cmd)
        # os.system(cmd)


for env_name in envs.CONTINUOUS_ENV_LIST:
    for algo_name in algorithm.CONTINUOUS_ALGO_LIST:
        path = generate_python_script(algo_name, env_name)
        cmd = f"python {path}"
        print(cmd)
        os.system(cmd)