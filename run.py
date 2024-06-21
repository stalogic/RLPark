import os
import envs
import algorithm
from generate import generate_python_script

for env_name in envs.DISCRETE_ENV_LIST:
    for algo_name in algorithm.DISCRETE_ALGO_LIST:
        path = generate_python_script(algo_name, env_name, 
                                      input_norm=True,
                                      hidden_dims=(64, 32),
                                      num_eval_episodes=5,
                                      num_episodes=2000,
                                      )
        cmd = f"python {path}"
        print(cmd)
        # os.system(cmd)


for env_name in envs.CONTINUOUS_ENV_LIST:
    for algo_name in algorithm.CONTINUOUS_ALGO_LIST:
        path = generate_python_script(algo_name, env_name)
        cmd = f"python {path}"
        print(cmd)
        # os.system(cmd)

for env_name in envs.DISCRETE_2D_ENV_LIST:
    for algo_name in algorithm.DISCRETE_ALGO_LIST:
        path = generate_python_script(
            algo_name,
            env_name,
            device="cuda",
            num_episodes=10000,
            hidden_dims=(64, 32),
            conv_layers=((32, 3), (16, 3)), # input shape (-1, 1, 84, 84), 经过CNN网络并Flatten后维度为1024
            num_eval_episodes=5, # 每次评估的episode数量为10
        )
        cmd = f"python {path}"
        print(cmd)
        # os.system(cmd)
