from .base_env import BaseEnv

def pong_v0() -> BaseEnv:
    return BaseEnv("ALE/Pong-v5", full_action_space=False)


if __name__ == '__main__':
    import inspect
    import sys

    current_module = sys.modules[__name__]
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        print(f"{name:*^150}")
        env = func()
        print(env)
        print(f"state_dim_or_shape: {env.state_dim_or_shape}")
        print(f"action_dim_or_shape: {env.action_dim_or_shape}")
        obs = env.reset()
        print(f"state: {obs}")
        print(f"action: {env.sample_action()}")
        print(env.step(env.sample_action()))
        while True:
            action = env.sample_action()
            obs, reward, done, terminal,info = env.step(action)
            print(f"state: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"terminal: {terminal}")
            print(f"info: {info}")
            if done or terminal:
                break
