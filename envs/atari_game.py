import math
from .base_env import Base2DEnv


def pong_v0(full_action_space=False, **kwargs) -> Base2DEnv:
    return Base2DEnv("ALE/Pong-v5", full_action_space=full_action_space, **kwargs)


def pong_v1(full_action_space=False, **kwargs) -> Base2DEnv:
    class PONG(Base2DEnv):
        def __init__(
            self,
            env_name: str,
            width: int = 84,
            height: int = 84,
            grayscale: bool = True,
            seqlen: int = 50,
            **kwargs,
        ) -> None:
            super().__init__(env_name, width, height, grayscale, seqlen, **kwargs)

        def reward_fn(self, obs=None, reward=None, done=None, terminal=None):
            if reward > 0:
                reward *= 100
            elif reward < 0:
                reward *= 10
            return reward + 1

    return PONG("ALE/Pong-v5", full_action_space=full_action_space, **kwargs)


def breakout_v0(full_action_space=False, **kwargs) -> Base2DEnv:
    return Base2DEnv("ALE/Breakout-v5", full_action_space=full_action_space, **kwargs)


if __name__ == "__main__":
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
            obs, reward, done, terminal, info = env.step(action)
            print(f"state: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"terminal: {terminal}")
            print(f"info: {info}")
            if done or terminal:
                break
