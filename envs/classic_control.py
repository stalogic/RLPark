import numpy as np
from .base_env import BaseEnv


def acrobot_v1(**kwargs) -> BaseEnv:
    return BaseEnv("Acrobot-v1", **kwargs)


def cart_pole_v0(**kwargs) -> BaseEnv:
    return BaseEnv("CartPole-v0", **kwargs)


def cart_pole_v1(**kwargs) -> BaseEnv:
    return BaseEnv("CartPole-v1", **kwargs)


def pendulum_v1(**kwargs) -> BaseEnv:
    return BaseEnv("Pendulum-v1", **kwargs)


def pendulum_v2(**kwargs) -> BaseEnv:
    class PendulumEnv(BaseEnv):
        def reward_fn(self, obs=None, reward=None, done=None, terminal=None):
            """将奖励修改为1 + reward / 8"""
            return (reward + 8) / 8

    return PendulumEnv("Pendulum-v1", **kwargs)


def mountain_car_v0(**kwargs) -> BaseEnv:
    return BaseEnv("MountainCar-v0", **kwargs)


def mountain_car_v1(**kwargs) -> BaseEnv:
    class MountainCarEnvV1(BaseEnv):
        def reward_fn(self, obs=None, reward=None, done=None, terminal=None):
            """完成奖励设置为10, 其他奖励设置为`abs(x+0.5) + 10 * abs(v)`, x为位置, v为速度"""
            if done:
                return 10
            else:
                return abs(obs[0] + 0.5) + 10 * abs(obs[1])

    return MountainCarEnvV1("MountainCar-v0", **kwargs)


def mountain_car_v2(**kwargs) -> BaseEnv:
    class MountainCarEnvV2(BaseEnv):
        def state_fn(self, obs=None, reward=None, done=None, terminal=None):
            """状态由(position, velocity)修改为(position, velocity, running_steps)"""
            return np.array([obs[0], obs[1], self.total_steps])

        @property
        def state_dim_or_shape(self) -> int | tuple[int]:
            return 3

        def reward_fn(self, obs=None, reward=None, done=None, terminal=None):
            """完成奖励设置为10, 其他奖励设置为`abs(x+0.5) + 10 * abs(v)`, x为位置, v为速度"""
            if done:
                return 10
            else:
                return abs(obs[0] + 0.5) + 10 * abs(obs[1])

    return MountainCarEnvV2("MountainCar-v0", **kwargs)


def mountain_car_v3(**kwargs) -> BaseEnv:
    class MountainCarEnvV3(BaseEnv):
        def state_fn(self, obs=None, reward=None, done=None, terminal=None):
            """状态由(position, velocity)修改为(position, velocity, running_steps)"""
            return np.array([obs[0], obs[1], self.total_steps])

        @property
        def state_dim_or_shape(self) -> int | tuple[int]:
            return 3

        def reward_fn(self, obs=None, reward=None, done=None, terminal=None):
            """完成奖励设置为`10 + rs * log(1 + rs)`, 其他奖励设置为`abs(x + 0.5) + 10 * abs(v) - 0.1 * log(1 + ts)`, x为位置, v为速度, rs为剩余步数, ts为当前步数"""
            if done:
                s = 0 if self.total_steps >= 200 else 200 - self.total_steps
                return 10 + s * np.log1p(s)
            else:
                return (
                    abs(obs[0] + 0.5)
                    + 10 * abs(obs[1])
                    - 0.1 * np.log1p(self.total_steps)
                )

    return MountainCarEnvV3("MountainCar-v0", **kwargs)


def mountain_car_continuous_v0(**kwargs) -> BaseEnv:
    return BaseEnv("MountainCarContinuous-v0", **kwargs)


def mountain_car_continuous_v1(**kwargs) -> BaseEnv:
    class MountainCarContinuousEnvV1(BaseEnv):
        def reward_fn(self, obs=None, reward=None, done=None, terminal=None):
            """done=False时奖励设置为`reward + abs(x+0.5) + 10 * abs(v)`, x为位置, v为速度"""
            if done:
                return reward + 20 * (1000 - self.total_steps)
            else:
                return reward + abs(obs[0] + 0.5) + 100 * abs(obs[1])

    return MountainCarContinuousEnvV1("MountainCarContinuous-v0", **kwargs)


def mountain_car_continuous_v2(**kwargs) -> BaseEnv:
    class MountainCarContinuousEnvV2(BaseEnv):
        def state_fn(self, obs=None, reward=None, done=None, terminal=None):
            """状态由(position, velocity)修改为(position, velocity, running_steps)"""
            return np.array([obs[0], obs[1], self.total_steps])

        @property
        def state_dim_or_shape(self) -> int | tuple[int]:
            return 3

        def reward_fn(self, obs=None, reward=None, done=None, terminal=None):
            """done=False时奖励设置为`reward + abs(x+0.5) + 10 * abs(v)`, x为位置, v为速度"""
            if done:
                return reward
            else:
                return reward + abs(obs[0] + 0.5) + 10 * abs(obs[1])

    return MountainCarContinuousEnvV2("MountainCarContinuous-v0", **kwargs)


def mountain_car_continuous_v3(**kwargs) -> BaseEnv:
    class MountainCarContinuousEnvV3(BaseEnv):
        def state_fn(self, obs=None, reward=None, done=None, terminal=None):
            """状态由(position, velocity)修改为(position, velocity, running_steps)"""
            return np.array([obs[0], obs[1], self.total_steps])

        @property
        def state_dim_or_shape(self) -> int | tuple[int]:
            return 3

        def reward_fn(self, obs=None, reward=None, done=None, terminal=None):
            """done=True时奖励设置为`reward + rs * log(1 + rs)`, done=False时奖励设置为`reward + abs(x + 0.5) + 10 * abs(v) - 0.1 * log(1 + ts)`, x为位置, v为速度, rs为剩余步数, ts为当前步数"""
            if done:
                s = 0 if self.total_steps >= 999 else 999 - self.total_steps
                return reward + s * np.log1p(s)
            else:
                height = abs(obs[0] + 0.5)
                velocity = 200 * abs(obs[1])
                penatly = -0.01 * np.log1p(self.total_steps)
                reward_ = reward + height + velocity + penatly
                print(
                    f"steps: {self.total_steps}, raw_reward: {reward:.3f}, height: {height:.3f}, velocity: {velocity:.3f}, penatly: {penatly:.3f}, reward: {reward_:.3f}"
                )
                input(">>>")
                return reward_

    return MountainCarContinuousEnvV3("MountainCarContinuous-v0", **kwargs)


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
