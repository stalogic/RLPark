import numpy as np
import gym
import wandb
import cv2
import collections

class BaseEnv(object):

    def __init__(self, env_name, **kwargs) -> None:
        self.env_name = env_name
        self.env = gym.make(env_name, **kwargs)

        self.total_reward = 0.0
        self.total_raw_reward = 0.0
        self.total_steps = 0
        self.max_reward = None
        self.min_reward = None
    
    def __str__(self) -> str:
        base_info = f'{self.__class__.__name__}{{env_name={self.env_name}, state_dim_or_shape={self.state_dim_or_shape}, action_dim_or_shape={self.action_dim_or_shape}}}'
        if hasattr(self, 'state_fn'):
            base_info += f'\nstate_fn:{self.state_fn.__doc__}'
        if hasattr(self, 'reward_fn'):
            base_info += f'\nreward_fn:{self.reward_fn.__doc__}'
        return base_info

    @property
    def state_dim_or_shape(self) -> int|tuple[int]:
        if len(self.env.observation_space.shape) == 1:
            return self.env.observation_space.shape[0]
        else:
            return self.env.observation_space.shape
    
    @property
    def action_dim_or_shape(self) -> int|tuple[int]:
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            return self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.box.Box):
            return self.env.action_space.shape
        raise NotImplementedError('action_space type not supported')
    
    @property
    def action_bound(self) -> int|tuple[int]:
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            return None
        elif isinstance(self.env.action_space, gym.spaces.box.Box):
            return self.env.action_space.high
        raise NotImplementedError('action_space type not supported')
    
    def sample_action(self) -> int | np.ndarray:
        return self.env.action_space.sample()

    def reset(self) -> np.ndarray:
        obs, info = self.env.reset()
        self.total_reward = 0.0
        self.total_raw_reward = 0.0
        self.total_steps = 0
        self.max_reward = None
        self.min_reward = None

        if hasattr(self, 'state_fn'):
            obs = self.state_fn(obs)
        
        return obs, info
    
    def step(self, action):
        if isinstance(action, (tuple, list, np.ndarray)) and isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            if np.isclose(np.sum(np.abs(action)), 1.0) :
                action = np.argmax(action)
            else:
                raise ValueError('action must be a single integer')
            
        obs, raw_reward, done, terminal, _ = self.env.step(action)
        if hasattr(self, 'state_fn'):
            obs = self.state_fn(obs, raw_reward, done, terminal)
        if hasattr(self, 'reward_fn'):
            reward = self.reward_fn(obs, raw_reward, done, terminal)
        else:
            reward = raw_reward

        self.total_steps += 1
        self.total_raw_reward += raw_reward
        self.total_reward += reward
        self.max_reward = max(self.max_reward, reward) if self.max_reward else reward
        self.min_reward = min(self.min_reward, reward) if self.min_reward else reward

        step_log = {
            'St_step': self.total_steps,
            'St_raw_reward': raw_reward,
            'St_reward': reward,
        }

        try: wandb.log(step_log)
        except: pass

        if done or terminal:
            game_log = {
                'Gm_total_steps': self.total_steps,
                'Gm_total_raw_reward': self.total_raw_reward,
                'Gm_total_reward': self.total_reward,
                'Gm_max_reward': self.max_reward,
                'Gm_min_reward': self.min_reward,
                'Gm_mean_reward': self.total_reward / self.total_steps,
            }
            try: wandb.log(game_log)
            except: pass

        return obs, reward, done, terminal, _


class Base2DEnv(BaseEnv):
    def __init__(self, env_name:str, width:int=84, height:int=84, grayscale:bool=True, seqlen:int=10, **kwargs) -> None:
        super().__init__(env_name, **kwargs)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.seqlen = seqlen
        self.stack = collections.deque(maxlen=seqlen)
        for _ in range(seqlen):
            channels = 1 if self.grayscale else 3
            empty_state = np.zeros((channels, self.width, self.height))
            self.stack.append(empty_state)

    @property
    def state_dim_or_shape(self) -> tuple:
        channels = 1 if self.grayscale else 3
        return (channels * self.seqlen, self.width, self.height)
    
    def state_fn(self, obs, raw_reward=None, done=None, terminal=None) -> np.ndarray:
        """
        根据配置处理观测值，转换为模型所需的 state 格式。
        
        参数:
        obs: 输入的原始观测值，通常是图像数据。
        raw_reward: (可选) 原始奖励值，本函数中未使用。
        done: (可选) 表示 episode 是否结束的布尔值，本函数中未使用。
        terminal: (可选) 表示是否到达终端状态的布尔值，本函数中未使用。
        
        返回:
        处理后的观测值，格式为 numpy 数组。
        """
        # 如果配置为灰度图像，则将彩色图像转换为灰度图像，并调整大小
        if self.grayscale:
            # 转换为灰度图像
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            # 调整图像大小，使用 INTER_AREA 插值方法以保持图像质量
            obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
            # 添加颜色通道维度，因为后续处理需要多通道图像
            obs = np.expand_dims(obs, axis=0)
        else:
            # 调整图像大小，使用 INTER_AREA 插值方法以保持图像质量
            obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
            # 调整图像通道顺序，从 HWC 调整为 CHW，以符合模型输入要求
            obs = np.transpose(obs, (2, 0, 1))
        self.stack.append(obs)
        state = np.concatenate(self.stack, axis=0)
        return state
    


if __name__ == '__main__':

    # env = BaseEnv('CartPole-v0')
    # env = BaseEnv('Pendulum-v1')
    # env = BaseEnv("ALE/Asteroids-v5")
    env = Base2DEnv("ALE/Breakout-v5", grayscale=False)
    print(env)
    for _ in range(3):
        state, _ = env.reset()
        while True:
            action = env.sample_action()
            next_state, reward, done, terminal, _ = env.step(action)
            print(f'{state=} {action=} {next_state=} {reward=} {done=} {terminal=}')
            if done or terminal:
                break
