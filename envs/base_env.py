import numpy as np
import gym
import wandb

class BaseEnv(object):

    def __init__(self, env_name) -> None:
        self.env_name = env_name
        self.env = gym.make(env_name)

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
    
    def sample_action(self) -> int | np.ndarray:
        return self.env.action_space.sample()

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self.total_reward = 0.0
        self.total_raw_reward = 0.0
        self.total_steps = 0
        self.max_reward = None
        self.min_reward = None

        if hasattr(self, 'state_fn'):
            obs = self.state_fn(obs)
        
        return obs, _
    
    def step(self, action):
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


if __name__ == '__main__':

    # env = BaseEnv('CartPole-v0')
    env = BaseEnv('Pendulum-v1')
    # env = BaseEnv("ALE/Asteroids-v5")
    print(env)
    for _ in range(3):
        state, _ = env.reset()
        while True:
            action = env.sample_action()
            next_state, reward, done, terminal, _ = env.step(action)
            print(f'{state=} {action=} {next_state=} {reward=} {done=} {terminal=}')
            if done or terminal:
                break
