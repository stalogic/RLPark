import gym
import wandb

class MountainCarEnv():

    def __init__(self) -> None:
        self.env = gym.make('MountainCar-v0')

        # infomation
        self.max_xvalue = None
        self.max_speed = None
        self.total_reward = 0
        self.total_raw_reward = 0
        self.total_steps = 0
    
    @property
    def state_dim(self) -> int:
        if hasattr(self, '_state_dim'):
            return self._state_dim
        self._state_dim = self.env.observation_space.shape[0]
        return self._state_dim

    @state_dim.setter
    def state_dim(self, value):
        self._state_dim = value
    
    @property
    def action_dim(self) -> int:
        return self.env.action_space.n

    def reset(self):
        obs, _ = self.env.reset()
        self.max_xvalue = obs[0]
        self.max_speed = abs(obs[1])
        self.total_reward = 0.0
        self.total_raw_reward = 0.0
        self.total_steps = 0
        
        if hasattr(self, 'state_func'):
            obs = self.state_func(obs)

        info = {
            'St_max_xvalue': self.max_xvalue,
            'St_max_speed': self.max_speed,
            'St_total_reward': self.total_reward,
            'St_total_raw_reward': self.total_raw_reward,
            'St_total_steps': self.total_steps
        }
        return obs, info
    
    def sample_action(self):
        return self.env.action_space.sample()
    
    def step(self, action):
        obs, reward, done, terminal, _ = self.env.step(action)

        self.total_steps += 1
        self.max_xvalue = max(self.max_xvalue, obs[0])
        self.max_speed = max(self.max_speed, abs(obs[1]))
        self.total_raw_reward += reward
        if hasattr(self, 'state_func'):
            obs = self.state_func(obs)
        if hasattr(self, 'reward_func'):
            reward = self.reward_func(obs, reward)
        self.total_reward += reward
        info = {
            'St_max_xvalue': self.max_xvalue,
            'St_max_speed': self.max_speed,
            'St_total_reward': self.total_reward,
            'St_total_raw_reward': self.total_raw_reward,
            'St_total_steps': self.total_steps
        }

        if done or terminal:
            logdata = {
                'Gm_max_xvalue': self.max_xvalue,
                'Gm_max_speed': self.max_speed,
                'Gm_total_reward': self.total_reward,
                'Gm_total_raw_reward': self.total_raw_reward,
                'Gm_total_steps': self.total_steps
            }
            try: wandb.log(logdata)
            except: pass

        return obs, reward, done, terminal, info
    

def mountain_car_raw():
    env = MountainCarEnv()
    return env

def mountain_car_reward_redefined():
    def _reward(self, obs, reward):
        x, v = obs
        if x >= 0.5:
            return 100
        return reward + 10 * (abs(x+0.5) + 10 * abs(v))
    
    env = MountainCarEnv()
    setattr(MountainCarEnv, "reward_func", _reward)
    return env

def mountain_car_state_reward_redefined():
    import numpy as np
    def _state(self, obs):
        position, velocity = obs[:2]
        steps_remaining = 200 - self.total_steps
        remaining = (steps_remaining - 100) / 200.
        return np.array([position, velocity, remaining])
    
    def _reward(self, obs, reward):
        x, v = obs[:2]
        s = 200 - self.total_steps
        if x >= 0.5:
            return 100 + 10 * s
        return reward + 10 * (abs(x+0.5) + 10 * abs(v))
    
    env = MountainCarEnv()
    env.state_dim += 1
    setattr(env, "steps_remaining", 201)
    setattr(MountainCarEnv, "state_func", _state)
    setattr(MountainCarEnv, "reward_func", _reward)
    return env


if __name__ == '__main__':
    env = mountain_car_state_reward_redefined()
    obs, info = env.reset()
    print(obs, info)

    while True:
        action = env.sample_action()
        obs, reward, done, terminal, info = env.step(action)
        print(obs, reward, done, terminal, info)
        if done or terminal:
            break
