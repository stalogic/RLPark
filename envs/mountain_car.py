import gym

class MountainCarEnv():

    def __init__(self) -> None:
        self.env = gym.make('MountainCar-v0')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # infomation
        self.rightmost_position = None
        self.max_speed = None
        self.total_reward = 0
        self.total_raw_reward = 0
        self.total_steps = 0

    def reset(self):
        obs, _ = self.env.reset()
        self.rightmost_position = obs[0]
        self.max_speed = abs(obs[1])
        
        if hasattr(self, 'state_func'):
            obs = self.state_func(obs)

        info = {
            'rightmost_position': self.rightmost_position,
            'max_speed': self.max_speed,
            'total_reward': self.total_reward,
            'total_raw_reward': self.total_raw_reward,
            'total_steps': self.total_steps
        }
        return obs, info
    
    def sample_action(self):
        return self.env.action_space.sample()
    
    def step(self, action):
        obs, reward, done, terminal, _ = self.env.step(action)

        self.total_steps += 1
        self.rightmost_position = max(self.rightmost_position, obs[0])
        self.max_speed = max(self.max_speed, abs(obs[1]))
        self.total_raw_reward += reward
        if hasattr(self, 'state_func'):
            obs = self.state_func(obs)
        if hasattr(self, 'reward_func'):
            reward = self.reward_func(obs, reward)
        self.total_reward += reward
        info = {
            'rightmost_position': self.rightmost_position,
            'max_speed': self.max_speed,
            'total_reward': self.total_reward,
            'total_raw_reward': self.total_raw_reward,
            'total_steps': self.total_steps
        }

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
        position, velocity = obs
        self.steps_remaining -= 1
        return np.array([position, velocity, self.steps_remaining / 200])
    
    def _reward(self, obs, reward):
        x, v, s = obs
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
    env = mountain_car_v0_state_reward_redefined()
    obs, info = env.reset()
    print(obs, info)

    while True:
        action = env.sample_action()
        obs, reward, done, terminal, info = env.step(action)
        print(obs, reward, done, terminal, info)
        if done or terminal:
            break
