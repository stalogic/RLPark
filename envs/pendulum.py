import gym
import wandb

class PendulumEnv():

    def __init__(self) -> None:
        self.env = gym.make('Pendulum-v1')

        # information
        self.total_reward = 0.0
        self.total_raw_reward = 0.0
        self.total_steps = 0
    
    @property
    def state_dim(self) -> int:
        return self.env.observation_space.shape[0]

    @property
    def action_dim(self) -> int:
        return self.env.action_space.n

    def reset(self):
        obs, _ = self.env.reset()
        self.total_reward = 0.0
        self.total_raw_reward = 0.0
        self.total_steps = 0
        
        if hasattr(self, 'state_func'):
            obs = self.state_func(obs)

        info = {
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
        self.total_raw_reward += reward
        if hasattr(self, 'state_func'):
            obs = self.state_func(obs)
        if hasattr(self, 'reward_func'):
            reward = self.reward_func(obs, reward)
        self.total_reward += reward
        info = {
            'St_total_reward': self.total_reward,
            'St_total_raw_reward': self.total_raw_reward,
            'St_total_steps': self.total_steps
        }

        if done or terminal:
            logdata = {
                'Gm_total_reward': self.total_reward,
                'Gm_total_raw_reward': self.total_raw_reward,
                'Gm_total_steps': self.total_steps
            }
            try: wandb.log(logdata)
            except: pass

        return obs, reward, done, terminal, info

def pendulum_v1():
    return PendulumEnv()

if __name__ == '__main__':
    env = pendulum_v1()
    obs, info = env.reset()
    print(obs, info)

    while True:
        action = env.sample_action()
        obs, reward, done, terminal, info = env.step(action)
        print(obs, reward, done, terminal, info)
        if done or terminal:
            break
