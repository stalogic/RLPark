import torch
import numpy as np
import gym
from algorithm.util.net import PolicyNetwork
from tqdm import tqdm

env = gym.make('MountainCar-v0', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128

policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
policy.load_state_dict(torch.load('ckpt/policy_net_I800.pth'))
policy.eval()

done_steps = []
num_failed = 0


with tqdm(range(100), leave=True) as pbar:
    for i in pbar:
        state, _ = env.reset()
        steps = 0
        while True:
            env.render()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # action = policy(state).argmax().item()
            action = torch.distributions.Categorical(policy(state)).sample().item()
            next_state, reward, done, terminal, _ = env.step(action)
            state = next_state
            steps += 1
            if done:
                done_steps.append(steps)
                break
            if terminal:
                num_failed += 1
                break

        pbar.update(1)
        pbar.set_description(f'Evaluation: {i}')
        if done_steps:
            pbar.set_postfix_str(f"success: {len(done_steps)}, failure: {num_failed}, min steps: {min(done_steps)}, max steps: {max(done_steps)}, avg steps: {sum(done_steps) / len(done_steps):.2f}")



"""
`action = policy(state).argmax().item()`

success: 78, failure: 22, min steps: 100, max steps: 188, avg steps: 153.54


`action = torch.distributions.Categorical(policy(state)).sample().item()`
success: 57, failure: 43, min steps: 160, max steps: 199, avg steps: 173.39
"""