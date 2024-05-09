import numpy as np
from functools import partial

class ReplayBuffer(object):

    def __init__(self, state_dim, action_dim, log_dir=None, capacity=100000, dtype=np.float16):
        self.capacity = capacity
        self.len = 0
        self.states = np.zeros((capacity, state_dim), dtype)
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros((capacity, 1))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, 1))
        if log_dir:
            self.log_dir = log_dir
            self.log_file = open(log_dir, 'w')

    def __len__(self):
        return self.len
    
    def _shift(self, index):
        self.states = np.roll(self.states, index, axis=0)
        self.actions = np.roll(self.actions, index, axis=0)
        self.rewards = np.roll(self.rewards, index, axis=0)
        self.next_states = np.roll(self.next_states, index, axis=0)
        self.dones = np.roll(self.dones, index, axis=0)

    def store(self, state, action, reward, next_state, done):
        assert state.shape[0] == action.shape[0] == reward.shape[0] == next_state.shape[0] == done.shape[0] > 0
        size = state.shape[0]
        # if self.log_dir:
        #     for i in range(size):
        #         x, s = state[i]
        #         nx, ns = next_state[i]
        #         content = f"({x:.4f}, {s:.4f}) + {action[i][0]} -> ({nx:.4f}, {ns:.4f}) | {reward[i][0]:.4f}"
        #         print(content, file=self.log_file)

        if self.len + size > self.capacity:
            self.len = self.capacity
        else:
            self.len += size
        
        self._shift(size)
        self.states[:size] = state
        self.actions[:size] = action
        self.rewards[:size] = reward
        self.next_states[:size] = next_state
        self.dones[:size] = done

    def sample(self, batch_size): 
        assert 0 < batch_size <= self.len
        index = np.random.randint(0, self.len, batch_size)
        return self.states[index], self.actions[index], self.rewards[index], self.next_states[index], self.dones[index]