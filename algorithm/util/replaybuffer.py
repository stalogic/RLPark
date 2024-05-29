import numpy as np

class ReplayBuffer(object):

    def __init__(self, state_dim, capacity=10000):
        self.capacity = capacity
        self.len = 0
        self.index = 0
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, 1))
        self.rewards = np.zeros((capacity, 1))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, 1))

    def __len__(self):
        return self.len

    def store(self, state, action, reward, next_state, done):
        assert state.shape[0] == action.shape[0] == reward.shape[0] == next_state.shape[0] == done.shape[0] > 0
        size = state.shape[0]

        if self.len + size > self.capacity:
            self.len = self.capacity
        else:
            self.len += size
        
        slices = [(self.index + i) % self.capacity for i in range(size)]
        self.index = (self.index + size) % self.capacity

        self.states[slices] = state
        self.actions[slices] = action
        self.rewards[slices] = reward
        self.next_states[slices] = next_state
        self.dones[slices] = done

    def sample(self, batch_size): 
        assert 0 < batch_size <= self.len
        index = np.random.randint(0, self.len, batch_size)
        return self.states[index], self.actions[index], self.rewards[index], self.next_states[index], self.dones[index]