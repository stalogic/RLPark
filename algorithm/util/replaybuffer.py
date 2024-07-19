import numpy as np
import collections
import random

class ReplayBuffer(object):
    def __init__(self, capacity=100000, *args, **kwargs):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
         return len(self.buffer)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)