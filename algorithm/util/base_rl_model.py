import numpy as np
from . import ReplayBuffer

class DiscreteRLModel(object):

    def __init__(self, state_dim, capcity=10000) -> None:
        self.replay_buffer = ReplayBuffer(state_dim, capcity)
        self.count = 0
    
    def add_experience(self, state, action, reward, next_state, done) -> None:
        if isinstance(state, (list, float, int)):
            state = np.array(state)
        state = state.reshape(-1, self.state_dim)
        if isinstance(next_state, (list, float, int)):
            next_state = np.array(next_state)
        next_state = next_state.reshape(-1, self.state_dim)
        if isinstance(action, (list, float, int)):
            action = np.array(action)
        action = action.reshape(-1, 1)
        if isinstance(reward, (list, float, int)):
            reward = np.array(reward)
        reward = reward.reshape(-1, 1)
        if isinstance(done, (list, float, int)):
            done = np.array(done)
        done = done.reshape(-1, 1)

        self.replay_buffer.store(state, action, reward, next_state, done)


class ContinuousRLModel(object):
    pass
    