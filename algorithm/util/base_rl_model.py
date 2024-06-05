import numpy as np
from . import ReplayBuffer

class OffPolicyRLModel(object):

    def __init__(self, state_dim_or_shape, action_dim_or_shape=1, capacity=10000) -> None:
        self.replay_buffer = ReplayBuffer(state_dim_or_shape, action_dim_or_shape, capacity)
        self.count = 0
    
    def add_experience(self, state:list|np.ndarray, action:int|float|list|np.ndarray, reward:int|float|list|np.ndarray, next_state:list|np.ndarray, done:bool|int|float) -> None:
        if isinstance(state, list):
            state = np.array(state)
        state = state.reshape(-1, *self.replay_buffer.state_shape)
        if isinstance(next_state, list):
            next_state = np.array(next_state)
        next_state = next_state.reshape(-1, *self.replay_buffer.state_shape)
        if isinstance(action, (list, float, int)):
            action = np.array(action)
        action = action.reshape(-1, *self.replay_buffer.action_shape)
        if isinstance(reward, (list, float, int)):
            reward = np.array(reward)
        reward = reward.reshape(-1, 1)
        if isinstance(done, (list, int)):
            done = np.array(done)
        elif isinstance(done, bool):
            done = np.array([done]).astype(np.float32)
        done = done.reshape(-1, 1)

        self.replay_buffer.store(state, action, reward, next_state, done)


        
class OnPolicyRLModel(object):
    def __init__(self, state_dim_or_shape, action_dim_or_shape=1, capacity=10000) -> None:
        self.count = 0
    