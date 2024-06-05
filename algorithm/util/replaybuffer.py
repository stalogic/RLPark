import numpy as np

class ReplayBuffer(object):

    def __init__(self, state_shape:int|tuple|list, action_shape:int|tuple|list=1, capacity=10000):
        """
        Replay buffer for RL
        :param state_shape: 目前仅支持一维状态
        :param action_shape: 如果是discrete环境，action_shape=1，如果是continuous环境，action_shape=action_space.n
        :param capacity: 存储容量
        """
        self.capacity = capacity
        self.len = 0
        self.index = 0

        if not isinstance(state_shape, (int, tuple, list)):
            raise ValueError('state_shape must be int or tuple or list')
        if isinstance(state_shape, int):
            self.state_shape = (state_shape, )
        elif isinstance(state_shape, list):
            self.state_shape = tuple(state_shape)
        self.states = np.zeros((capacity, ) + self.state_shape, dtype=np.float32)
        self.next_states = np.zeros((capacity, ) + self.state_shape, dtype=np.float32)

        if not isinstance(action_shape, (int, tuple, list)):
            raise ValueError('action_shape must be int or tuple or list')
        if isinstance(action_shape, int):
            self.action_shape = (action_shape, )
        elif isinstance(action_shape, list):
            self.action_shape = tuple(action_shape)
        self.actions = np.zeros((capacity,) + self.action_shape, dtype=np.float32)

        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

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