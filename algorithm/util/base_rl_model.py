import torch
import numpy as np
from contextlib import contextmanager

from . import ReplayBuffer


class OnPolicyRLModel(object):
    def __init__(self, state_dim_or_shape, action_dim_or_shape=1, capacity=10000, **kwargs) -> None:
        self.count = 0

    @contextmanager
    def eval_mode(self):
        changed_model :dict[str, torch.nn.Module] = dict()
        for key, field in self.__dict__.items():
            if isinstance(field, torch.nn.Module) and field.training:
                field.eval()
                changed_model[key] = field
        yield
        for key, field in changed_model.items():
            field.train()
    
    def update_learning_rate(self) -> None:
        for name, field in self.__dict__.items():
            if isinstance(field, torch.optim.lr_scheduler.LRScheduler):
                field.step()

    def _soft_update_target_net(self, net, target_net, tau):
        for param_name in net.state_dict().keys():
            target_param = target_net.state_dict()[param_name]
            net_param = net.state_dict()[param_name]
            target_param.data.copy_(target_param.data * (1 - tau) + net_param.data * tau)

class OffPolicyRLModel(OnPolicyRLModel):

    def __init__(self, state_dim_or_shape, action_dim_or_shape=1, capacity=10000, **kwargs) -> None:
        self.replay_buffer = ReplayBuffer(state_dim_or_shape, action_dim_or_shape, capacity)

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


        
