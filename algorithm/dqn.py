import torch
import copy
import wandb
import numpy as np
from pathlib import Path
from .util import OffPolicyRLModel, QValueNetwork


class DQN(OffPolicyRLModel):
    
    def __init__(self, state_dim_or_shape, action_dim_or_shape, hidden_dims=(32,), conv_layers=((32, 3),), batch_size=128, epsilon=0.1, lr=1e-3, gamma=0.99, device='cpu', capacity=10000, **kwargs) -> None:
        super().__init__(state_dim_or_shape, action_dim_or_shape, capacity)
        if not isinstance(state_dim_or_shape, (int, tuple, list)):
            raise TypeError('state_dim_or_shape must be int, tuple or list')
        if not isinstance(action_dim_or_shape, (int, tuple, list)):
            raise TypeError('action_dim_or_shape must be int, tuple or list')
        
        self.state_shape = (state_dim_or_shape,) if isinstance(state_dim_or_shape, int) else tuple(state_dim_or_shape)
        self.action_dim = action_dim_or_shape[0] if not isinstance(action_dim_or_shape, int) else action_dim_or_shape
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.kwargs = kwargs

        self.q_net = QValueNetwork(self.state_shape, self.action_dim, hidden_dims=hidden_dims, conv_layers=conv_layers, **kwargs)
        self.target_q_net = copy.deepcopy(self.q_net)

        self.q_net.to(device)
        self.target_q_net.to(device)
        self.target_q_net.eval()

        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.q_net_scheduler = torch.optim.lr_scheduler.StepLR(self.q_net_optimizer, step_size=100, gamma=0.955)

    def take_action(self, state) -> int:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad(), self.eval_mode():
                state = torch.tensor(state, dtype=torch.float).to(self.device)
                action = self.q_net(state).argmax().item()
        return action
    
    def take_action_with_mask(self, state, mask: list|np.ndarray) -> int:
        if not isinstance(mask, (list, np.ndarray)) or len(mask) != self.action_dim:
            raise ValueError("mask must be a list or numpy array with length of action_dim")
        if isinstance(mask, list):
            mask = np.array(mask)
        mask = mask.astype(int)

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_dim, p=mask/mask.sum())
        else:
            with torch.no_grad(), self.eval_mode():
                state = torch.tensor(state, dtype=torch.float).to(self.device)
                q_value = self.q_net(state).numpy()
                q_value[:, mask == 0] = -np.inf
                action = q_value.argmax()

        return action
    
    def predict_action(self, state) -> int:
        with torch.no_grad(), self.eval_mode():
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    def predict_action_with_mask(self, state, mask: list|np.ndarray) -> int:
        if not isinstance(mask, (list, np.ndarray)) or len(mask) != self.action_dim:
            raise ValueError("mask must be a list or numpy array with length of action_dim")
        if isinstance(mask, list):
            mask = np.array(mask)
        mask = mask.astype(int)
        
        with torch.no_grad(), self.eval_mode():
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            q_value = self.q_net(state).numpy()
            q_value[:, mask == 0] = -np.inf
            action = q_value.argmax()
        return action

    def update(self) -> None:
        if len(self.replay_buffer) < self.kwargs.get('batch_size', 1000):
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad(), self.eval_mode():
            next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = torch.nn.functional.mse_loss(q_values, q_target)
        self.q_net_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.kwargs.get('max_grad_norm', 0.5))
        self.q_net_optimizer.step()
        self.q_net_scheduler.step()

        try: wandb.log({'Tr_loss': loss.item(), 'Tr_learning_rate': self.q_net_scheduler.get_last_lr()[0]})
        except: pass

        self.count += 1
        if tau:=self.kwargs.get('tau'):
            for param_name in self.q_net.state_dict().keys():
                target_param = self.target_q_net.state_dict()[param_name]
                param = self.q_net.state_dict()[param_name]
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        elif self.count % self.kwargs.get('target_update_frequency', 100) == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())           

        if self.kwargs.get('save_path') and self.count % self.kwargs.get('save_frequency', 1000) == 0:
            self.save()

    def save(self) -> None:
        path = Path(self.kwargs.get('save_path')) / f"{self.count}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path / 'q_net.pth')

    def load(self, version:str):
        pass

