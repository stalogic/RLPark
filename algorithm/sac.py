import copy
import torch
import numpy as np
import wandb
from pathlib import Path


from .util import OffPolicyRLModel, ContinuousQValueNetwork, PolicyNetwork, QValueNetwork


class SAC(OffPolicyRLModel):

    def __init__(self, state_dim_or_shape, action_dim_or_shape, hidden_dim=32, batch_size=128, lr=1e-3, gamma=0.99, tau=0.005, target_entropy=1.0, device='cpu', **kwargs) -> None:
        super().__init__(state_dim_or_shape, action_dim_or_shape, **kwargs)
        if not isinstance(state_dim_or_shape, (int, tuple, list)):
            raise TypeError(f"state_dim_or_shape must be int, tuple or list")
        if not isinstance(action_dim_or_shape, (int, tuple, list)):
            raise TypeError(f"action_dim_or_shape must be int, tuple or list")

        self.state_shape = (state_dim_or_shape,) if isinstance(state_dim_or_shape, int) else tuple(state_dim_or_shape)
        self.action_dim = action_dim_or_shape[0] if not isinstance(action_dim_or_shape, int) else action_dim_or_shape
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.device = device
        self.kwargs = kwargs

        self.policy_net = PolicyNetwork(self.state_shape, self.action_dim, hidden_dim).to(device)
        self.q1_net = QValueNetwork(self.state_shape, self.action_dim, hidden_dim).to(device)
        self.q2_net = QValueNetwork(self.state_shape, self.action_dim, hidden_dim).to(device)
        self.target_q1_net = copy.deepcopy(self.q1_net).to(device)
        self.target_q2_net = copy.deepcopy(self.q2_net).to(device)
        self.target_q1_net.eval()
        self.target_q2_net.eval()

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.kwargs.get('actor_lr', lr))
        self.q1_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=self.kwargs.get('critic_lr', lr))
        self.q2_optimizer = torch.optim.Adam(self.q2_net.parameters(), lr=self.kwargs.get('critic_lr', lr))
        self.policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=100, gamma=0.955)
        self.q_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.q1_optimizer, step_size=100, gamma=0.955)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.kwargs.get('alpha_lr', 1e-2))

    def take_action(self, state) -> int:
        with torch.no_grad():
            self.policy_net.eval()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action_prob = self.policy_net(state)
            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample().item()
            self.policy_net.train()
        return action
    
    def take_action_with_mask(self, state, mask: list|np.ndarray) -> int:
        if not isinstance(mask, (list, np.ndarray)) or len(mask) != self.action_dim:
            raise TypeError(f"mask must be list or np.ndarray with length {self.action_dim}")
        
        if isinstance(mask, list):
            mask = np.array(mask)
        mask = mask.astype(np.float32)

        with torch.no_grad():
            self.policy_net.eval()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action_prob = self.policy_net(state)
            action_dist = torch.distributions.Categorical(action_prob * mask)
            action = action_dist.sample().item()
            self.policy_net.train()
        return action
    
    def predict_action(self, state) -> int:
        with torch.no_grad():
            self.policy_net.eval()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action_prob = self.policy_net(state)
            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample().item()
            self.policy_net.train()
        return action
    
    def predict_action_with_mask(self, state, mask: list|np.ndarray) -> int:
        if not isinstance(mask, (list, np.ndarray)) or len(mask) != self.action_dim:
            raise TypeError(f"mask must be list or np.ndarray with length {self.action_dim}")
        
        if isinstance(mask, list):
            mask = np.array(mask)
        mask = mask.astype(np.float32)

        with torch.no_grad():
            self.policy_net.eval()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action_prob = self.policy_net(state)
            action_prob = action_prob * torch.tensor(mask, dtype=torch.float32).to(self.device)
            action = action_prob.argmax(dim=1).item()
            self.policy_net.train()
        return action
    
    def _calc_target(self, rewards, next_states, dones):
        with torch.no_grad():
            self.policy_net.eval()
            next_action_prob = self.policy_net(next_states)
            next_log_action_prob = torch.log(next_action_prob + 1e-8)
            entropy = -torch.sum(next_action_prob * next_log_action_prob, dim=1, keepdim=True)
            q1_value = self.target_q1_net(next_states)
            q2_value = self.target_q2_net(next_states)
            min_q_value = torch.sum(next_action_prob * torch.min(q1_value, q2_value), dim=1, keepdim=True)
            next_value = min_q_value + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target
    
    def _soft_update(self, net, target_net):
        for param_name in net.state_dict().keys():
            target_param = target_net.state_dict()[param_name]
            net_param = net.state_dict()[param_name]
            target_param.data.copy_(target_param.data * (1 - self.tau) + net_param.data * self.tau)


    def update(self) -> None:
        if len(self.replay_buffer) < self.batch_size * 10:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).view(-1, *self.state_shape).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).view(-1, *self.state_shape).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1).to(self.device)

        td_target = self._calc_target(rewards, next_states, dones)
        q1_value = self.q1_net(states).gather(1, actions)
        q1_loss = torch.nn.functional.mse_loss(q1_value, td_target)
        q2_value = self.q2_net(states).gather(1, actions)
        q2_loss = torch.nn.functional.mse_loss(q2_value, td_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1_net.parameters(), self.kwargs.get('value_max_grad_norm', 0.5))
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2_net.parameters(), self.kwargs.get('value_max_grad_norm', 0.5))
        self.q2_optimizer.step()

        probs = self.policy_net(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.q1_net(states)
        q2_value = self.q2_net(states)
        min_q_value = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        policy_loss = -torch.mean(min_q_value + self.log_alpha.exp() * entropy)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.kwargs.get('policy_max_grad_norm', 0.5))
        self.policy_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha.grad /= self.log_alpha.exp()
        torch.nn.utils.clip_grad_norm_(self.log_alpha, self.kwargs.get('alpha_max_grad_norm', 0.3))
        self.alpha_optimizer.step()

        self._soft_update(self.q1_net, self.target_q1_net)
        self._soft_update(self.q2_net, self.target_q2_net)
        self.q_lr_scheduler.step()
        self.policy_lr_scheduler.step()

        log_data = {
            'Tr_q1_loss': q1_loss.item(),
            'Tr_q2_loss': q2_loss.item(),
            'Tr_policy_loss': policy_loss.item(),
            'Tr_alpha_loss': alpha_loss.item(),
            'Tr_alpha': self.log_alpha.exp().item(),
            'Tr_entropy': entropy.mean().item(),
            'Tr_entropy_gap': (entropy - self.target_entropy).mean().item(),
            'Tr_q_learning_rate': self.q_lr_scheduler.get_last_lr()[0],
            'Tr_policy_learning_rate': self.policy_lr_scheduler.get_last_lr()[0],
        }

        try: wandb.log(log_data)
        except: pass

