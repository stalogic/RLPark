import torch
import copy
import numpy as np
import wandb
from pathlib import Path
from .util import DiscreteRLModel, PolicyNetwork, ValueNetwork

class ActorCritic(DiscreteRLModel):
    
    def __init__(self, state_dim, action_dim, hidden_dim=32, batch_size=128, lr=1e-3, gamma=0.99, device='cpu', **kwargs) -> None:
        super().__init__(state_dim, **kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.kwargs = kwargs

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.target_policy_net = copy.deepcopy(self.policy_net)
        self.target_value_net = copy.deepcopy(self.value_net)

        self.policy_net.to(device)
        self.value_net.to(device)
        self.target_policy_net.to(device)
        self.target_value_net.to(device)
        self.target_policy_net.eval()
        self.target_value_net.eval()

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.kwargs.get('actor_lr', lr))
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.kwargs.get('critic_lr', lr))


    def take_action(self, state) -> int:
        if epsilon := self.kwargs.get('epsilon'):
            if  epsilon > 1 or epsilon < 0:
                raise ValueError('epsilon must be in (0, 1)')

        if epsilon and np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state.reshape(-1, self.state_dim), dtype=torch.float32).to(self.device)
            action_prob = self.target_policy_net(state)
            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample().item()

        return action
    
    def update(self) -> None:
        if len(self.replay_buffer) < self.kwargs.get('min_buffer_size', 1000):
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        v_value = self.value_net(states)

        with torch.no_grad(): 
            next_v_value = self.target_value_net(next_states)
            td_target = rewards + self.gamma * next_v_value * (1 - dones)
            td_delta = td_target - v_value
        
        value_loss = torch.nn.functional.mse_loss(v_value, td_target)
        log_prob = torch.log(self.policy_net(states).gather(1, actions))
        policy_loss = torch.mean(-log_prob * td_delta)

        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        value_loss.backward()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.kwargs.get('value_max_grad_norm', 0.5))
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.kwargs.get('policy_max_grad_norm', 0.5))
        self.value_optimizer.step()
        self.policy_optimizer.step()

        try: wandb.log({'Tr_value_loss': value_loss.item(), 'Tr_policy_loss': policy_loss.item()})
        except: pass

        self.count += 1
        if self.count % self.kwargs.get('update_target_frequency', 100) == 0:
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
            self.target_value_net.load_state_dict(self.value_net.state_dict())

        if self.kwargs.get('save_path') and self.count % self.kwargs.get('save_frequency', 1000) == 0:
            self.save()

    def save(self) -> None:
        path = Path(self.kwargs.get('save_path')) / f"{self.count}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path / "policy_net.pth")
        torch.save(self.value_net.state_dict(), path / "value_net.pth")

    def load(self, version:str) -> None:
        path = Path(self.kwargs.get('save_path'))
        if not path.exists():
            raise FileNotFoundError(f"{path} not found, Please set save_path in kwargs")
        if not version:
            version = max([int(i) for i in path.iterdir()])
        else:
            if not (path / version).exists():
                raise FileNotFoundError(f"{version} not found, Please set version in kwargs")
            
        self.policy_net.load_state_dict(torch.load(path / version / "policy_net.pth"))
        self.value_net.load_state_dict(torch.load(path / version / "value_net.pth"))