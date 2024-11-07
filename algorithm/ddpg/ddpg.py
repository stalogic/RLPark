import copy
import wandb
import torch
import numpy as np
from ..util import (
    OffPolicyRLModel,
    DeterministicPolicyNetwork,
    ContinuousQValueNetwork,
)


class DDPG(OffPolicyRLModel):

    def __init__(
        self,
        state_dim_or_shape,
        action_dim_or_shape,
        hidden_dims=(32,),
        conv_layers=((32, 3),),
        batch_size=128,
        action_bound=1,
        sigma=0.05,
        lr=1e-3,
        gamma=0.99,
        capacity=10000,
        device="cpu",
        **kwargs,
    ) -> None:
        super().__init__(state_dim_or_shape, action_dim_or_shape, capacity)

        if not isinstance(state_dim_or_shape, (int, tuple, list)):
            raise TypeError("state_dim_or_shape must be int, tuple or list")
        if not isinstance(action_dim_or_shape, (int, tuple, list)):
            raise TypeError("action_dim_or_shape must be int, tuple or list")

        self.state_shape = (
            (state_dim_or_shape,)
            if isinstance(state_dim_or_shape, int)
            else tuple(state_dim_or_shape)
        )
        self.action_shape = (
            (action_dim_or_shape,)
            if isinstance(action_dim_or_shape, int)
            else tuple(action_dim_or_shape)
        )
        self.action_bound = action_bound
        self.sigma = sigma
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.kwargs = kwargs

        self.policy_net = DeterministicPolicyNetwork(
            self.state_shape,
            self.action_shape,
            action_bound,
            hidden_dims=hidden_dims,
            conv_layers=conv_layers,
            **kwargs,
        )
        self.qvalue_net = ContinuousQValueNetwork(
            self.state_shape,
            self.action_shape,
            hidden_dims=hidden_dims,
            conv_layers=conv_layers,
            **kwargs,
        )
        self.target_policy_net = copy.deepcopy(self.policy_net)
        self.target_qvalue_net = copy.deepcopy(self.qvalue_net)

        self.policy_net.to(device)
        self.qvalue_net.to(device)
        self.target_policy_net.to(device)
        self.target_qvalue_net.to(device)
        self.target_policy_net.eval()
        self.target_qvalue_net.eval()

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.kwargs.get("actor_lr", lr)
        )
        self.qvalue_optimizer = torch.optim.Adam(
            self.qvalue_net.parameters(), lr=self.kwargs.get("critic_lr", lr)
        )
        self.policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_optimizer,
            step_size=self.kwargs.get("scheduler_step_size", 100),
            gamma=0.955,
        )
        self.qvalue_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.qvalue_optimizer,
            step_size=self.kwargs.get("scheduler_step_size", 100),
            gamma=0.955,
        )

    def take_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad(), self.eval_mode():
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.policy_net(state).cpu().numpy()
            action = action + self.sigma * np.random.randn(*action.shape)
        return np.reshape(action, self.action_shape)

    def predict_action(self, state) -> np.ndarray:
        with torch.no_grad(), self.eval_mode():
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.policy_net(state).cpu().numpy()
        return np.reshape(action, self.action_shape)

    def update(self) -> None:
        if len(self.replay_buffer) < self.batch_size * 2:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad(), self.eval_mode():
            next_q_values = self.target_qvalue_net(
                next_states, self.target_policy_net(next_states)
            )
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        value_loss = torch.nn.functional.mse_loss(
            self.qvalue_net(states, actions), q_targets
        )
        self.qvalue_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qvalue_net.parameters(), 0.5)
        self.qvalue_optimizer.step()

        policy_loss = -torch.mean(self.qvalue_net(states, self.policy_net(states)))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()

        try:
            wandb.log(
                {
                    "Tr_policy_loss": policy_loss.item(),
                    "Tr_value_loss": value_loss.item(),
                    "Tr_policy_lr": self.policy_lr_scheduler.get_last_lr()[0],
                    "Tr_value_lr": self.qvalue_lr_scheduler.get_last_lr()[0],
                }
            )
        except:
            pass

        self.count += 1
        if tau := self.kwargs.get("tau"):
            for param_name in self.target_policy_net.state_dict().keys():
                target_param = self.target_policy_net.state_dict()[param_name]
                param = self.policy_net.state_dict()[param_name]
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
            for param_name in self.target_qvalue_net.state_dict().keys():
                target_param = self.target_qvalue_net.state_dict()[param_name]
                param = self.qvalue_net.state_dict()[param_name]
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
        elif self.count % self.kwargs.get("target_update_frequency", 100) == 0:
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
            self.target_qvalue_net.load_state_dict(self.qvalue_net.state_dict())
