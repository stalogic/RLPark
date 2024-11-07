import copy
import torch
import numpy as np
import wandb

from ..util import (
    OffPolicyRLModel,
    ContinuousPolicyNetwork,
    ContinuousQValueNetwork,
)


class SACContinuous(OffPolicyRLModel):

    def __init__(
        self,
        state_dim_or_shape,
        action_dim_or_shape,
        hidden_dims=(32,),
        conv_layers=((32, 3),),
        action_bound=1.0,
        batch_size=128,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        target_entropy=1.0,
        device="cpu",
        **kwargs,
    ):
        super().__init__(state_dim_or_shape, action_dim_or_shape, **kwargs)
        if not isinstance(state_dim_or_shape, (int, tuple, list)):
            raise TypeError(f"state_dim_or_shape must be int, tuple or list")
        if not isinstance(action_dim_or_shape, (int, tuple, list)):
            raise TypeError(f"action_dim_or_shape must be int, tuple or list")

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
        self.batch_size = batch_size
        self.action_bound = torch.tensor(
            action_bound, dtype=torch.float32, device=device
        )
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.device = device
        self.kwargs = kwargs

        self.policy_net = ContinuousPolicyNetwork(
            self.state_shape,
            self.action_shape,
            hidden_dims=hidden_dims,
            conv_layers=conv_layers,
            **kwargs,
        ).to(self.device)
        self.q1_net = ContinuousQValueNetwork(
            self.state_shape,
            self.action_shape,
            hidden_dims=hidden_dims,
            conv_layers=conv_layers,
            **kwargs,
        ).to(self.device)
        self.q2_net = ContinuousQValueNetwork(
            self.state_shape,
            self.action_shape,
            hidden_dims=hidden_dims,
            conv_layers=conv_layers,
            **kwargs,
        ).to(self.device)
        self.target_q1_net = copy.deepcopy(self.q1_net).to(self.device).eval()
        self.target_q2_net = copy.deepcopy(self.q2_net).to(self.device).eval()
        self.log_alpha = torch.tensor(
            np.log(0.01), dtype=torch.float32, requires_grad=True, device=self.device
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.kwargs.get("policy_lr", self.lr)
        )
        self.q1_optimizer = torch.optim.Adam(
            self.q1_net.parameters(), lr=self.kwargs.get("value_lr", self.lr)
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2_net.parameters(), lr=self.kwargs.get("value_lr", self.lr)
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.kwargs.get("alpha_lr", 1e-2)
        )
        self.policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_optimizer,
            step_size=self.kwargs.get("scheduler_step_size", 100),
            gamma=0.955,
        )
        self.q1_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.q1_optimizer,
            step_size=self.kwargs.get("scheduler_step_size", 100),
            gamma=0.955,
        )
        self.q2_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.q2_optimizer,
            step_size=self.kwargs.get("scheduler_step_size", 100),
            gamma=0.955,
        )

    def take_action(self, state: list | np.ndarray) -> list | np.ndarray:
        with torch.no_grad(), self.eval_mode():
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            mu, std = self.policy_net(state)
            action_dist = torch.distributions.Normal(mu, std)
            sample = action_dist.sample()
            action = torch.tanh(sample) * self.action_bound
        return np.reshape(action.cpu().numpy(), self.action_shape)

    def predict_action(self, state: list | np.ndarray) -> list | np.ndarray:
        with torch.no_grad(), self.eval_mode():
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            mu, _ = self.policy_net(state)
            action = torch.tanh(mu) * self.action_bound
        return np.reshape(action.cpu().numpy(), self.action_shape)

    def _calc_action_log_prob(self, state: list | np.ndarray) -> torch.Tensor:
        mu, std = self.policy_net(state)
        action_dist = torch.distributions.Normal(mu, std)
        log_prob = action_dist.log_prob(mu)
        sample = action_dist.rsample()
        action = torch.tanh(sample)
        log_prob = log_prob - torch.log(1 - action.square() + 1e-7)
        action = action * self.action_bound
        return action, log_prob

    def _calc_target(self, rewards, next_states, dones):
        with torch.no_grad(), self.eval_mode():
            next_actions, log_probs = self._calc_action_log_prob(next_states)
            entropy = -log_probs
            q1_values = self.target_q1_net(next_states, next_actions)
            q2_values = self.target_q2_net(next_states, next_actions)
            next_values = (
                torch.min(q1_values, q2_values) + self.log_alpha.exp() * entropy
            )
            td_target = rewards + (1.0 - dones) * self.gamma * next_values
        return td_target

    def update(self):
        if len(self.replay_buffer) < self.batch_size * 10:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = torch.tensor(states, dtype=torch.float32, device=self.device).reshape(
            -1, *self.state_shape
        )
        actions = torch.tensor(
            actions, dtype=torch.float32, device=self.device
        ).reshape(-1, *self.action_shape)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        ).reshape(-1, *self.state_shape)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).reshape(
            -1, 1
        )

        td_target = self._calc_target(rewards, next_states, dones)
        q1_loss = torch.nn.functional.mse_loss(self.q1_net(states, actions), td_target)
        q2_loss = torch.nn.functional.mse_loss(self.q2_net(states, actions), td_target)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q1_net.parameters(), self.kwargs.get("value_max_grad_norm", 0.5)
        )
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q2_net.parameters(), self.kwargs.get("value_max_grad_norm", 0.5)
        )
        self.q2_optimizer.step()

        new_actions, log_probs = self._calc_action_log_prob(states)
        entropy = -log_probs
        q1_values = self.q1_net(states, new_actions)
        q2_values = self.q2_net(states, new_actions)
        policy_loss = -torch.mean(
            self.log_alpha.exp() * entropy + torch.min(q1_values, q2_values)
        )
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.kwargs.get("policy_max_grad_norm", 0.5)
        )
        self.policy_optimizer.step()

        alpha_loss = -torch.mean(
            self.log_alpha.exp() * (entropy - self.target_entropy).detach()
        )
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update_target_net(self.q1_net, self.target_q1_net, self.tau)
        self._soft_update_target_net(self.q2_net, self.target_q2_net, self.tau)

        log_data = {
            "Tr_q1_loss": q1_loss.item(),
            "Tr_q2_loss": q2_loss.item(),
            "Tr_policy_loss": policy_loss.item(),
            "Tr_alpha_loss": alpha_loss.item(),
            "Tr_alpha": self.log_alpha.exp().item(),
            "Tr_entropy": entropy.mean().item(),
            "Tr_entropy_gap": (entropy - self.target_entropy).mean().item(),
            "Tr_q1_learning_rate": self.q1_lr_scheduler.get_last_lr()[0],
            "Tr_policy_learning_rate": self.policy_lr_scheduler.get_last_lr()[0],
        }

        try:
            wandb.log(log_data)
        except:
            pass
