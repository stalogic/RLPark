import torch
import numpy as np
import wandb
from pathlib import Path
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


from ..util import (
    OnPolicyRLModel,
    ContinuousPolicyNetwork,
    ValueNetwork,
    compute_return,
)


class PPOContinuous(OnPolicyRLModel):

    def __init__(
        self,
        state_dim_or_shape,
        action_dim_or_shape,
        hidden_dims=(32,),
        conv_layers=((32, 3),),
        batch_size=128,
        lr=1e-3,
        gamma=0.99,
        lamda=0.95,
        eps=0.2,
        epochs=10,
        device="cpu",
        **kwargs,
    ) -> None:
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
        self.lr = lr
        self.gamma = gamma
        self.lamda = lamda
        self.eps = eps
        self.epochs = epochs
        self.device = device
        self.kwargs = kwargs

        self.policy_net = ContinuousPolicyNetwork(
            self.state_shape,
            self.action_shape,
            hidden_dims=hidden_dims,
            conv_layers=conv_layers,
            **kwargs,
        )
        self.value_net = ValueNetwork(
            self.state_shape, hidden_dims=hidden_dims, conv_layers=conv_layers, **kwargs
        )

        self.policy_net.to(device)
        self.value_net.to(device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.kwargs.get("actor_lr", lr)
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=self.kwargs.get("critic_lr", lr)
        )
        self.policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_optimizer,
            step_size=self.kwargs.get("scheduler_step_size", 100),
            gamma=0.955,
        )
        self.value_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.value_optimizer,
            step_size=self.kwargs.get("scheduler_step_size", 100),
            gamma=0.955,
        )

    def take_action(self, state) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad(), self.eval_mode():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            mu, std = self.policy_net(state)
            action_dist = torch.distributions.Normal(mu, std + 1e-6)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            action = np.reshape(action.cpu().numpy(), self.action_shape)
            log_prob = np.reshape(log_prob.cpu().numpy(), self.action_shape)
        return action, log_prob

    def take_action_with_mask(self, state, mask: list | np.ndarray) -> int:
        raise NotImplementedError("take_action_with_mask is not implemented")

    def predict_action(self, state) -> np.ndarray:
        with torch.no_grad(), self.eval_mode():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            mu, _ = self.policy_net(state)
            action = mu.cpu().numpy()
        return np.reshape(action, self.action_shape)

    def predict_action_with_mask(self, state, mask: list | np.ndarray) -> int:
        raise NotImplementedError("predict_action_with_mask is not implemented")

    def update(self, transition_dict) -> None:

        states = np.array(transition_dict["states"]).reshape(-1, *self.state_shape)
        actions = np.array(transition_dict["actions"]).reshape(-1, *self.action_shape)
        old_log_probs = np.array(transition_dict["log_probs"]).reshape(
            -1, *self.action_shape
        )
        next_states = np.array(transition_dict["next_states"]).reshape(
            -1, *self.state_shape
        )
        rewards = np.array(transition_dict["rewards"]).reshape(-1, 1)
        dones = np.array(transition_dict["dones"]).reshape(-1, 1)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        targets = compute_return(gamma=self.gamma, rewards=rewards)

        for _ in range(self.epochs):
            policy_loss, value_loss = 0.0, 0.0
            for indices in BatchSampler(
                SubsetRandomSampler(range(len(targets))),
                batch_size=min(256, len(targets)),
                drop_last=True,
            ):
                batch_states = states[indices].to(self.device)
                batch_targets = targets[indices].to(self.device)
                with torch.no_grad(), self.eval_mode():
                    batch_values = self.value_net(batch_states)
                    batch_advantages = batch_targets - batch_values

                batch_actions = actions[indices].to(self.device)
                batch_old_log_probs = old_log_probs[indices].to(self.device)

                batch_mu, batch_std = self.policy_net(batch_states)
                action_dist = torch.distributions.Normal(batch_mu, batch_std + 1e-6)
                batch_new_log_probs = action_dist.log_prob(batch_actions)

                batch_ratio = torch.exp(
                    torch.clamp(batch_new_log_probs - batch_old_log_probs, -1, 1)
                )
                batch_surr1 = batch_ratio * batch_advantages
                batch_surr2 = (
                    torch.clamp(batch_ratio, 1 - self.eps, 1 + self.eps)
                    * batch_advantages
                )
                batch_policy_loss = -torch.mean(torch.min(batch_surr1, batch_surr2))
                batch_value_loss = torch.nn.functional.mse_loss(
                    self.value_net(batch_states), batch_targets
                )

                self.policy_optimizer.zero_grad()
                batch_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(),
                    self.kwargs.get("policy_max_grad_norm", 0.5),
                )
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                batch_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(),
                    self.kwargs.get("value_max_grad_norm", 0.5),
                )
                self.value_optimizer.step()

                policy_loss += batch_policy_loss.item()
                value_loss += batch_value_loss.item()

            try:
                wandb.log(
                    {
                        "Tr_value_loss": value_loss,
                        "Tr_policy_loss": policy_loss,
                        "Tr_value_learning_rate": self.value_lr_scheduler.get_last_lr()[
                            0
                        ],
                        "Tr_policy_learning_rate": self.policy_lr_scheduler.get_last_lr()[
                            0
                        ],
                    }
                )
            except:
                pass

            self.count += 1
            if (
                self.kwargs.get("save_path")
                and self.count % self.kwargs.get("save_frequency", 1000) == 0
            ):
                self.save()

    def save(self) -> None:
        path = Path(self.kwargs.get("save_path")) / f"{self.count}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path / "policy_net.pth")
        torch.save(self.value_net.state_dict(), path / "value_net.pth")

    def load(self, version: str) -> None:
        path = Path(self.kwargs.get("save_path"))
        if not path.exists():
            raise FileNotFoundError(f"{path} not found, Please set save_path in kwargs")
        if not version:
            version = max([int(i) for i in path.iterdir()])
        else:
            if not (path / version).exists():
                raise FileNotFoundError(
                    f"{version} not found, Please set version in kwargs"
                )

        self.policy_net.load_state_dict(torch.load(path / version / "policy_net.pth"))
        self.value_net.load_state_dict(torch.load(path / version / "value_net.pth"))