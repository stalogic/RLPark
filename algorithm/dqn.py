import torch
import copy
import wandb
import numpy as np
from pathlib import Path
from .util import OffPolicyRLModel, QValueNetwork, ValueAdvanceNetwork


class DQN(OffPolicyRLModel):

    def __init__(
        self,
        state_dim_or_shape,
        action_dim_or_shape,
        hidden_dims=(32,),
        conv_layers=((32, 3),),
        batch_size=128,
        epsilon=0.05,
        lr=1e-3,
        gamma=0.99,
        dueling_dqn: bool = False,
        double_dqn: bool = False,
        device="cpu",
        capacity=10000,
        **kwargs,
    ) -> None:
        super().__init__(state_dim_or_shape, action_dim_or_shape, capacity, **kwargs)
        if not isinstance(state_dim_or_shape, (int, tuple, list)):
            raise TypeError("state_dim_or_shape must be int, tuple or list")
        if not isinstance(action_dim_or_shape, (int, tuple, list)):
            raise TypeError("action_dim_or_shape must be int, tuple or list")

        self.state_shape = (
            (state_dim_or_shape,)
            if isinstance(state_dim_or_shape, int)
            else tuple(state_dim_or_shape)
        )
        self.action_dim = (
            action_dim_or_shape[0]
            if not isinstance(action_dim_or_shape, int)
            else action_dim_or_shape
        )
        self.batch_size = batch_size
        self.init_epsilon = 1.5
        self.min_epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.dueling_dqn = dueling_dqn
        self.double_dqn = double_dqn
        self.device = device
        self.kwargs = kwargs

        if self.dueling_dqn:
            self.q_net = ValueAdvanceNetwork(
                self.state_shape,
                self.action_dim,
                hidden_dims=hidden_dims,
                conv_layers=conv_layers,
                **kwargs,
            )
        else:
            self.q_net = QValueNetwork(
                self.state_shape,
                self.action_dim,
                hidden_dims=hidden_dims,
                conv_layers=conv_layers,
                **kwargs,
            )
        self.target_q_net = copy.deepcopy(self.q_net)

        self.q_net.to(device)
        self.target_q_net.to(device)
        self.target_q_net.eval()

        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.q_net_scheduler = torch.optim.lr_scheduler.StepLR(
            self.q_net_optimizer,
            step_size=self.kwargs.get("scheduler_step_size", 100),
            gamma=0.955,
        )

    @property
    def epsilon(self) -> float:
        self.init_epsilon *= 1 - 1e-6
        ep = (
            self.init_epsilon
            if self.init_epsilon > self.min_epsilon
            else self.min_epsilon
        )
        try:
            wandb.log({"TR_Epsilon": ep})
        except:
            pass
        return ep

    def take_action(self, state) -> int:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad(), self.eval_mode():
                state = torch.tensor(state, dtype=torch.float).to(self.device)
                action = self.q_net(state).argmax().item()
        return action

    def take_action_with_mask(self, state, mask: list | np.ndarray) -> int:
        if not isinstance(mask, (list, np.ndarray)) or len(mask) != self.action_dim:
            raise ValueError(
                "mask must be a list or numpy array with length of action_dim"
            )
        if isinstance(mask, list):
            mask = np.array(mask)
        mask = mask.astype(int)

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_dim, p=mask / mask.sum())
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

    def predict_action_with_mask(self, state, mask: list | np.ndarray) -> int:
        if not isinstance(mask, (list, np.ndarray)) or len(mask) != self.action_dim:
            raise ValueError(
                "mask must be a list or numpy array with length of action_dim"
            )
        if isinstance(mask, list):
            mask = np.array(mask)
        mask = mask.astype(int)

        with torch.no_grad(), self.eval_mode():
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            q_value = self.q_net(state).numpy()
            q_value[:, mask == 0] = -np.inf
            action = q_value.argmax()
        return action

    def update(
        self, transitions: tuple, weights=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states, actions, rewards, next_states, dones = transitions
        states = states.to(dtype=torch.float, device=self.device)
        actions = actions.view(-1, 1).to(dtype=torch.int64, device=self.device)
        rewards = rewards.view(-1, 1).to(dtype=torch.float, device=self.device)
        next_states = next_states.to(dtype=torch.float, device=self.device)
        dones = dones.view(-1, 1).to(dtype=torch.float, device=self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad(), self.eval_mode():
            if self.double_dqn:
                max_action = self.q_net(next_states).argmax(1).view(-1, 1)
                next_q_values = self.target_q_net(next_states).gather(1, max_action)
            else:
                next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_target = rewards + self.gamma * next_q_values * (1 - dones)

        if weights is None:
            weights = torch.ones_like(q_target)
        weights = weights.to(self.device)

        td_error = torch.abs(q_target - q_values).detach().squeeze()
        loss = torch.mean((q_target - q_values) ** 2 * weights)
        self.q_net_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), self.kwargs.get("max_grad_norm", 0.5)
        )
        self.q_net_optimizer.step()

        try:
            wandb.log(
                {
                    "Tr_loss": loss.item(),
                    "Tr_learning_rate": self.q_net_scheduler.get_last_lr()[0],
                }
            )
        except:
            pass

        self.count += 1
        if tau := self.kwargs.get("tau"):
            for param_name in self.q_net.state_dict().keys():
                target_param = self.target_q_net.state_dict()[param_name]
                param = self.q_net.state_dict()[param_name]
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
        elif self.count % self.kwargs.get("target_update_frequency", 100) == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        if (
            self.kwargs.get("save_path")
            and self.count % self.kwargs.get("save_frequency", 1000) == 0
        ):
            self.save()

        return loss, td_error

    def save(self) -> None:
        path = Path(self.kwargs.get("save_path")) / f"{self.count}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path / "q_net.pth")

    def load(self, version: str):
        pass
