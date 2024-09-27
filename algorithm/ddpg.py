import copy
import wandb
import torch
import numpy as np
from .util import (
    OffPolicyRLModel,
    DeterministicPolicyNetwork,
    DiscreteDeterministicPolicyNetwork,
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


class DiscreteDDPG(OffPolicyRLModel):
    """使用Gumbel-Softmax重参数法，将DDPG算法应用与离散动作空间"""

    @staticmethod
    def onehot_from_logits(logits, eps=0.01):
        """生成最优动作的独热（one-hot）形式"""
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
        # 生成随机动作,转换成独热形式
        rand_acs = torch.autograd.Variable(
            torch.eye(logits.shape[1])[
                [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
            ],
            requires_grad=False,
        ).to(logits.device)
        # 通过epsilon-贪婪算法来选择用哪个动作
        return torch.stack(
            [
                argmax_acs[i] if r > eps else rand_acs[i]
                for i, r in enumerate(torch.rand(logits.shape[0]))
            ]
        )

    @staticmethod
    def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
        """从Gumbel(0,1)分布中采样"""
        U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
        return -torch.log(-torch.log(U + eps) + eps)

    @staticmethod
    def gumbel_softmax_sample(logits, temperature):
        """从Gumbel-Softmax分布中采样"""
        y = logits + DiscreteDDPG.sample_gumbel(
            logits.shape, tens_type=type(logits.data)
        ).to(logits.device)
        return torch.nn.functional.softmax(y / temperature, dim=1)

    @staticmethod
    def gumbel_softmax(logits, temperature=1.0):
        """从Gumbel-Softmax分布中采样,并进行离散化"""
        y = DiscreteDDPG.gumbel_softmax_sample(logits, temperature)
        y_hard = DiscreteDDPG.onehot_from_logits(y)
        y = (y_hard.to(logits.device) - y).detach() + y
        # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
        # 正确地反传梯度
        return y

    def __init__(
        self,
        state_dim_or_shape,
        action_dim_or_shape,
        hidden_dims=(32,),
        conv_layers=((32, 3),),
        batch_size=128,
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
        assert len(self.action_shape) == 1, "action_dim_or_shape must be a int"
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.kwargs = kwargs

        self.policy_net = DiscreteDeterministicPolicyNetwork(
            self.state_shape,
            self.action_shape,
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
            action = self.policy_net(state)
            action = DiscreteDDPG.gumbel_softmax(action).cpu().numpy()
        return np.reshape(action, self.action_shape)

    def predict_action(self, state) -> np.ndarray:
        with torch.no_grad(), self.eval_mode():
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.policy_net(state)
            action = DiscreteDDPG.onehot_from_logits(action).cpu().numpy()
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
            next_action_logits = self.target_policy_net(next_states)
            next_action = DiscreteDDPG.onehot_from_logits(next_action_logits)
            next_q_values = self.target_qvalue_net(next_states, next_action)
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        value_loss = torch.nn.functional.mse_loss(
            self.qvalue_net(states, actions), q_targets
        )
        self.qvalue_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qvalue_net.parameters(), 0.5)
        self.qvalue_optimizer.step()

        action_logits = self.policy_net(states)
        policy_loss = -torch.mean(
            self.qvalue_net(states, DiscreteDDPG.gumbel_softmax(action_logits))
        )
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
