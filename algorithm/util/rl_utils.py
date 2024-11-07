import os
import time
import tqdm
import torch
import random
import numpy as np
import wandb
from collections import defaultdict

from . import OnPolicyRLModel, OffPolicyRLModel, ReplayBuffer, PrioritizedReplayBuffer


def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class MovingAverage(object):
    def __init__(self, gamma=0.9) -> None:
        self.gamma = gamma
        self.sum = 0.0
        self.factor = 1.0

    def append(self, value):
        self.sum *= self.gamma
        self.sum += value
        self.factor *= self.gamma
        self.factor += 1.0

    @property
    def value(self) -> int:
        return int(self.sum / self.factor)


def compute_advantage(gamma, lamda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lamda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def compute_return(gamma, rewards):
    rewards = rewards.detach().cpu().numpy()
    return_list = []
    cum_reward = 0.0
    for reward in reversed(rewards):
        cum_reward = reward + gamma * cum_reward
        return_list.append(cum_reward)
    return_list.reverse()
    return torch.tensor(np.array(return_list), dtype=torch.float)


def train_and_evaluate(env, agent, num_episodes=1000, val_env=None, **kwargs):
    """
    训练和评估一个智能体在一个给定环境中的表现。

    参数:
    - env: 环境接口，用于模拟智能体所处的环境。
    - agent: 智能体接口，用于执行动作和学习经验。
    - num_episodes: 训练和评估的总episode数量，默认为1000。
    - **kwargs: 可变参数字典，用于提供更多配置选项，包括但不限于:
        - num_eval_frequence: 评估频率，即每隔多少个episode进行一次性能评估，默认若未提供则为100。
        - num_eval_episodes: 在每次评估周期中执行的评估episode数量，默认若未提供则为10。
        其他自定义配置项也可通过此方式传递，但需确保在函数内部有相应的处理逻辑。

    返回值:
    - 无
    """

    if hasattr(agent, "replay_buffer"):
        train_and_evaluate_offpolicy_agent(env, agent, num_episodes, val_env, **kwargs)
    else:
        train_and_evaluate_onpolicy_agent(env, agent, num_episodes, val_env, **kwargs)


def train_and_evaluate_offpolicy_agent(
    env, agent: OffPolicyRLModel, num_episodes=1000, val_env=None, **kwargs
):

    # 统计单局游戏的移动平均步数，即episode长度的移动平均
    game_step = MovingAverage()
    train_freq = None

    # 初始化用于记录指标的字典
    metrics = {
        "episode_return": [],  # 每个episode的回报
        "episode_step": [],  # 每个episode的步数
        "episode_done": [],  # 每个episode是否完成
        "episode_terminal": [],  # 每个episode是否到达终止状态
    }

    # 使用tqdm创建进度条以可视化训练进度
    with tqdm.tqdm(range(num_episodes), desc="Training") as pbar:
        for i_episode in pbar:
            # 重置环境，开始新episode
            state, _ = env.reset()
            game_time, update_time = 0.0, 0.0
            episode_return, episode_step, episode_done, episode_terminal = (
                0.0,
                0,
                None,
                None,
            )
            while True:
                t0 = time.time()
                # 选择并执行动作
                if hasattr(env, "action_mask"):
                    action = agent.take_action_with_mask(state, env.action_mask)
                else:
                    action = agent.take_action(state)
                # 环境反馈，更新状态
                next_state, reward, done, terminal, _ = env.step(action)
                # 学习经验
                agent.add_experience(state, action, reward, next_state, done)
                state = next_state

                episode_return += reward
                episode_step += 1

                game_time += time.time() - t0

                # 更新智能体模型
                if train_freq is not None and episode_step % train_freq == 0:
                    if len(agent.replay_buffer) >= agent.batch_size:
                        if isinstance(agent.replay_buffer, ReplayBuffer):
                            transitions = agent.replay_buffer.sample(agent.batch_size)
                            losses, td_error = agent.update(transitions)
                        elif isinstance(agent.replay_buffer, PrioritizedReplayBuffer):
                            transitions, weights, tree_idxs = (
                                agent.replay_buffer.sample(agent.batch_size)
                            )
                            losses, td_error = agent.update(
                                transitions, weights=weights
                            )
                            agent.replay_buffer.update_priorities(
                                tree_idxs, td_error.cpu().numpy()
                            )

                        else:
                            raise RuntimeError("Unknown Replay Buffer")

                update_time += time.time() - t0

                # 判断episode是否结束
                if done or terminal:
                    episode_done = int(done)
                    episode_terminal = int(terminal)
                    break

            # 更新学习率
            agent.update_learning_rate()

            game_step.append(episode_step)
            train_freq = game_step.value // kwargs.get("trains_per_episode", 10)
            if train_freq <= 0:
                train_freq = 1

            logdata = {
                "TR_Index": i_episode,
                "TR_Episode_Return": episode_return,
                "TR_Episode_Step": episode_step,
                "TR_Episode_Done": episode_done,
                "TR_Episode_Terminal": episode_terminal,
                "TR_Game_Time": game_time,
                "TR_Update_Time": update_time - game_time,
            }

            try:
                wandb.log(logdata)
            except:
                pass

            # 达到评估周期时进行性能评估
            if (i_episode + 1) % kwargs.get("num_eval_frequence", 100) == 0:
                # 执行多轮评估episode
                for i_val in tqdm.tqdm(
                    range(kwargs.get("num_eval_episodes", 100)),
                    desc="Evaluating",
                    leave=False,
                ):
                    episode_return, episode_step = 0.0, 0

                    if i_val % kwargs.get("render_period", 1) == 0:
                        env_ = val_env if val_env else env
                    else:
                        env_ = env
                    state, _ = env_.reset()
                    while True:
                        # 使用预测模式选择动作
                        if hasattr(env_, "action_mask"):
                            action = agent.predict_action_with_mask(
                                state, env_.action_mask
                            )
                        else:
                            action = agent.predict_action(state)
                        next_state, reward, done, terminal, _ = env_.step(action)
                        state = next_state

                        # 累加评估指标
                        episode_return += reward
                        episode_step += 1

                        # 记录评估结果
                        if done or terminal:
                            metrics["episode_return"].append(episode_return)
                            metrics["episode_step"].append(episode_step)
                            metrics["episode_done"].append(int(done))
                            metrics["episode_terminal"].append(int(terminal))
                            break

                # 计算平均指标并记录日志
                logdata = {
                    "Ev_Index": i_episode,
                    "Ev_Avg_Return": np.mean(metrics["episode_return"]),
                    "Ev_Avg_Step": np.mean(metrics["episode_step"]),
                    "Ev_Avg_Done": np.mean(metrics["episode_done"]),
                    "Ev_Avg_Terminal": np.mean(metrics["episode_terminal"]),
                }

                # 清空指标准备下一轮评估
                for key in metrics.keys():
                    metrics[key] = []

                # 更新进度条信息，并尝试通过wandb记录日志
                pbar.set_postfix(logdata)
                try:
                    wandb.log(logdata)
                except:
                    pass  # 忽略wandb未安装或配置不当的情况


def train_and_evaluate_onpolicy_agent(
    env, agent: OnPolicyRLModel, num_episodes=1000, val_env=None, **kwargs
):

    # 初始化用于记录指标的字典
    metrics = {
        "episode_return": [],  # 每个episode的回报
        "episode_step": [],  # 每个episode的步数
        "episode_done": [],  # 每个episode是否完成
        "episode_terminal": [],  # 每个episode是否到达终止状态
    }

    # 使用tqdm创建进度条以可视化训练进度
    with tqdm.tqdm(range(num_episodes), desc="Training") as pbar:
        for i_episode in pbar:
            # 重置环境，开始新episode
            state, _ = env.reset()
            transition_dict = defaultdict(list)
            episode_return, episode_step, episode_done, episode_terminal = (
                0.0,
                0,
                None,
                None,
            )
            t0 = time.time()
            while True:
                # 选择并执行动作
                if hasattr(env, "action_mask"):
                    action = agent.take_action_with_mask(state, env.action_mask)
                else:
                    action = agent.take_action(state)
                if isinstance(action, tuple) and len(action) == 2:
                    action, log_prob = action
                else:
                    log_prob = None
                # 环境反馈，更新状态
                next_state, reward, done, terminal, _ = env.step(action)
                # 学习经验
                transition_dict["states"].append(state)
                transition_dict["actions"].append(action)
                transition_dict["next_states"].append(next_state)
                transition_dict["rewards"].append(reward)
                transition_dict["dones"].append(done)
                if log_prob:
                    transition_dict["log_probs"].append(log_prob)
                state = next_state

                episode_return += reward
                episode_step += 1

                # 判断episode是否结束
                if done or terminal:
                    episode_done = int(done)
                    episode_terminal = int(terminal)
                    break

            t1 = time.time()

            # 更新智能体模型
            agent.update(transition_dict)
            agent.update_learning_rate()

            t2 = time.time()
            logdata = {
                "TR_Index": i_episode,
                "TR_Episode_Return": episode_return,
                "TR_Episode_Step": episode_step,
                "TR_Episode_Done": episode_done,
                "TR_Episode_Terminal": episode_terminal,
                "TR_Game_Time": t1 - t0,
                "TR_Update_Time": t2 - t1,
            }

            # 更新进度条信息，并尝试通过wandb记录日志
            # pbar.set_postfix(logdata)
            try:
                wandb.log(logdata)
            except:
                pass  # 忽略wandb未安装或配置不当的情况

            # 达到评估周期时进行性能评估
            if (i_episode + 1) % kwargs.get("num_eval_frequence", 100) == 0:
                # 执行多轮评估episode
                for i_val in tqdm.tqdm(
                    range(kwargs.get("num_eval_episodes", 100)),
                    desc="Evaluating",
                    leave=False,
                ):
                    episode_return, episode_step = 0.0, 0

                    if i_val % kwargs.get("render_period", 1) == 0:
                        env_ = val_env if val_env else env
                    else:
                        env_ = env
                    state, _ = env_.reset()
                    while True:
                        # 使用预测模式选择动作
                        if hasattr(env_, "action_mask"):
                            action = agent.predict_action_with_mask(
                                state, env_.action_mask
                            )
                        else:
                            action = agent.predict_action(state)
                        next_state, reward, done, terminal, _ = env_.step(action)
                        state = next_state

                        # 累加评估指标
                        episode_return += reward
                        episode_step += 1

                        # 记录评估结果
                        if done or terminal:
                            metrics["episode_return"].append(episode_return)
                            metrics["episode_step"].append(episode_step)
                            metrics["episode_done"].append(int(done))
                            metrics["episode_terminal"].append(int(terminal))
                            break

                # 计算平均指标并记录日志
                logdata = {
                    "Ev_Index": i_episode,
                    "Ev_Avg_Return": np.mean(metrics["episode_return"]),
                    "Ev_Avg_Step": np.mean(metrics["episode_step"]),
                    "Ev_Avg_Done": np.mean(metrics["episode_done"]),
                    "Ev_Avg_Terminal": np.mean(metrics["episode_terminal"]),
                }

                # 清空指标准备下一轮评估
                for key in metrics.keys():
                    metrics[key] = []

                # 更新进度条信息，并尝试通过wandb记录日志
                pbar.set_postfix(logdata)
                try:
                    wandb.log(logdata)
                except:
                    pass  # 忽略wandb未安装或配置不当的情况
