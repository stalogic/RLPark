import torch
import tqdm
import numpy as np
import wandb

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def train_and_evaluate(env, agent, num_episodes=1000, **kwargs):
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
            while True:
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
                
                # 判断episode是否结束
                if done or terminal:
                    break
            
            # 更新智能体模型
            agent.update()

            # 达到评估周期时进行性能评估
            if i_episode > 0 and i_episode % kwargs.get("num_eval_frequence", 100) == 0:
                # 执行多轮评估episode
                for _ in tqdm.tqdm(range(kwargs.get("num_eval_episodes", 100)), desc="Evaluating", leave=False):
                    episode_return, episode_step = 0.0, 0
                    state, _ = env.reset()
                    while True:
                        # 使用预测模式选择动作
                        if hasattr(env, "action_mask"):
                            action = agent.predict_action_with_mask(state, env.action_mask)
                        else:
                            action = agent.predict_action(state)
                        next_state, reward, done, terminal, _ = env.step(action)
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
                try: wandb.log(logdata)
                except: pass  # 忽略wandb未安装或配置不当的情况

