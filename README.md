# RLPark
 Toy Code for Reinforment Learning


# 重要tricks：
1. 加入梯度裁剪`torch.nn.utils.clip_grad_norm_(q_net.parameters(), 0.5)`，通过梯度裁剪，防止梯度爆炸，防止损失函数出现几个数量级的差异。
2. 在离散动作空间的环境中，策略网络使用Softmax输出概率分布，容易出现数值溢出，导致训练中断，需要在Policy网络中加入BN层来缓解。在**MountainCar-V0**中还能提升模型效果。
```python
class PolicyNetwork(torch.nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(action_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        x = self.softmax(x)
        return x
    
```

3. 使用`Policy`网络输出的概率分布来采样`action`时，需要注意`Policy`是否过早收敛，可以加入`epsilon`贪心策略，避免模型过拟合。