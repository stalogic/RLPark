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



# 踩过的坑
1. 更新target网络时，soft_update方法会使用
```python
for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
```
来向target网络更新参数，但是model.parameters()方法中没有BN层的统计参数（running_mean, running_var, num_batches_tracked），如果网络中使用了BN层，就不能使用这个方法来更新target网络。可以使用model.load_state_dict()方法来更新target网络。

```python
print(self.policy_net.state_dict().keys())
>>>
odict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked', 'bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked'])

print(dict(self.policy_net.named_parameters()).keys())  # named_parameters() 与 parameters() 类似，不过还会返回参数的名字。
>>>
dict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias', 'bn1.weight', 'bn1.bias', 'bn2.weight', 'bn2.bias', 'bn3.weight', 'bn3.bias'])
```