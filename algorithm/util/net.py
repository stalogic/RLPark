import torch
import numpy as np

class PolicyNetwork(torch.nn.Module):

    def __init__(self, state_shape:tuple, action_dim:int, hidden_dim:int):
        super(PolicyNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.fc1 = torch.nn.Linear(np.prod(self.state_shape), hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, self.action_dim)
        self.bn0 = torch.nn.BatchNorm1d(np.prod(self.state_shape))
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(self.action_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, s):
        s = torch.reshape(s, (-1, np.prod(self.state_shape)))
        x = self.bn0(s)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        x = self.softmax(x)
        return x
    

class ValueNetwork(torch.nn.Module):

    def __init__(self, state_shape:tuple, hidden_dim:int):
        super(ValueNetwork, self).__init__()
        self.state_shape = state_shape
        self.fc1 = torch.nn.Linear(np.prod(self.state_shape), hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        self.bn0 = torch.nn.BatchNorm1d(np.prod(self.state_shape))
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, s):
        s = torch.reshape(s, (-1, np.prod(self.state_shape)))
        x = self.bn0(s)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class QValueNetwork(torch.nn.Module):

    def __init__(self, state_shape:tuple, action_dim:int, hidden_dim:int):
        super(QValueNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.fc1 = torch.nn.Linear(np.prod(self.state_shape), hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.bn0 = torch.nn.BatchNorm1d(np.prod(self.state_shape))
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, s):
        s = torch.reshape(s, (-1, np.prod(self.state_shape)))
        x = self.bn0(s)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    

class ContinuousPolicyNetwork(torch.nn.Module):

    def __init__(self, state_shape:tuple, action_shape:tuple, hidden_dim:int):
        super(ContinuousPolicyNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.fc1 = torch.nn.Linear(np.prod(state_shape), hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, np.prod(self.action_shape))
        self.fc_std = torch.nn.Linear(hidden_dim, np.prod(self.action_shape))

        self.bn0 = torch.nn.BatchNorm1d(np.prod(state_shape))
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn_mu = torch.nn.BatchNorm1d(np.prod(self.action_shape))
        self.bn_std = torch.nn.BatchNorm1d(np.prod(self.action_shape))

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.softplus = torch.nn.Softplus()

    def forward(self, s):
        s = torch.reshape(s, (-1, np.prod(self.state_shape)))
        x = self.bn0(s)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        mu = 2.0 * self.tanh(self.bn_mu(self.fc_mu(x)))
        std = self.softplus(self.bn_std(self.fc_std(x)))
        mu = torch.reshape(mu, (-1,) + self.action_shape)
        std = torch.reshape(std, (-1,) + self.action_shape)
        return mu, std
    

class DeterministicPolicyNetwork(torch.nn.Module):
    
    def __init__(self, state_shape:tuple, action_shape:tuple, action_bound:float|np.ndarray, hidden_dim:int):
        super(DeterministicPolicyNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_bound = torch.tensor(action_bound)

        self.fc1 = torch.nn.Linear(np.prod(state_shape), hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, np.prod(action_shape))

        self.bn0 = torch.nn.BatchNorm1d(np.prod(state_shape))
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(np.prod(action_shape))

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x) -> torch.Tensor:
        x = self.bn0(torch.reshape(x, (-1, np.prod(self.state_shape))))
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.tanh(self.bn3(self.fc3(x)))
        x = torch.reshape(x, (-1,)+self.action_shape)
        return x * self.action_bound


class ContinuousQValueNetwork(torch.nn.Module):
    
    def __init__(self, state_shape:tuple, action_shape:tuple, hidden_dim:int):
        super(ContinuousQValueNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.fc1 = torch.nn.Linear(np.prod(self.state_shape) + np.prod(self.action_shape), hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

        self.bn0 = torch.nn.BatchNorm1d(np.prod(self.state_shape) + np.prod(self.action_shape))
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()


    def forward(self, s, a) -> torch.Tensor:
        s = torch.reshape(s, (-1, np.prod(self.state_shape)))
        a = torch.reshape(a, (-1, np.prod(self.action_shape)))
        x = torch.cat((s, a), dim=1)
        x = self.bn0(x)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x