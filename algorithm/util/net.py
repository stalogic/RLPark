import torch
import numpy as np

class PolicyNetwork(torch.nn.Module):

    def __init__(self, state_dim_or_shape:int, action_dim_or_shape:int, hidden_dim:int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim_or_shape, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim_or_shape)
        self.bn0 = torch.nn.BatchNorm1d(state_dim_or_shape)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(action_dim_or_shape)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.bn0(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        x = self.softmax(x)
        return x
    

class ValueNetwork(torch.nn.Module):

    def __init__(self, state_dim_or_shape:int, hidden_dim:int):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim_or_shape, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        self.bn0 = torch.nn.BatchNorm1d(state_dim_or_shape)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.bn0(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class QValueNetwork(torch.nn.Module):

    def __init__(self, state_dim_or_shape:int, action_dim_or_shape:int, hidden_dim:int):
        super(QValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim_or_shape, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim_or_shape)
        self.bn0 = torch.nn.BatchNorm1d(state_dim_or_shape)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.bn0(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    

class ContinuousPolicyNetwork(torch.nn.Module):

    def __init__(self, state_dim_or_shape:int, action_dim_or_shape:int, hidden_dim:int):
        super(ContinuousPolicyNetwork, self).__init__()
        self.action_dim_or_shape = action_dim_or_shape

        self.fc1 = torch.nn.Linear(state_dim_or_shape, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, np.prod(action_dim_or_shape))
        self.fc_std = torch.nn.Linear(hidden_dim, np.prod(action_dim_or_shape))

        self.bn0 = torch.nn.BatchNorm1d(state_dim_or_shape)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn_mu = torch.nn.BatchNorm1d(np.prod(action_dim_or_shape))
        self.bn_std = torch.nn.BatchNorm1d(np.prod(action_dim_or_shape))

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        x = self.bn0(x)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        mu = 2.0 * self.tanh(self.bn_mu(self.fc_mu(x)))
        std = self.softplus(self.bn_std(self.fc_std(x)))
        mu = torch.reshape(mu, (-1, *self.action_dim_or_shape))
        std = torch.reshape(std, (-1, *self.action_dim_or_shape))
        return mu, std