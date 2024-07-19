import torch
import numpy as np

from .net import MLPNetwork, CNNNetwork

class ValueNetwork(torch.nn.Module):

    def __init__(self, state_shape:tuple, hidden_dims:tuple=(128,), conv_layers:tuple=((32, 3),), **kwargs):
        super(ValueNetwork, self).__init__()
        self.state_shape = state_shape
        
        if len(self.state_shape) < 3:
            self.mlp = MLPNetwork(np.prod(state_shape), 1, hidden_dims, **kwargs)
        elif len(self.state_shape) == 3:
            self.cnn = CNNNetwork(self.state_shape, hidden_dims[0], conv_layers, **kwargs)
            self.mlp = MLPNetwork(hidden_dims[0], 1, hidden_dims, **kwargs)
        else:
            raise ValueError("state_shape must be 1D or 3D")

        print(self)

    def forward(self, s):
        if hasattr(self, "cnn"):
            s = torch.reshape(s, (-1, *self.state_shape))
            x = self.cnn(s)
            x = self.mlp(x)
        else:
            s = torch.reshape(s, (-1, np.prod(self.state_shape)))
            x = self.mlp(s)
        return x
    

class QValueNetwork(torch.nn.Module):

    def __init__(self, state_shape:tuple, action_dim:int, hidden_dims:tuple=(128,), conv_layers:tuple=((32, 3),), **kwargs):
        super(QValueNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim

        if len(self.state_shape) < 3:
            self.mlp = MLPNetwork(np.prod(state_shape), action_dim, hidden_dims, **kwargs)
        elif len(self.state_shape) == 3:
            self.cnn = CNNNetwork(state_shape, hidden_dims[0], conv_layers, **kwargs)
            self.mlp = MLPNetwork(hidden_dims[0], action_dim, hidden_dims, **kwargs)
        else:
            raise ValueError("state_shape must be 1D or 3D")

        print(self)

    def forward(self, s):
        if hasattr(self, "cnn"):
            s = torch.reshape(s, (-1, *self.state_shape))
            x = self.cnn(s)
            x = self.mlp(x)
        else:
            s = torch.reshape(s, (-1, np.prod(self.state_shape)))
            x = self.mlp(s)
        return x
    

class ContinuousQValueNetwork(torch.nn.Module):
    
    def __init__(self, state_shape:tuple, action_shape:tuple, hidden_dims:tuple=(128,), conv_layers:tuple=((32, 3),), **kwargs):
        super(ContinuousQValueNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape

        if len(self.state_shape) < 3:
            input_dim = np.prod(state_shape) + np.prod(action_shape)
            self.mlp = MLPNetwork(input_dim, 1, hidden_dims, **kwargs)
        elif len(self.state_shape) == 3:
            sqrt_dim = int(np.sqrt(hidden_dims[0] * np.prod(action_shape)))
            cnn_output_dim = max(10, min(128, sqrt_dim))
            self.cnn = CNNNetwork(state_shape, cnn_output_dim, conv_layers, **kwargs)
            self.mlp = MLPNetwork(cnn_output_dim+np.prod(action_shape), 1, hidden_dims, **kwargs)
        else:
            raise ValueError("state_shape must be 1D or 3D")

        print(self)


    def forward(self, s, a) -> torch.Tensor:
        if hasattr(self, "cnn"):
            s = torch.reshape(s, (-1, *self.state_shape))
            a = torch.reshape(a, (-1, np.prod(self.action_shape)))
            x = self.cnn(s)
            x = torch.cat((x, a), dim=1)
            x = self.mlp(x)
        else:
            s = torch.reshape(s, (-1, np.prod(self.state_shape)))
            a = torch.reshape(a, (-1, np.prod(self.action_shape)))
            x = torch.cat((s, a), dim=1)
            x = self.mlp(x)
        return x
    

class PolicyNetwork(torch.nn.Module):

    def __init__(self, state_shape:tuple, action_dim:int, hidden_dims:tuple=(128,), conv_layers:tuple=((32, 3),), **kwargs):
        super(PolicyNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim

        if len(self.state_shape) < 3:
            input_dim = np.prod(state_shape)
            self.mlp = MLPNetwork(input_dim, action_dim, hidden_dims, **kwargs)
        elif len(self.state_shape) == 3:
            self.cnn = CNNNetwork(state_shape, hidden_dims[0], conv_layers, **kwargs)
            self.mlp = MLPNetwork(hidden_dims[0], action_dim, hidden_dims, **kwargs)
        else:
            raise ValueError("state_shape must be 1D or 3D")
        self.softmax = torch.nn.Softmax(dim=1)

        print(self)

    def forward(self, s):
        if hasattr(self, "cnn"):
            x = torch.reshape(s, (-1, *self.state_shape))
            x = self.cnn(x)
            x = self.mlp(x)
        else:
            x = torch.reshape(s, (-1, np.prod(self.state_shape)))
            x = self.mlp(x)
        x = self.softmax(x)
        return x
    

class ContinuousPolicyNetwork(torch.nn.Module):

    def __init__(self, state_shape:tuple, action_shape:tuple, hidden_dims:tuple=(128,), conv_layers:tuple=((32, 3),), **kwargs):
        super(ContinuousPolicyNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape

        if len(self.state_shape) < 3:
            self.mlp = MLPNetwork(np.prod(state_shape), hidden_dims[-1], hidden_dims, **kwargs)
        elif len(self.state_shape) == 3:
            self.cnn = CNNNetwork(state_shape, hidden_dims[0], conv_layers, **kwargs)
            self.mlp = MLPNetwork(hidden_dims[0], hidden_dims[-1], hidden_dims, **kwargs)
        else:
            raise ValueError("state_shape must be 1D or 3D")
        
        self.mu = torch.nn.Linear(hidden_dims[-1], np.prod(self.action_shape))
        self.std = torch.nn.Linear(hidden_dims[-1], np.prod(self.action_shape))
        self.softplus = torch.nn.Softplus()

    def forward(self, s):
        if hasattr(self, "cnn"):
            x = torch.reshape(s, (-1, *self.state_shape))
            x = self.cnn(x)
            x = self.mlp(x)
        else:
            x = torch.reshape(s, (-1, np.prod(self.state_shape)))
            x = self.mlp(x)
        mu = self.mu(x)
        std = self.softplus(self.std(x))
        mu = torch.reshape(mu, (-1,) + self.action_shape)
        std = torch.reshape(std, (-1,) + self.action_shape)
        return mu, std
    

class DeterministicPolicyNetwork(torch.nn.Module):
    
    def __init__(self, state_shape:tuple, action_shape:tuple, action_bound:float|np.ndarray, hidden_dims:tuple=(128,), conv_layers:tuple=((32, 3),), **kwargs):
        super(DeterministicPolicyNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_bound = torch.tensor(action_bound)

        if len(self.state_shape) < 3:
            self.mlp = MLPNetwork(np.prod(state_shape), np.prod(action_shape), hidden_dims, **kwargs)
        elif len(self.state_shape) == 3:
            self.cnn = CNNNetwork(state_shape, hidden_dims[0], conv_layers, **kwargs)
            self.mlp = MLPNetwork(hidden_dims[0], np.prod(action_shape), hidden_dims, **kwargs)
        else:
            raise ValueError("state_shape must be 1D or 3D")
        self.tanh = torch.nn.Tanh()

        print(self)

    def forward(self, s) -> torch.Tensor:
        if hasattr(self, "cnn"):
            x = torch.reshape(s, (-1, *self.state_shape))
            x = self.cnn(x)
            x = self.mlp(x)
        else:
            x = torch.reshape(s, (-1, np.prod(self.state_shape)))
            x = self.mlp(x)
        x = self.tanh(x)
        x = torch.reshape(x, (-1, *self.action_shape))
        return x * self.action_bound


class DiscreteDeterministicPolicyNetwork(torch.nn.Module):
    """用于离散动作DDPG算法的策略网络，输出gumbel-softmax分布的logits"""
    
    def __init__(self, state_shape:tuple, action_shape:tuple, hidden_dims:tuple=(128,), conv_layers:tuple=((32, 3),), **kwargs):
        super(DiscreteDeterministicPolicyNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape

        if len(self.state_shape) < 3:
            self.mlp = MLPNetwork(np.prod(state_shape), np.prod(action_shape), hidden_dims, **kwargs)
        elif len(self.state_shape) == 3:
            self.cnn = CNNNetwork(state_shape, hidden_dims[0], conv_layers, **kwargs)
            self.mlp = MLPNetwork(hidden_dims[0], np.prod(action_shape), hidden_dims, **kwargs)
        else:
            raise ValueError("state_shape must be 1D or 3D")

        print(self)

    def forward(self, s) -> torch.Tensor:
        if hasattr(self, "cnn"):
            x = torch.reshape(s, (-1, *self.state_shape))
            x = self.cnn(x)
            x = self.mlp(x)
        else:
            x = torch.reshape(s, (-1, np.prod(self.state_shape)))
            x = self.mlp(x)
        x = torch.reshape(x, (-1, *self.action_shape))
        return x


