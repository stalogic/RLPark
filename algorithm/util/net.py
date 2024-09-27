import torch
import numpy as np


class MLPNetwork(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple,
        activation=torch.nn.ReLU(),
        batch_norm: bool = True,
        dropout: float = None,
        input_norm: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        if dropout and not (0.0 < dropout < 1.0):
            raise ValueError("dropout must be between 0 and 1")

        self.mlp = torch.nn.ModuleList()
        if input_norm:
            self.mlp.append(torch.nn.BatchNorm1d(input_dim))
        in_dim = input_dim
        if hidden_dims:
            if isinstance(hidden_dims, int):
                hidden_dims = (hidden_dims,)
            for out_dim in hidden_dims:
                self.mlp.append(torch.nn.Linear(in_dim, out_dim))
                if batch_norm:
                    self.mlp.append(torch.nn.BatchNorm1d(out_dim))
                self.mlp.append(activation)
                if dropout:
                    self.mlp.append(torch.nn.Dropout(p=dropout))
                in_dim = out_dim
        self.mlp.append(torch.nn.Linear(in_dim, output_dim))

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class CNNNetwork(torch.nn.Module):

    def __init__(
        self,
        input_shape: tuple,
        output_dim: int,
        conv_layers: tuple,
        activation=torch.nn.ReLU(),
        batch_norm: bool = True,
        dropout: float = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        if dropout and not (0.0 < dropout < 1.0):
            raise ValueError("dropout must be between 0 and 1")

        self.input_shape = input_shape

        self.convs = torch.nn.ModuleList()
        in_channels = input_shape[0]
        for out_channels, kernel_size in conv_layers:
            self.convs.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size))
            if batch_norm:
                self.convs.append(torch.nn.BatchNorm2d(out_channels))
            self.convs.append(activation)
            if dropout:
                self.convs.append(torch.nn.Dropout2d(p=dropout))
            self.convs.append(torch.nn.MaxPool2d(kernel_size))
            in_channels = out_channels
        self.convs.append(torch.nn.Flatten())

        fc_input_dim = self._calc_conv_flatten_dim()
        self.fc = torch.nn.Linear(fc_input_dim, output_dim)

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = self.fc(x)
        return x

    def _calc_conv_flatten_dim(
        self,
    ) -> int:
        x = torch.zeros(1, *self.input_shape)
        for layer in self.convs:
            x = layer(x)
        return x.shape[1]
