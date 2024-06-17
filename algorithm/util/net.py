import torch


class MLPNetwork(torch.nn.Module):
    
    def __init__(self, input_dim:int, output_dim:int, layers:tuple, activation=torch.nn.ReLU(), batch_norm:bool=True, dropout:float=None, input_norm:bool=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if dropout and not (0.0 < dropout < 1.0):
            raise ValueError("dropout must be between 0 and 1")

        

        self.mlp = torch.nn.ModuleList()
        if input_norm:
            self.mlp.append(torch.nn.BatchNorm1d(input_dim))
        in_dim = input_dim
        for out_dim in layers:
            self.mlp.append(torch.nn.Linear(in_dim, out_dim))
            if batch_norm:
                self.mlp.append(torch.nn.BatchNorm1d(out_dim))
            self.mlp.append(activation)
            if dropout:
                self.mlp.append(torch.nn.Dropout(p=dropout))
            in_dim = out_dim
        self.mlp.append(torch.nn.Linear(in_dim, output_dim))

        print(self)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class CNNNetwork(torch.nn.Module):

    def __init__(self,input_channels:int, output_dim:int, conv_layers:tuple, activate_fn=torch.nn.ReLU(), batch_norm:bool=True, dorpout:float=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv_layers = torch.nn.ModuleList()
        self.pool_layers = torch.nn.ModuleList()

        for i, (out_channels, kernel_size) in enumerate(conv_layers):
            in_channels = input_channels if i == 0 else conv_layers[i-1][0]
            self.conv_layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size))
            if batch_norm:
                self.conv_layers.append(torch.nn.BatchNorm2d(out_channels))
            self.conv_layers.append(torch.nn.ReLU())

    def forward(self, x):
        pass