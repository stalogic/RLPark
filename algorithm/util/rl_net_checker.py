import torch
import torch.nn as nn
import gymnasium.spaces as spaces

class RLNetworkChecker(object):
    @staticmethod
    def get_output_layer(model:nn.Module):
        for module in model.children():
            pass
        else:
            return module
    
    @staticmethod
    def check(model:nn.Module, state_spec:spaces.Box, action_spec:spaces.Box|spaces.Discrete):
        raise NotImplementedError

class QsNetworkChecker(RLNetworkChecker):

    @staticmethod
    def check(model: nn.Module, state_spec: spaces.Box, action_spec: spaces.Box | spaces.Discrete):
        assert isinstance(model, nn.Module)
        assert isinstance(state_spec, spaces.Box)
        assert isinstance(action_spec, spaces.Discrete)

        output_layer = RLNetworkChecker.get_output_layer(model)
        if not isinstance(output_layer, nn.Linear):
            import warnings
            warnings.warn(f"The output layer of the Q-network should be Linear in most cases, {type(output_layer).__name__} founded.")
        
        try:
            param = next(model.parameters())
            device = param.device
            dtype = param.dtype
            shape = state_spec.shape

            state = torch.randn(shape, dtype=dtype, device=device).unsqueeze(0)
            out = model(state).squeeze()
        except Exception as err:
            assert False, f"Model call fail:\n{err}"
        assert out.shape[-1] == action_spec.n, f"Q(s) network output not match action num. \nmodel out shape: {out.shape}, action num: {action_spec.n}"


class QsaNetworkChecker(RLNetworkChecker):

    @staticmethod
    def check(model: nn.Module, state_spec: spaces.Box, action_spec: spaces.Box | spaces.Discrete):
        assert isinstance(model, nn.Module)
        assert isinstance(state_spec, spaces.Box)
        assert isinstance(action_spec, spaces.Box)

        output_layer = RLNetworkChecker.get_output_layer(model)
        if not isinstance(output_layer, nn.Linear):
            import warnings
            warnings.warn(f"The output layer of the V-network should be Linear in most cases, {type(output_layer).__name__} founded.")
        
        try:
            param = next(model.parameters())
            device = param.device
            dtype = param.dtype

            state = torch.randn(state_spec.shape, dtype=dtype, device=device).unsqueeze(0)
            action = torch.randn(action_spec.shape, dtype=dtype, device=device).unsqueeze(0)
            out = model(state, action).squeeze()
        except Exception as err:
            assert False, f"Model call fail:\n{err}"
        assert len(out.shape) == 0, f"V(s) network output not equals 1. \nV network out shape: {out.shape}"


class VsNetworkChecker(RLNetworkChecker):

    @staticmethod
    def check(model: nn.Module, state_spec: spaces.Box, action_spec: spaces.Box | spaces.Discrete):
        assert isinstance(model, nn.Module)
        assert isinstance(state_spec, spaces.Box)

        output_layer = RLNetworkChecker.get_output_layer(model)
        if not isinstance(output_layer, nn.Linear):
            import warnings
            warnings.warn(f"The output layer of the V-network should be Linear in most cases, {type(output_layer).__name__} founded.")
        
        try:
            param = next(model.parameters())
            device = param.device
            dtype = param.dtype
            shape = state_spec.shape

            state = torch.randn(shape, dtype=dtype, device=device).unsqueeze(0)
            out = model(state).squeeze()
        except Exception as err:
            assert False, f"Model call fail:\n{err}"
        assert len(out.shape) == 0, f"Q(s,a) network output not equals 1. \nV network out shape: {out.shape}"


class PsNetworkChecker(RLNetworkChecker):

    @staticmethod
    def check(model: nn.Module, state_spec: spaces.Box, action_spec: spaces.Box | spaces.Discrete):
        assert isinstance(model, nn.Module)
        assert isinstance(state_spec, spaces.Box)
        assert isinstance(action_spec, spaces.Discrete)

        output_layer = RLNetworkChecker.get_output_layer(model)
        if not isinstance(output_layer, nn.Softmax):
            import warnings
            warnings.warn(f"The output layer of the Pi(s) network should be Linear in most cases, {type(output_layer).__name__} founded.")
        
        try:
            param = next(model.parameters())
            device = param.device
            dtype = param.dtype
            shape = state_spec.shape

            state = torch.randn(shape, dtype=dtype, device=device).unsqueeze(0)
            out = model(state).squeeze()
        except Exception as err:
            assert False, f"Model call fail:\n{err}"
        assert out.shape[0] == action_spec.n, f"Pi(s) network output not match action num. \nPi(s) network out shape: {out.shape}, action num: {action_spec.n}"


class NsNetworkChecker(RLNetworkChecker):

    @staticmethod
    def check(model: nn.Module, state_spec: spaces.Box, action_spec: spaces.Box | spaces.Discrete):
        assert isinstance(model, nn.Module)
        assert isinstance(state_spec, spaces.Box)
        assert isinstance(action_spec, spaces.Box)

        output_layer = RLNetworkChecker.get_output_layer(model)
        if not isinstance(output_layer, nn.Linear):
            import warnings
            warnings.warn(f"The output layer of the Nu(s) network should be Linear in most cases, {type(output_layer).__name__} founded.")
        
        try:
            param = next(model.parameters())
            device = param.device
            dtype = param.dtype
            shape = state_spec.shape

            state = torch.randn(shape, dtype=dtype, device=device).unsqueeze(0)
            mu, std = model(state)
        except Exception as err:
            assert False, f"Model call fail:\n{err}"
        
        shape_len = len(action_spec.shape)
        assert mu.shape[-shape_len:] == std.shape[-shape_len:] == action_spec.shape, \
            f"Nu(s) network output not match action num. \nNu(s) network mu shape: {mu.shape}, std shape: {std.shape}, action shape: {action_spec.shape}"


class MsNetworkChecker(RLNetworkChecker):
    @staticmethod
    def check(model: nn.Module, state_spec: spaces.Box, action_spec: spaces.Box | spaces.Discrete):
        assert isinstance(model, nn.Module)
        assert isinstance(state_spec, spaces.Box)
        assert isinstance(action_spec, spaces.Box)

        output_layer = RLNetworkChecker.get_output_layer(model)
        if not isinstance(output_layer, nn.Linear):
            import warnings
            warnings.warn(f"The output layer of the Nu(s) network should be Linear in most cases, {type(output_layer).__name__} founded.")
        
        try:
            param = next(model.parameters())
            device = param.device
            dtype = param.dtype
            shape = state_spec.shape

            state = torch.randn(shape, dtype=dtype, device=device).unsqueeze(0)
            out = model(state)
        except Exception as err:
            assert False, f"Model call fail:\n{err}"
        
        shape_len = len(action_spec.shape)
        assert out.shape[-shape_len:] == action_spec.shape, \
            f"Nu(s) network output not match action num. \nNu(s) network out shape: {out.shape}, action shape: {action_spec.shape}"

if __name__ == "__main__":
    state_spec = spaces.Box(low=-1, high=1, shape=(6,), dtype=float)
    action_spec = spaces.Discrete(10)
    q_net = nn.Sequential(
        nn.Linear(6, 128),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(128, 10),
        nn.Softmax(1)
    )
    QsNetworkChecker.check(model=q_net, state_spec=state_spec, action_spec=action_spec)

    state_spec = spaces.Box(low=-1, high=1, shape=(6,), dtype=float)
    action_spec = spaces.Discrete(10)
    v_net = nn.Sequential(
        nn.Linear(6, 128),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(128, 1),
        nn.Softplus()
    )
    VsNetworkChecker.check(model=v_net, state_spec=state_spec, action_spec=action_spec)

    state_spec = spaces.Box(low=-1, high=1, shape=(6,), dtype=float)
    action_spec = spaces.Box(low=-10, high=0, shape=(20,), dtype=float)
    class QSA(nn.Module): 
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            
            self.state_layers = nn.Sequential(
                nn.Linear(6, 128),
                nn.ReLU(),
                nn.Dropout(),
            )
            self.action_layers = nn.Sequential(
                nn.Linear(20, 128),
                nn.ReLU(),
                nn.Dropout(),
            )
            self.out_layers = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(32, 1),
                nn.Softplus()
            )
        
        def forward(self, state, action):
            state = self.state_layers(state)
            action = self.action_layers(action)
            return self.out_layers(state + action)
    qsa_net = QSA()
    QsaNetworkChecker.check(model=qsa_net, state_spec=state_spec, action_spec=action_spec)

    state_spec = spaces.Box(low=-1, high=1, shape=(6, 4, 4), dtype=float)
    action_spec = spaces.Discrete(10)
    pi_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(6 * 4 * 4, 128),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(128, 10),
        nn.Softmax(1)
    )
    QsNetworkChecker.check(model=pi_net, state_spec=state_spec, action_spec=action_spec)

    state_spec = spaces.Box(low=-1, high=1, shape=(6, 4, 4), dtype=float)
    action_spec = spaces.Box(low=-1, high=1, shape=(2,3), dtype=float)
    class Nu(nn.Module):
        def __init__(self):
            super().__init__()
            self.mid_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(6*4*4, 128),
                nn.ReLU(),
                nn.Dropout()
            )
            self.mu = nn.Linear(128, 6)
            self.std = nn.Linear(128, 6)

        def forward(self, state):
            x = self.mid_layers(state)
            mu = self.mu(x)
            std = self.std(x)
            mu = torch.reshape(mu, (2,3))
            std = torch.reshape(std, (2,3))
            return mu, std
        
    nu_net = Nu()
    NsNetworkChecker.check(nu_net, state_spec=state_spec, action_spec=action_spec)
    
    state_spec = spaces.Box(low=-1, high=1, shape=(6, 4, 4), dtype=float)
    action_spec = spaces.Box(low=-1, high=1, shape=(2,3), dtype=float)
    
    class Ms(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(6*4*4, 128),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(128, 6)
            )

        def forward(self, state):
            out = self.layers(state)
            act = torch.reshape(out, (2, 3))
            return act
    
    ms_net = Ms()
    MsNetworkChecker.check(model=ms_net, state_spec=state_spec, action_spec=action_spec)