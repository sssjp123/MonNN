import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable, Literal, List
from contextlib import contextmanager
from functools import lru_cache
from src.utils import init_weights


# =====================================================
# MonoDense
# =====================================================

class MonoDense(nn.Module):

    def __init__(
            self,
            in_features: int,
            units: int,
            activation: Optional[Union[str, Callable]] = None,
            monotonicity_indicator: Union[int, list, torch.Tensor] = 1,
            is_convex: bool = False,
            is_concave: bool = False,
            activation_weights: Tuple[float, float, float] = (7.0, 7.0, 2.0),
            init_method: Literal[
                'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
                'kaiming_normal', 'he_uniform', 'he_normal',
                'truncated_normal'] = 'xavier_uniform'
    ):
        super().__init__()

        self.in_features = in_features
        self.units = units
        self.org_activation = activation
        self.is_convex = is_convex
        self.is_concave = is_concave
        self.init_method = init_method

        self.activation_weights = nn.Parameter(
            torch.tensor(activation_weights, dtype=torch.float32)
        )

        indicator = self.get_monotonicity_indicator(
            monotonicity_indicator,
            in_features,
            units
        )

        # 🔥 用 buffer，而不是 Parameter
        self.register_buffer("monotonicity_indicator", indicator)

        self.weight = nn.Parameter(torch.empty(units, in_features))
        self.bias = nn.Parameter(torch.empty(units))

        self.reset_parameters()

        self.convex_activation, self.concave_activation, self.saturated_activation = \
            self.get_activation_functions(self.org_activation)

    def reset_parameters(self):
        init_weights(self.weight, method=self.init_method)
        init_weights(self.bias, method='zeros')

    # =====================================================
    # Monotonicity indicator processing
    # =====================================================

    @staticmethod
    def get_monotonicity_indicator(monotonicity_indicator, in_features, units):

        if isinstance(monotonicity_indicator, torch.Tensor):
            indicator = monotonicity_indicator.clone().detach().float()
        else:
            indicator = torch.tensor(monotonicity_indicator, dtype=torch.float32)

        if indicator.dim() < 2:
            indicator = indicator.reshape(-1, 1)

        indicator = indicator.expand(in_features, units).t()

        if not torch.all(
                (indicator == -1) |
                (indicator == 0) |
                (indicator == 1)):
            raise ValueError("monotonicity_indicator must be -1, 0, or 1")

        return indicator

    # =====================================================
    # Activation
    # =====================================================

    @staticmethod
    @lru_cache(None)
    def get_activation_functions(activation):

        if callable(activation):
            return activation, lambda x: -activation(-x), \
                   MonoDense.get_saturated_activation(
                       activation, lambda x: -activation(-x)
                   )

        if isinstance(activation, str):
            activation = activation.lower()

        activations = {
            'relu': F.relu,
            'elu': F.elu,
            'selu': F.selu,
            'gelu': F.gelu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            None: lambda x: x,
            'linear': lambda x: x
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        convex = activations[activation]
        concave = lambda x: -convex(-x)
        saturated = MonoDense.get_saturated_activation(convex, concave)

        return convex, concave, saturated

    @staticmethod
    def get_saturated_activation(convex_activation, concave_activation):

        def saturated(x):
            cc = convex_activation(torch.ones_like(x))
            return torch.where(
                x <= 0,
                convex_activation(x + 1.0) - cc,
                concave_activation(x - 1.0) + cc,
            )

        return saturated

    # =====================================================
    # Forward
    # =====================================================

    def apply_monotonicity_indicator_to_kernel(self, kernel):
        abs_kernel = torch.abs(kernel)
        kernel = torch.where(self.monotonicity_indicator == 1, abs_kernel, kernel)
        kernel = torch.where(self.monotonicity_indicator == -1, -abs_kernel, kernel)
        return kernel

    def forward(self, x):
        modified_weight = self.apply_monotonicity_indicator_to_kernel(self.weight)
        h = F.linear(x, modified_weight, self.bias)

        if self.org_activation is None:
            return h

        return self.convex_activation(h)


# =====================================================
# ConstrainedMonotonicNeuralNetwork
# =====================================================

class ConstrainedMonotonicNeuralNetwork(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 device: torch.device,
                 activation: str = 'elu',
                 monotonicity_indicator: List[int] = None,
                 final_activation: Optional[Callable] = None,
                 init_method: str = 'xavier_uniform',
                 architecture_type: str = 'type1'):

        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.final_activation = final_activation
        self.architecture_type = architecture_type

        indicator = torch.tensor(
            monotonicity_indicator,
            dtype=torch.float32
        )

        if len(indicator) != input_size:
            raise ValueError(
                f"Length of monotonicity_indicator ({len(indicator)}) "
                f"must match input_size ({input_size})"
            )

        # 🔥 用 buffer
        self.register_buffer("monotonicity_indicator", indicator)

        if architecture_type == 'type1':
            self.network = self._build_type1()
        else:
            self.network = self._build_type2()

        self.init_weights(init_method)
        self.to(device)

    # =====================================================
    # Build
    # =====================================================

    def _build_type1(self):

        layers = nn.ModuleList()

        layers.append(MonoDense(
            in_features=self.input_size,
            units=self.hidden_sizes[0],
            activation=self.activation,
            monotonicity_indicator=self.monotonicity_indicator
        ))

        for i in range(1, len(self.hidden_sizes)):
            layers.append(MonoDense(
                in_features=self.hidden_sizes[i - 1],
                units=self.hidden_sizes[i],
                activation=self.activation,
                monotonicity_indicator=1
            ))

        layers.append(MonoDense(
            in_features=self.hidden_sizes[-1],
            units=self.output_size,
            activation=None,
            monotonicity_indicator=1
        ))

        return layers

    def _build_type2(self):
        return self._build_type1()

    # =====================================================
    # Init
    # =====================================================

    def init_weights(self, method):
        for module in self.modules():
            if isinstance(module, MonoDense):
                init_weights(module.weight, method=method)
                init_weights(module.bias, method='zeros')

    # =====================================================
    # Forward
    # =====================================================

    def forward(self, x):
        for layer in self.network:
            x = layer(x)

        if self.final_activation:
            x = self.final_activation(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)