# UMNNModel.py

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Positive Integrand Network g(x) >= 0
# =========================================================
class PositiveIntegrand(nn.Module):
    """
    1D integrand network producing non-negative outputs via softplus,
    guaranteeing monotonicity after integration.
    """
    def __init__(self, hidden_sizes: List[int], activation: str = "tanh"):
        super().__init__()

        act = nn.Tanh() if activation == "tanh" else nn.ReLU()

        layers = []
        in_dim = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act)
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # softplus ensures >= 0
        return F.softplus(self.net(x))


# =========================================================
# 1D UMNN block: f(x) = ∫_0^x g(t) dt (numerical integration)
# =========================================================
class UMNN1D(nn.Module):
    """
    Numerically integrates a non-negative integrand to obtain a monotone function.
    Uses simple Riemann (midpoint-like) approximation on [0, x].
    Stable, differentiable, GPU-friendly, no custom CUDA ops.
    """

    def __init__(self, hidden_sizes: List[int], n_steps: int = 32, activation: str = "tanh"):
        super().__init__()
        self.integrand = PositiveIntegrand(hidden_sizes, activation=activation)
        self.n_steps = int(n_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1)
        returns: (B, 1)
        """
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(f"UMNN1D expects x shape (B,1), got {tuple(x.shape)}")

        device = x.device
        B = x.shape[0]

        # t in [0,1], then scale to [0,x]
        # shape: (1, S, 1)
        t = torch.linspace(0.0, 1.0, self.n_steps, device=device).view(1, self.n_steps, 1)

        x_exp = x.view(B, 1, 1)                        # (B,1,1)
        points = t * x_exp                              # (B,S,1) in [0,x]
        g_vals = self.integrand(points)                 # (B,S,1) >= 0

        # integral approx: mean(g(points)) * x
        integral = g_vals.mean(dim=1) * x               # (B,1)
        return integral


# =========================================================
# Full UMNN model for mixed monotone + non-monotone features
# =========================================================
class UMNNModel(nn.Module):
    """
    Assumption (matches your loaders):
    - monotonic features are reordered to the front: x[:, :k] are monotone
    - non-monotone features follow: x[:, k:]
    - monotonic_indices is usually [0..k-1] (from get_reordered_monotonic_indices)

    Forward:
    y = sum_i w_i * f_i(x_mono_i) + MLP_nonmono(x_nonmono)
    where each f_i is monotone in its input (via integration of positive integrand)
    """

    def __init__(
        self,
        input_size: int,
        monotonic_indices: List[int],
        mono_hidden_sizes: List[int],
        nonmono_hidden_sizes: List[int],
        n_integration_steps: int = 32,
        mono_activation: str = "tanh",
        nonmono_activation: str = "relu",
        output_size: int = 1,
    ):
        super().__init__()

        self.input_size = int(input_size)
        self.monotonic_indices = list(monotonic_indices)
        self.k = len(self.monotonic_indices)

        if self.k <= 0:
            raise ValueError("[UMNNModel] monotonic_indices is empty. UMNN requires at least 1 monotonic feature.")

        if self.k > self.input_size:
            raise ValueError(f"[UMNNModel] k={self.k} > input_size={self.input_size}.")

        # Monotone 1D blocks
        self.mono_blocks = nn.ModuleList([
            UMNN1D(mono_hidden_sizes, n_steps=n_integration_steps, activation=mono_activation)
            for _ in range(self.k)
        ])

        # Linear combiner for monotone outputs
        self.mono_linear = nn.Linear(self.k, output_size, bias=False)

        # Non-monotone MLP (can be empty if no non-mono features)
        nonmono_dim = self.input_size - self.k
        act_nm = nn.ReLU() if nonmono_activation == "relu" else nn.Tanh()

        if nonmono_dim <= 0:
            self.nonmono_net = None
        else:
            layers = []
            in_dim = nonmono_dim
            for h in nonmono_hidden_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(act_nm)
                in_dim = h
            layers.append(nn.Linear(in_dim, output_size))
            self.nonmono_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"[UMNNModel] expected x shape (B,{self.input_size}), got {tuple(x.shape)}")

        mono_part = x[:, :self.k]            # (B,k)
        nonmono_part = x[:, self.k:]         # (B,d-k)

        # each mono dim -> 1D UMNN
        mono_outs = []
        for i in range(self.k):
            xi = mono_part[:, i:i+1]
            mono_outs.append(self.mono_blocks[i](xi))   # (B,1)

        mono_stack = torch.cat(mono_outs, dim=1)         # (B,k)
        mono_term = self.mono_linear(mono_stack)         # (B,1)

        if self.nonmono_net is None:
            return mono_term

        nonmono_term = self.nonmono_net(nonmono_part)    # (B,1)
        return mono_term + nonmono_term