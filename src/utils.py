import csv
import json
import itertools
from typing import List, Literal, Union, Dict

import torch
from torch import nn


# Monotonicity Check
def monotonicity_check(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_x: torch.Tensor,
    monotonic_indices: List[int],
    device: torch.device
) -> float:
    """
    Compute fraction of monotonicity violations via input gradients.

    A violation occurs if ∂f/∂x_i < 0 for any monotonic feature i.
    """

    if not monotonic_indices:
        return 0.0

    model.eval()

    data_x = data_x.to(device)
    n_points = data_x.shape[0]

    # Create mask
    monotonic_mask = torch.zeros(
        data_x.shape[1], dtype=torch.bool, device=device
    )
    monotonic_mask[monotonic_indices] = True

    data_monotonic = data_x[:, monotonic_mask]
    data_non_monotonic = data_x[:, ~monotonic_mask]

    data_monotonic.requires_grad_(True)

    optimizer.zero_grad()

    outputs = model(torch.cat([data_monotonic, data_non_monotonic], dim=1))
    loss = torch.sum(outputs)
    loss.backward()

    grads = data_monotonic.grad

    if grads is None:
        return 0.0

    # Minimum gradient among monotonic features per sample
    min_grad = grads.min(dim=1)[0]

    violations = (min_grad < -1e-8).sum().item()

    return violations / n_points


# Reordered Monotonic Indices
def get_reordered_monotonic_indices(dataset_name: str) -> List[int]:
    dataset_name = dataset_name.replace("load_", "")

    monotonic_feature_counts = {
        'abalone': 4,
        'auto_mpg': 7,
        'boston_housing': 2,
        'compas': 4,
        'era': 4,
        'esl': 4,
        'heart': 2,
        'lev': 4,
        'swd': 7
    }

    num = monotonic_feature_counts.get(dataset_name, 0)
    return list(range(num))


# Monotonicity Indicator
def create_monotonicity_indicator(
    monotonic_indices: List[int],
    input_size: int
) -> List[int]:

    indicator = [0] * input_size

    for idx in monotonic_indices:
        if 0 <= idx < input_size:
            indicator[idx] = 1

    return indicator


# Weight Initialization
def init_weights(
    module_or_tensor: Union[nn.Module, torch.Tensor],
    method: Literal[
        'xavier_uniform',
        'xavier_normal',
        'kaiming_uniform',
        'kaiming_normal',
        'he_uniform',
        'he_normal',
        'truncated_normal',
        'uniform',
        'zeros'
    ],
    **kwargs
) -> None:

    def init_tensor(tensor):

        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(tensor)

        elif method == 'xavier_normal':
            nn.init.xavier_normal_(tensor)

        elif method in ['kaiming_uniform', 'he_uniform']:
            nn.init.kaiming_uniform_(tensor)

        elif method in ['kaiming_normal', 'he_normal']:
            nn.init.kaiming_normal_(tensor)

        elif method == 'truncated_normal':
            mean = kwargs.get('mean', 0.)
            std = kwargs.get('std', 1.)

            with torch.no_grad():
                tensor.normal_(mean, std)

                while True:
                    cond = (
                        (tensor < mean - 2 * std) |
                        (tensor > mean + 2 * std)
                    )
                    if not torch.sum(cond):
                        break
                    tensor[cond] = tensor[cond].normal_(mean, std)

        elif method == 'uniform':
            a = kwargs.get('a', 0.)
            b = kwargs.get('b', 1.)
            nn.init.uniform_(tensor, a=a, b=b)

        elif method == 'zeros':
            nn.init.zeros_(tensor)

        else:
            raise ValueError(f"Unsupported initialization method: {method}")

    if isinstance(module_or_tensor, nn.Module):
        for param in module_or_tensor.parameters():
            init_tensor(param)

    elif isinstance(module_or_tensor, torch.Tensor):
        init_tensor(module_or_tensor)

    else:
        raise TypeError("Input must be nn.Module or torch.Tensor")


# Positive Weight Transform
def transform_weights(
    module_or_tensor: Union[nn.Module, torch.Tensor],
    method: Literal['exp', 'explin', 'sqr']
):

    def transform_tensor(tensor):

        if method == 'exp':
            return torch.exp(tensor)

        elif method == 'explin':
            return torch.where(
                tensor > 1.,
                tensor,
                torch.exp(tensor - 1.)
            )

        elif method == 'sqr':
            return torch.square(tensor)

        else:
            raise ValueError(f"Unsupported transform method: {method}")

    if isinstance(module_or_tensor, nn.Module):
        return nn.ParameterList([
            nn.Parameter(transform_tensor(param))
            for param in module_or_tensor.parameters()
        ])

    elif isinstance(module_or_tensor, torch.Tensor):
        return transform_tensor(module_or_tensor)

    else:
        raise TypeError("Input must be nn.Module or torch.Tensor")


# CSV
def write_results_to_csv(
    filename: str,
    dataset_name: str,
    task_type: str,
    metric_name: str,
    metric_mean: float,
    metric_std: float,
    n_params: int,
    best_config: Dict,
    mono_metrics: Dict
):

    best_config_str = json.dumps(best_config)


    m_mean = f"{metric_mean:.4f}" if isinstance(metric_mean, (int, float)) else metric_mean
    m_std = f"{metric_std:.4f}" if isinstance(metric_std, (int, float)) else metric_std


    row = [
        dataset_name,
        task_type,
        metric_name,
        m_mean,
        m_std,
        n_params,
        best_config_str
    ]


    for key in ['random', 'train', 'val']:
        mean, std = mono_metrics.get(key, (0.0, 0.0))
        row.extend([f"{mean:.4f}", f"{std:.4f}"])

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


# Parameter Counter
def count_parameters(module: nn.Module) -> int:
    return sum(
        p.numel() for p in module.parameters()
        if p.requires_grad
    )


# Layer Combination Generator
def generate_layer_combinations(
    min_layers=1,
    max_layers=3,
    units=[8, 16, 32, 64]
):

    combinations = []

    for n_layers in range(min_layers, max_layers + 1):
        for combo in itertools.product(units, repeat=n_layers):
            combinations.append(list(combo))

    return [str(combo) for combo in combinations]