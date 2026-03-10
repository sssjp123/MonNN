# exp_common.py

import random
from typing import Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, accuracy_score


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError(f"Binary classification only, got labels: {uniq}")
    if set(uniq.tolist()) == {0, 1}:
        return y.astype(np.float32)
    return (y == uniq[1]).astype(np.float32)


def fold_minmax_scale_X(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    span = x_max - x_min
    span[span == 0] = 1.0
    return (X_train - x_min) / span, (X_val - x_min) / span


def fold_standardize_y(
    y_train: np.ndarray,
    y_val: np.ndarray,
    task_type: str
):
    """
    Regression: z-score based on TRAIN split only.
    Classification: return unchanged, mean/std = None.
    """
    if task_type != "regression":
        return y_train, y_val, None, None

    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)

    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if std == 0:
        std = 1.0

    return (y_train - mean) / std, (y_val - mean) / std, mean, std


@torch.no_grad()
def eval_for_early_stop(
    model,
    loader,
    task_type: str,
    device: torch.device
) -> float:
    """
    Scalar metric for early stopping / Optuna objective.
    Regression: RMSE on standardized y (because y in loader is standardized).
    Classification: error rate.
    """
    model.eval()
    preds, trues = [], []

    for X, y in loader:
        X = X.to(device)
        out = model(X).detach().cpu().numpy().reshape(-1)
        y_np = y.detach().cpu().numpy().reshape(-1)
        preds.append(out)
        trues.append(y_np)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    if task_type == "regression":
        return float(np.sqrt(mean_squared_error(trues, preds)))

    prob = 1.0 / (1.0 + np.exp(-np.clip(preds, -50, 50)))
    y_pred = (prob > 0.5).astype(np.int64)
    return float(1.0 - accuracy_score(trues.astype(np.int64), y_pred))


@torch.no_grad()
def eval_regression_raw_metrics(
    model,
    loader,
    device: torch.device,
    y_mean: float,
    y_std: float
) -> Tuple[float, float]:
    """
    Regression final reporting:
    inverse-transform standardized predictions back to RAW y scale,
    compute RMSE_raw and NRMSE.
    """
    model.eval()
    preds, trues = [], []

    for X, y in loader:
        X = X.to(device)
        out = model(X).detach().cpu().numpy().reshape(-1)
        y_np = y.detach().cpu().numpy().reshape(-1)
        preds.append(out)
        trues.append(y_np)

    preds_std = np.concatenate(preds, axis=0)
    trues_std = np.concatenate(trues, axis=0)

    preds_raw = preds_std * y_std + y_mean
    trues_raw = trues_std * y_std + y_mean

    rmse = float(np.sqrt(mean_squared_error(trues_raw, preds_raw)))
    denom = float(np.max(trues_raw) - np.min(trues_raw) + 1e-8)
    nrmse = float(rmse / denom)
    return rmse, nrmse