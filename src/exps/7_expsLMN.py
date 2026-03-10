# exps_LMN.py

import ast
import csv
import copy
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import optuna
from schedulefree import AdamWScheduleFree

from src.LipschitzMonotonicNeuralNetworks import LMNNetwork
from dataPreprocessing.loaders import (
    load_abalone, load_auto_mpg,
    load_boston_housing, load_compas,
    load_era, load_esl, load_heart,
    load_lev, load_swd
)

from src.utils import (
    monotonicity_check,
    get_reordered_monotonic_indices,
    write_results_to_csv,
    count_parameters,
    generate_layer_combinations
)

from src.exp_common import (
    set_global_seed,
    ensure_binary_labels,
    fold_minmax_scale_X,
    fold_standardize_y,
    eval_for_early_stop,
    eval_regression_raw_metrics,
)

GLOBAL_SEED = 42
SEARCH_EPOCHS = 20
FINAL_EPOCHS = 100

N_TRIALS = 20
N_SPLITS = 5
MAX_MONO_POINTS = 1000



# Task type
def get_task_type(loader: Callable) -> str:
    regression_tasks = [
        load_abalone, load_auto_mpg,
        load_boston_housing, load_era,
        load_esl, load_lev, load_swd
    ]
    return "regression" if loader in regression_tasks else "classification"



# Dataset builder
def build_tensor_dataset(X: np.ndarray, y: np.ndarray, task_type: str) -> TensorDataset:
    if task_type == "classification":
        y = ensure_binary_labels(y)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
    return TensorDataset(X_t, y_t)



# Random sampling in real data domain
def sample_random_in_domain(X_ref: np.ndarray, n_points: int, seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    X_ref = np.asarray(X_ref)

    x_min = np.nanmin(X_ref, axis=0)
    x_max = np.nanmax(X_ref, axis=0)
    span = x_max - x_min
    span[span == 0] = 1.0

    u = rng.rand(n_points, X_ref.shape[1])
    X_rand = x_min + u * span
    return torch.FloatTensor(X_rand).to(device)



# Safe monotonicity wrapper
def safe_monotonicity_check(
    model: nn.Module,
    optimizer,
    data_tensor: torch.Tensor,
    monotonic_indices,
    device: torch.device
) -> float:
    model_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    opt_state = copy.deepcopy(optimizer.state_dict())

    try:
        score = monotonicity_check(model, optimizer, data_tensor, monotonic_indices, device)
    finally:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(opt_state)

        # re-place optimizer tensors on device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    return float(score)



# Model
def create_model(config: Dict, input_size: int, monotonic_indices, seed: int) -> nn.Module:
    torch.manual_seed(seed)

    hidden_sizes = config["hidden_sizes"]
    if isinstance(hidden_sizes, str):
        hidden_sizes = ast.literal_eval(hidden_sizes)

    # monotone_constraints: 1 for monotonic features, 0 otherwise
    monotone_constraints = [0] * input_size
    for idx in monotonic_indices:
        if 0 <= idx < input_size:
            monotone_constraints[idx] = 1


    return LMNNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=1,
        monotone_constraints=monotone_constraints,
        output_activation=nn.Identity(),
        lipschitz_constant=1.0
    )



# Training
def get_criterion(task_type: str):
    return nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()


def train_model(model, optimizer, train_loader, val_loader, config: Dict, task_type: str, device: torch.device) -> float:
    criterion = get_criterion(task_type)

    best_val = float("inf")
    patience = 10
    counter = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for _epoch in range(config["epochs"]):

        model.train()
        if hasattr(optimizer, "train"):
            optimizer.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            def closure():
                optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                return loss

            optimizer.step(closure)

        val_metric = eval_for_early_stop(model, val_loader, task_type, device)

        if val_metric < best_val:
            best_val = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict(best_state)
    return float(best_val)



# Optuna objective
def objective(trial, X_full: np.ndarray, y_full: np.ndarray, task_type: str, monotonic_indices):

    hidden_sizes_options = generate_layer_combinations(
        min_layers=2, max_layers=2, units=[8, 16, 32, 64]
    )

    config = {
        "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        "hidden_sizes": trial.suggest_categorical("hidden_sizes", hidden_sizes_options),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "epochs": SEARCH_EPOCHS,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(GLOBAL_SEED)


    n = len(X_full)
    idx = np.arange(n)
    rng = np.random.RandomState(GLOBAL_SEED)
    rng.shuffle(idx)

    tr_size = int(0.8 * n)
    tr_idx = idx[:tr_size]
    va_idx = idx[tr_size:]

    X_tr, X_va = X_full[tr_idx], X_full[va_idx]
    y_tr, y_va = y_full[tr_idx], y_full[va_idx]


    X_tr, X_va = fold_minmax_scale_X(X_tr, X_va)
    y_tr, y_va, _, _ = fold_standardize_y(y_tr, y_va, task_type)

    train_ds = build_tensor_dataset(X_tr, y_tr, task_type)
    val_ds = build_tensor_dataset(X_va, y_va, task_type)

    g = torch.Generator().manual_seed(GLOBAL_SEED)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    model = create_model(config, X_full.shape[1], monotonic_indices, seed=GLOBAL_SEED).to(device)

    optimizer = AdamWScheduleFree(model.parameters(), lr=config["lr"], warmup_steps=5)

    val_metric = train_model(model, optimizer, train_loader, val_loader, config, task_type, device)
    return float(val_metric)


def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, task_type: str, monotonic_indices, n_trials: int = N_TRIALS):

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED)
    )

    study.optimize(
        lambda trial: objective(trial, X, y, task_type, monotonic_indices),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )

    best = study.best_params
    best["epochs"] = FINAL_EPOCHS
    return best



# Cross validation
def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    best_config: Dict,
    task_type: str,
    monotonic_indices,
    n_splits: int = N_SPLITS
):
    if isinstance(best_config.get("hidden_sizes"), str):
        best_config["hidden_sizes"] = ast.literal_eval(best_config["hidden_sizes"])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mono_metrics = {"random": [], "train": [], "val": []}
    n_params = None

    if task_type == "regression":
        rmse_list, nrmse_list = [], []
    else:
        err_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        set_global_seed(GLOBAL_SEED + fold)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]


        X_train, X_val = fold_minmax_scale_X(X_train, X_val)
        y_train, y_val, y_mean, y_std = fold_standardize_y(y_train, y_val, task_type)

        train_ds = build_tensor_dataset(X_train, y_train, task_type)
        val_ds = build_tensor_dataset(X_val, y_val, task_type)

        g = torch.Generator().manual_seed(GLOBAL_SEED + fold)
        train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True, generator=g)
        val_loader = DataLoader(val_ds, batch_size=best_config["batch_size"])

        model = create_model(best_config, X.shape[1], monotonic_indices, seed=GLOBAL_SEED + fold).to(device)

        if n_params is None:
            n_params = count_parameters(model)

        optimizer = AdamWScheduleFree(model.parameters(), lr=best_config["lr"], warmup_steps=5)

        train_model(model, optimizer, train_loader, val_loader, best_config, task_type, device)

        # performance metric
        if task_type == "regression":
            rmse_raw, nrmse = eval_regression_raw_metrics(model, val_loader, device, y_mean=y_mean, y_std=y_std)
            rmse_list.append(rmse_raw)
            nrmse_list.append(nrmse)
        else:
            err = eval_for_early_stop(model, val_loader, task_type, device)
            err_list.append(err)

        # monotonicity metric
        if len(monotonic_indices) == 0:
            mono_metrics["random"].append(0.0)
            mono_metrics["train"].append(0.0)
            mono_metrics["val"].append(0.0)
            continue

        n_points = min(MAX_MONO_POINTS, len(X_train), len(X_val))
        if n_points <= 1:
            mono_metrics["random"].append(0.0)
            mono_metrics["train"].append(0.0)
            mono_metrics["val"].append(0.0)
            continue

        rng = np.random.RandomState(GLOBAL_SEED + fold)
        tr_s = rng.choice(len(X_train), n_points, replace=False)
        va_s = rng.choice(len(X_val), n_points, replace=False)

        train_sample = torch.FloatTensor(X_train[tr_s]).to(device)
        val_sample = torch.FloatTensor(X_val[va_s]).to(device)
        random_sample = sample_random_in_domain(X_train, n_points, GLOBAL_SEED + fold, device)

        mono_metrics["random"].append(
            safe_monotonicity_check(model, optimizer, random_sample, monotonic_indices, device)
        )
        mono_metrics["train"].append(
            safe_monotonicity_check(model, optimizer, train_sample, monotonic_indices, device)
        )
        mono_metrics["val"].append(
            safe_monotonicity_check(model, optimizer, val_sample, monotonic_indices, device)
        )


    avg_mono = {
        k: (float(np.mean(v)), float(np.std(v)))
        for k, v in mono_metrics.items()
    }

    if task_type == "regression":
        return rmse_list, nrmse_list, avg_mono, int(n_params or 0)
    else:
        return err_list, None, avg_mono, int(n_params or 0)



# Main
def main():
    set_global_seed(GLOBAL_SEED)

    dataset_loaders = [
        load_abalone, load_auto_mpg,
        load_boston_housing, load_compas,
        load_era, load_esl, load_heart,
        load_lev, load_swd
    ]


    results_file = "exps_LMN.csv"


    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Task Type", "Metric Name",
            "Metric Mean", "Metric Std",
            "NumOfParameters", "Best Configuration",
            "Mono Random Mean", "Mono Random Std",
            "Mono Train Mean", "Mono Train Std",
            "Mono Val Mean", "Mono Val Std"
        ])

    for loader in dataset_loaders:
        print(f"\nProcessing dataset: {loader.__name__} with LMN...")

        X, y = loader()
        task_type = get_task_type(loader)
        monotonic_indices = get_reordered_monotonic_indices(loader.__name__)


        best_config = optimize_hyperparameters(X, y, task_type, monotonic_indices, n_trials=N_TRIALS)


        scores, nrmse_scores, mono_metrics, n_params = cross_validate(
            X, y, best_config, task_type, monotonic_indices, n_splits=N_SPLITS
        )


        if task_type == "regression":
            metric_name = "NRMSE"

            final_mean = float(np.mean(nrmse_scores))
            final_std = float(np.std(nrmse_scores))
        else:
            metric_name = "Error Rate"

            final_mean = float(np.mean(scores))
            final_std = float(np.std(scores))


        write_results_to_csv(
            filename=results_file,
            dataset_name=loader.__name__,
            task_type=task_type,
            metric_name=metric_name,
            metric_mean=final_mean,
            metric_std=final_std,
            n_params=n_params,
            best_config=best_config,
            mono_metrics=mono_metrics
        )


        print(f"{loader.__name__} | {metric_name}: {final_mean:.4f} ± {final_std:.4f}")



if __name__ == "__main__":
    main()