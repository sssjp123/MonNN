# exps_WeightsConstrained.py

import ast
import csv
import copy
from typing import Callable, Tuple, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import optuna
from schedulefree import AdamWScheduleFree

from src.WeightsConstrainedMLP import WeightsConstrainedMLP

from dataPreprocessing.loaders import (
    load_abalone, load_auto_mpg, load_boston_housing,
    load_compas, load_era, load_esl, load_heart, load_lev, load_swd
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
N_SPLITS = 5
MAX_MONO_POINTS = 1000



# Task Type
def get_task_type(loader: Callable) -> str:
    regression_tasks = [
        load_abalone, load_auto_mpg,
        load_boston_housing, load_era,
        load_esl, load_lev, load_swd
    ]
    return "regression" if loader in regression_tasks else "classification"



# Model
def create_model(config, input_size, seed):
    torch.manual_seed(seed)

    hidden_sizes = config["hidden_sizes"]
    if isinstance(hidden_sizes, str):
        hidden_sizes = ast.literal_eval(hidden_sizes)

    return WeightsConstrainedMLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=1
    )



# Safe Monotonicity Check
def sample_random_in_domain(X_ref, n_points, seed, device):
    rng = np.random.RandomState(seed)
    x_min = np.nanmin(X_ref, axis=0)
    x_max = np.nanmax(X_ref, axis=0)
    span = x_max - x_min
    span[span == 0] = 1.0
    X_rand = x_min + rng.rand(n_points, X_ref.shape[1]) * span
    return torch.FloatTensor(X_rand).to(device)


def safe_monotonicity_check(model, optimizer, data_tensor, monotonic_indices, device):
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt_state = copy.deepcopy(optimizer.state_dict())

    try:
        score = monotonicity_check(model, optimizer, data_tensor, monotonic_indices, device)
    finally:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(opt_state)

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    return float(score)



# Training
def train_model(model, optimizer, train_loader, val_loader,
                config, task_type, device):

    criterion = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()

    best_val = float("inf")
    patience = 10
    counter = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for _ in range(config["epochs"]):

        model.train()
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
    return best_val



# Optuna
def objective(trial, X, y, task_type, monotonic_indices):

    hidden_options = generate_layer_combinations(2, 2, [8, 16, 32, 64])

    config = {
        "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        "hidden_sizes": trial.suggest_categorical("hidden_sizes", hidden_options),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "epochs": SEARCH_EPOCHS,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(GLOBAL_SEED)

    if task_type == "classification":
        y = ensure_binary_labels(y)

    n = len(X)
    idx = np.arange(n)
    rng = np.random.RandomState(GLOBAL_SEED)
    rng.shuffle(idx)

    split = int(0.8 * n)
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    X_tr, X_va = fold_minmax_scale_X(X_tr, X_va)
    y_tr, y_va, _, _ = fold_standardize_y(y_tr, y_va, task_type)

    train_ds = TensorDataset(torch.FloatTensor(X_tr),
                             torch.FloatTensor(y_tr).reshape(-1, 1))
    val_ds = TensorDataset(torch.FloatTensor(X_va),
                           torch.FloatTensor(y_va).reshape(-1, 1))

    g = torch.Generator().manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(train_ds,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              generator=g)

    val_loader = DataLoader(val_ds,
                            batch_size=config["batch_size"],
                            shuffle=False)

    model = create_model(config, X.shape[1], GLOBAL_SEED).to(device)
    optimizer = AdamWScheduleFree(model.parameters(),
                                  lr=config["lr"],
                                  warmup_steps=5)

    return train_model(model, optimizer, train_loader, val_loader,
                       config, task_type, device)


def optimize(X, y, task_type, monotonic_indices):

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED)
    )

    study.optimize(
        lambda trial: objective(trial, X, y, task_type, monotonic_indices),
        n_trials=20,
        n_jobs=1
    )

    best = study.best_params
    best["epochs"] = FINAL_EPOCHS
    return best



# Cross Validation (return avg_mono)
def cross_validate(X, y, config, task_type, monotonic_indices):

    if task_type == "classification":
        y = ensure_binary_labels(y)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task_type == "regression":
        rmse_list, nrmse_list = [], []
    else:
        err_list = []

    mono_collect = {"random": [], "train": [], "val": []}
    n_params = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        set_global_seed(GLOBAL_SEED + fold)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train, X_val = fold_minmax_scale_X(X_train, X_val)
        y_train, y_val, y_mean, y_std = fold_standardize_y(y_train, y_val, task_type)

        train_ds = TensorDataset(torch.FloatTensor(X_train),
                                 torch.FloatTensor(y_train).reshape(-1, 1))
        val_ds = TensorDataset(torch.FloatTensor(X_val),
                               torch.FloatTensor(y_val).reshape(-1, 1))

        g = torch.Generator().manual_seed(GLOBAL_SEED + fold)

        train_loader = DataLoader(train_ds,
                                  batch_size=config["batch_size"],
                                  shuffle=True,
                                  generator=g)

        val_loader = DataLoader(val_ds,
                                batch_size=config["batch_size"],
                                shuffle=False)

        model = create_model(config, X.shape[1], GLOBAL_SEED + fold).to(device)

        if n_params is None:
            n_params = count_parameters(model)

        optimizer = AdamWScheduleFree(model.parameters(),
                                      lr=config["lr"],
                                      warmup_steps=5)

        train_model(model, optimizer, train_loader, val_loader,
                    config, task_type, device)

        # performance
        if task_type == "regression":
            rmse, nrmse = eval_regression_raw_metrics(model, val_loader, device, y_mean, y_std)
            rmse_list.append(rmse)
            nrmse_list.append(nrmse)
        else:
            err = eval_for_early_stop(model, val_loader, task_type, device)
            err_list.append(err)

        # monotonicity
        if not monotonic_indices:
            mono_collect["random"].append(0.0)
            mono_collect["train"].append(0.0)
            mono_collect["val"].append(0.0)
        else:
            n_points = min(MAX_MONO_POINTS, len(X_train), len(X_val))
            rng = np.random.RandomState(GLOBAL_SEED + fold)

            tr_s = rng.choice(len(X_train), n_points, replace=False)
            va_s = rng.choice(len(X_val), n_points, replace=False)

            train_sample = torch.FloatTensor(X_train[tr_s]).to(device)
            val_sample = torch.FloatTensor(X_val[va_s]).to(device)
            rand_sample = sample_random_in_domain(X_train, n_points, GLOBAL_SEED + fold, device)

            mono_collect["random"].append(
                safe_monotonicity_check(model, optimizer, rand_sample, monotonic_indices, device)
            )
            mono_collect["train"].append(
                safe_monotonicity_check(model, optimizer, train_sample, monotonic_indices, device)
            )
            mono_collect["val"].append(
                safe_monotonicity_check(model, optimizer, val_sample, monotonic_indices, device)
            )

    avg_mono = {k: (float(np.mean(v)), float(np.std(v))) for k, v in mono_collect.items()}

    if task_type == "regression":
        return rmse_list, nrmse_list, avg_mono, n_params
    else:
        return err_list, None, avg_mono, n_params


def process_dataset(loader):

    X, y = loader()
    task_type = get_task_type(loader)

    # WeightConstrained
    monotonic_indices = list(range(X.shape[1]))

    best_config = optimize(X, y, task_type, monotonic_indices)

    scores, nrmse_scores, mono_metrics, n_params = cross_validate(
        X, y, best_config, task_type, monotonic_indices
    )

    return scores, nrmse_scores, best_config, mono_metrics, n_params, task_type


def main():
    set_global_seed(GLOBAL_SEED)

    results_file = "exps_WeightsConstrained.csv"

    dataset_loaders = [
        load_abalone, load_auto_mpg,
        load_boston_housing, load_compas,
        load_era, load_esl, load_heart,
        load_lev, load_swd
    ]

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

        scores, nrmse_scores, best_config, mono_metrics, n_params, task_type = process_dataset(loader)

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