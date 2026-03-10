# exps_PWL.py

import ast
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import optuna
from schedulefree import AdamWScheduleFree
from typing import Callable, Dict, List, Tuple, Any

from src.MLP import StandardMLP
from src.PWLNetwork import pwl_mono_reg

from dataPreprocessing.loaders import (
    load_abalone, load_auto_mpg, load_boston_housing,
    load_compas, load_era, load_esl, load_heart,
    load_lev, load_swd
)

from src.utils import (
    write_results_to_csv,
    count_parameters,
    generate_layer_combinations,
    monotonicity_check,
    get_reordered_monotonic_indices
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
N_TRIALS = 20
PATIENCE = 10
MAX_MONO_POINTS = 1000


# =====================================================
# Loader unification
# =====================================================
def load_full_dataset(loader: Callable) -> Tuple[np.ndarray, np.ndarray]:
    out = loader()
    if len(out) == 2:
        return np.asarray(out[0]), np.asarray(out[1])
    elif len(out) == 4:
        X, y, X_test, y_test = out
        X_full = np.vstack([X, X_test])
        y_full = np.concatenate([y, y_test])
        return np.asarray(X_full), np.asarray(y_full)
    else:
        raise ValueError("Unexpected loader output.")


def get_task_type(loader: Callable) -> str:
    regression_tasks = [
        load_abalone, load_auto_mpg,
        load_boston_housing, load_era,
        load_esl, load_lev, load_swd
    ]
    return "regression" if loader in regression_tasks else "classification"


def make_tensor_dataset(X, y, task_type):
    if task_type == "classification":
        y = ensure_binary_labels(y)

    return TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y).reshape(-1, 1)
    )


# =====================================================
# Model
# =====================================================
def create_model(config: Dict, input_size: int, seed: int):
    torch.manual_seed(seed)

    hidden_sizes = config["hidden_sizes"]
    if isinstance(hidden_sizes, str):
        hidden_sizes = ast.literal_eval(hidden_sizes)

    return StandardMLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=1,
        activation=nn.ReLU(),
        output_activation=nn.Identity()
    )


# =====================================================
# Monotonicity safe wrapper
# =====================================================
def sample_random_in_domain(X_ref, n_points, seed, device):
    rng = np.random.RandomState(seed)
    x_min = np.nanmin(X_ref, axis=0)
    x_max = np.nanmax(X_ref, axis=0)
    span = x_max - x_min
    span[span == 0] = 1.0
    X_rand = x_min + rng.rand(n_points, X_ref.shape[1]) * span
    return torch.FloatTensor(X_rand).to(device)


def safe_monotonicity_check(model, optimizer, data_tensor, mono_idx, device):
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt_state = copy.deepcopy(optimizer.state_dict())

    try:
        score = monotonicity_check(model, optimizer, data_tensor, mono_idx, device)
    finally:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(opt_state)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    return float(score)


# =====================================================
# Training
# =====================================================
def train_model(model, optimizer, train_loader, val_loader,
                config, task_type, device, mono_idx):

    criterion = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()

    best_val = float("inf")
    counter = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for _ in range(config["epochs"]):

        model.train()
        if hasattr(optimizer, "train"):
            optimizer.train()

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            def closure():
                optimizer.zero_grad()

                out = model(Xb)
                empirical = criterion(out, yb)

                if len(mono_idx) == 0:
                    mono_loss = torch.zeros((), device=device)
                else:
                    # 使用配置中的 offset 值
                    mono_loss = pwl_mono_reg(
                        model,
                        Xb,
                        mono_idx,
                        float(config["offset"])  # 这里使用 config["offset"]
                    )

                loss = empirical + float(config["monotonicity_weight"]) * mono_loss
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

        if counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return float(best_val)


# =====================================================
# Optuna
# =====================================================
# 修改 objective 函数中的配置字典，确保包含 offset 默认值
def objective(trial, X_full, y_full, task_type, mono_idx):

    hidden_options = generate_layer_combinations(2, 2, [8, 16, 32, 64])

    config = {
        "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        "hidden_sizes": trial.suggest_categorical("hidden_sizes", hidden_options),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "monotonicity_weight": trial.suggest_categorical("monotonicity_weight", [1., 10., 100., 1000.]),
        "offset": trial.suggest_float("offset", 0.0, 0.5, step=0.05),  # 使用 optuna 选择 offset 值
        "epochs": SEARCH_EPOCHS,
    }

    # 其他代码保持不变
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(GLOBAL_SEED)

    idx = np.arange(len(X_full))
    rng = np.random.RandomState(GLOBAL_SEED)
    rng.shuffle(idx)

    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, X_va = X_full[tr_idx], X_full[va_idx]
    y_tr, y_va = y_full[tr_idx], y_full[va_idx]

    X_tr, X_va = fold_minmax_scale_X(X_tr, X_va)
    y_tr, y_va, _, _ = fold_standardize_y(y_tr, y_va, task_type)

    g = torch.Generator().manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(
        make_tensor_dataset(X_tr, y_tr, task_type),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=g
    )

    val_loader = DataLoader(
        make_tensor_dataset(X_va, y_va, task_type),
        batch_size=config["batch_size"]
    )

    model = create_model(config, X_full.shape[1], GLOBAL_SEED).to(device)
    optimizer = AdamWScheduleFree(model.parameters(), lr=config["lr"], warmup_steps=5)

    return train_model(model, optimizer, train_loader, val_loader,
                       config, task_type, device, mono_idx)


def optimize_hyperparameters(X, y, task_type, mono_idx):
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))

    study.optimize(lambda trial: objective(trial, X, y, task_type, mono_idx),
                   n_trials=N_TRIALS, n_jobs=1)

    best = study.best_params
    if isinstance(best["hidden_sizes"], str):
        best["hidden_sizes"] = ast.literal_eval(best["hidden_sizes"])
    best["epochs"] = FINAL_EPOCHS
    return best


# =====================================================
# Cross Validation (FULLY UNIFIED)
# =====================================================
def cross_validate(X, y, best_config, task_type, mono_idx):

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rmse_list, nrmse_list = [], []
    err_list = []
    mono_collect = {"random": [], "train": [], "val": []}

    tmp_model = create_model(best_config, X.shape[1], GLOBAL_SEED).to(device)
    n_params = int(count_parameters(tmp_model))
    del tmp_model

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):

        set_global_seed(GLOBAL_SEED + fold)

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr, X_va = fold_minmax_scale_X(X_tr, X_va)
        y_tr, y_va, y_mean, y_std = fold_standardize_y(y_tr, y_va, task_type)

        g = torch.Generator().manual_seed(GLOBAL_SEED + fold)

        train_loader = DataLoader(
            make_tensor_dataset(X_tr, y_tr, task_type),
            batch_size=best_config["batch_size"],
            shuffle=True,
            generator=g
        )

        val_loader = DataLoader(
            make_tensor_dataset(X_va, y_va, task_type),
            batch_size=best_config["batch_size"]
        )

        model = create_model(best_config, X.shape[1], GLOBAL_SEED + fold).to(device)
        optimizer = AdamWScheduleFree(model.parameters(), lr=best_config["lr"], warmup_steps=5)

        train_model(model, optimizer, train_loader, val_loader,
                    best_config, task_type, device, mono_idx)

        if task_type == "regression":
            rmse, nrmse = eval_regression_raw_metrics(model, val_loader, device, y_mean, y_std)
            rmse_list.append(float(rmse))
            nrmse_list.append(float(nrmse))
        else:
            err = eval_for_early_stop(model, val_loader, task_type, device)
            err_list.append(float(err))

        # monotonicity
        n_points = min(MAX_MONO_POINTS, len(X_tr), len(X_va))
        if len(mono_idx) == 0 or n_points <= 1:
            for k in mono_collect:
                mono_collect[k].append(0.0)
            continue

        rng = np.random.RandomState(GLOBAL_SEED + fold)
        tr_ids = rng.choice(len(X_tr), n_points, replace=False)
        va_ids = rng.choice(len(X_va), n_points, replace=False)

        train_sample = torch.FloatTensor(X_tr[tr_ids]).to(device)
        val_sample = torch.FloatTensor(X_va[va_ids]).to(device)
        rand_sample = sample_random_in_domain(X_tr, n_points, GLOBAL_SEED + fold, device)

        mono_collect["random"].append(
            safe_monotonicity_check(model, optimizer, rand_sample, mono_idx, device)
        )
        mono_collect["train"].append(
            safe_monotonicity_check(model, optimizer, train_sample, mono_idx, device)
        )
        mono_collect["val"].append(
            safe_monotonicity_check(model, optimizer, val_sample, mono_idx, device)
        )

    avg_mono = {k: (float(np.mean(v)), float(np.std(v))) for k, v in mono_collect.items()}

    if task_type == "regression":
        return rmse_list, nrmse_list, avg_mono, n_params
    else:
        return err_list, None, avg_mono, n_params


# =====================================================
# Dataset Processor
# =====================================================
def process_dataset(loader: Callable):
    X, y = load_full_dataset(loader)
    task_type = get_task_type(loader)
    mono_idx = get_reordered_monotonic_indices(loader.__name__)

    best_config = optimize_hyperparameters(X, y, task_type, mono_idx)

    scores, nrmse_scores, mono_metrics, n_params = cross_validate(
        X, y, best_config, task_type, mono_idx
    )

    return scores, nrmse_scores, mono_metrics, best_config, n_params, task_type


# =====================================================
# Main
# =====================================================
def main():
    set_global_seed(GLOBAL_SEED)

    # ✅ 1. 保留 9 个核心数据集
    dataset_loaders = [
        load_abalone, load_auto_mpg,
        load_boston_housing, load_compas,
        load_era, load_esl, load_heart,
        load_lev, load_swd
    ]

    # ✅ 2. 结果文件名
    results_file = "exps_PWL.csv"

    # ✅ 3. 写入统一表头
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
        print(f"\nProcessing dataset: {loader.__name__} with PWL Regularization...")

        # 获取实验结果
        scores, nrmse_scores, mono_metrics, best_config, n_params, task_type = process_dataset(loader)

        # ✅ 4. 回归用 NRMSE，分类用 Error Rate
        if task_type == "regression":
            metric_name = "NRMSE"
            final_mean = float(np.mean(nrmse_scores))
            final_std = float(np.std(nrmse_scores))
        else:
            metric_name = "Error Rate"
            final_mean = float(np.mean(scores))
            final_std = float(np.std(scores))

        # ✅ 5. 写入 CSV
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