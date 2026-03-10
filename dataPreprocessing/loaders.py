import os
import numpy as np
import pandas as pd
from typing import List, Optional, Callable



# Path settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")

def get_data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)

# Core loader
def load_data(
    path: str,
    mono_inc_list: List[int],
    mono_dec_list: List[int],
    target_column: str,
    preprocess_func: Optional[Callable] = None,
    strict_mono_check: bool = True,
):

    """
    Generic data loading function with strict monotonic feature handling.
    """

    # Load
    data = pd.read_csv(get_data_path(path))

    if preprocess_func is not None:
        data = preprocess_func(data)

    data = data.dropna()

    # Split features and target
    X = data.drop(columns=[target_column]).values.astype(np.float32)
    y = data[target_column].values

    d = X.shape[1]

    # 1) Monotonic index validity check

    bad_inc = [i for i in mono_inc_list if i < 0 or i >= d]
    bad_dec = [i for i in mono_dec_list if i < 0 or i >= d]

    if strict_mono_check and (bad_inc or bad_dec):
        raise ValueError(
            f"[load_data] Monotonic index out of range for {path}: "
            f"d={d}, bad_inc={bad_inc}, bad_dec={bad_dec}. "
            f"Check preprocessing and monotonic index definitions."
        )

    # 2) Transform decreasing to increasing

    for col in mono_dec_list:
        if 0 <= col < d:
            X[:, col] = -X[:, col]

    # 3) Reorder features: monotonic first

    mono_list = list(mono_inc_list) + list(mono_dec_list)
    mono_list = [i for i in mono_list if 0 <= i < d]

    if strict_mono_check and len(mono_list) != len(set(mono_list)):
        raise ValueError(
            f"[load_data] Duplicated monotonic indices for {path}: {mono_list}"
        )

    non_mono_list = [i for i in range(d) if i not in set(mono_list)]

    new_order = mono_list + non_mono_list
    X = X[:, new_order]

    # 4) Structural consistency check

    if strict_mono_check:
        k = len(mono_list)
        assert X.shape[1] == d
        assert k <= d

    return X, y


# Preprocessing functions

def preprocess_abalone(data):
    data = data.copy()
    data["Sex"] = pd.Categorical(data["Sex"]).codes
    return data


def preprocess_auto_mpg(data):
    data = data.copy()
    if "car name" in data.columns:
        data = data.drop("car name", axis=1)
    return data


def preprocess_compas(data):
    data = data.copy()

    data = data[
        (data["days_b_screening_arrest"] <= 30)
        & (data["days_b_screening_arrest"] >= -30)
    ]
    data = data[data["is_recid"] != -1]
    data = data[data["c_charge_degree"] <= "O"]
    data = data[data["score_text"] != "N/A"]

    data["race"] = pd.Categorical(data["race"]).codes
    data["sex"] = pd.Categorical(data["sex"]).codes

    data = data[
        [
            "priors_count",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "age",
            "race",
            "sex",
            "two_year_recid",
        ]
    ]

    return data


# Dataset-specific loaders

def load_abalone():
    return load_data(
        "abalone.csv",
        mono_inc_list=[4, 5, 6, 7],
        mono_dec_list=[],
        target_column="Rings",
        preprocess_func=preprocess_abalone,
    )


def load_auto_mpg():
    return load_data(
        "auto-mpg.csv",
        mono_inc_list=[4, 5, 6],
        mono_dec_list=[0, 1, 2, 3],
        target_column="mpg",
        preprocess_func=preprocess_auto_mpg,
    )


def load_boston_housing():
    return load_data(
        "BostonHousing.csv",
        mono_inc_list=[5],
        mono_dec_list=[0],
        target_column="MEDV",
    )


def load_compas():
    return load_data(
        "compas_scores_two_years.csv",
        mono_inc_list=[0, 1, 2, 3],
        mono_dec_list=[],
        target_column="two_year_recid",
        preprocess_func=preprocess_compas,
    )


def load_era():
    return load_data(
        "era.csv",
        mono_inc_list=[0, 1, 2, 3],
        mono_dec_list=[],
        target_column="out1",
    )


def load_esl():
    return load_data(
        "esl.csv",
        mono_inc_list=[0, 1, 2, 3],
        mono_dec_list=[],
        target_column="out1",
    )


def load_heart():
    return load_data(
        "heart.csv",
        mono_inc_list=[3, 4],
        mono_dec_list=[],
        target_column="target",
    )


def load_lev():
    return load_data(
        "lev.csv",
        mono_inc_list=[0, 1, 2, 3],
        mono_dec_list=[],
        target_column="Out1",
    )



def load_swd():
    return load_data(
        "swd.csv",
        mono_inc_list=[0, 1, 2, 4, 6, 8, 9],
        mono_dec_list=[],
        target_column="Out1",
    )