# anomalies/inject.py

import numpy as np
import pandas as pd


def initialize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "is_true_anomaly" not in df.columns:
        df["is_true_anomaly"] = 0

    return df


# ---------------------------------------------------
# SPIKE ANOMALY
# ---------------------------------------------------

def inject_spike(
    df: pd.DataFrame,
    index: int,
    magnitude: float = 3.0
) -> pd.DataFrame:

    df = initialize_labels(df)

    std = df["value"].std()

    df.loc[index, "value"] += magnitude * std

    df.loc[index, "is_true_anomaly"] = 1

    return df


# ---------------------------------------------------
# DRIFT ANOMALY
# ---------------------------------------------------

def inject_drift(
    df: pd.DataFrame,
    start_index: int,
    length: int = 12,
    magnitude: float = 2.0
) -> pd.DataFrame:

    df = initialize_labels(df)

    std = df["value"].std()

    drift = np.linspace(0, magnitude * std, length)

    end_index = min(start_index + length, len(df))

    actual_length = end_index - start_index

    df.loc[start_index:end_index - 1, "value"] += drift[:actual_length]

    df.loc[start_index:end_index - 1, "is_true_anomaly"] = 1

    return df


# ---------------------------------------------------
# DROPOUT ANOMALY
# ---------------------------------------------------

def inject_dropout(
    df: pd.DataFrame,
    start_index: int,
    length: int = 8
) -> pd.DataFrame:

    df = initialize_labels(df)

    end_index = min(start_index + length, len(df))

    df.loc[start_index:end_index - 1, "value"] = 0

    df.loc[start_index:end_index - 1, "is_true_anomaly"] = 1

    return df


# ---------------------------------------------------
# RANDOMIZED PIPELINE
# ---------------------------------------------------

def inject_random_anomalies(
    df: pd.DataFrame,
    n_spikes: int = 5,
    n_drifts: int = 2,
    n_dropouts: int = 2,
    random_state: int = 42
) -> pd.DataFrame:

    np.random.seed(random_state)

    df = initialize_labels(df)

    n = len(df)

    # -----------------------------
    # Spikes
    # -----------------------------
    for _ in range(n_spikes):
        idx = np.random.randint(0, n)

        df = inject_spike(
            df,
            index=idx,
            magnitude=np.random.uniform(2, 5)
        )

    # -----------------------------
    # Drifts
    # -----------------------------
    for _ in range(n_drifts):
        idx = np.random.randint(0, n - 20)

        df = inject_drift(
            df,
            start_index=idx,
            length=np.random.randint(8, 20),
            magnitude=np.random.uniform(1, 3)
        )

    # -----------------------------
    # Dropouts
    # -----------------------------
    for _ in range(n_dropouts):
        idx = np.random.randint(0, n - 10)

        df = inject_dropout(
            df,
            start_index=idx,
            length=np.random.randint(4, 12)
        )

    return df