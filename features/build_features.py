# features/build_features.py

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame, window: int = 4, verbose: bool = True) -> pd.DataFrame:
    """
    window=4 → 1 hour for 15-min data
    """

    df = df.copy()

    if verbose:
        print("=== FEATURE ENGINEERING START ===")
        print(f"Input shape: {df.shape}")

    # Ensure datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Set index for rolling ops
    df = df.set_index("timestamp")

    # -------------------------
    # 1. Basic features
    # -------------------------
    df["load"] = df["value"]

    # -------------------------
    # 2. Rolling statistics
    # -------------------------
    df["rolling_mean"] = df["load"].rolling(window).mean()
    df["rolling_std"] = df["load"].rolling(window).std()

    # -------------------------
    # 3. Ramp rate (first derivative)
    # -------------------------
    df["ramp_rate"] = df["load"].diff()

    # -------------------------
    # 4. Z-score (local anomaly signal)
    # -------------------------
    df["z_score"] = (
        df["load"] - df["rolling_mean"]
    ) / df["rolling_std"]

    # -------------------------
    # 5. Time-based features
    # -------------------------
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # -------------------------
    # 6. Handle NaNs (from rolling/diff)
    # -------------------------
    nan_count = df.isna().sum().sum()

    if verbose:
        print(f"NaN values introduced: {nan_count}")

    df = df.dropna()

    # -------------------------
    # 7. Reset index
    # -------------------------
    df = df.reset_index()

    if verbose:
        print(f"Output shape: {df.shape}")
        print("=== FEATURE ENGINEERING COMPLETE ===")

    return df