# processing/cleaner.py

import pandas as pd


def clean_time_series(
    df: pd.DataFrame,
    freq: str = "15min",
    interpolate: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    df = df.copy()

    if verbose:
        print("=== CLEANING START ===")
        print(f"Initial shape: {df.shape}")

    # 1. Ensure datetime type
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. Sort
    df = df.sort_values("timestamp")

    # 3. Drop duplicates
    dup_count = df.duplicated(subset="timestamp").sum()
    if dup_count > 0 and verbose:
        print(f"Removing {dup_count} duplicate timestamps")

    df = df.drop_duplicates(subset="timestamp")

    # 4. Set index
    df = df.set_index("timestamp")

    # 5. Check original spacing
    diffs = df.index.to_series().diff().value_counts()

    if verbose:
        print("Original time delta distribution:")
        print(diffs)

    # 6. Create full time index
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq,
        tz=df.index.tz
    )

    # 7. Reindex (introduces NaNs where missing)
    df = df.reindex(full_index)

    missing_count = df["value"].isna().sum()

    if verbose:
        print(f"Missing timestamps introduced: {missing_count}")

    # 8. Handle missing values
    if interpolate:
        df["value"] = df["value"].interpolate(method="time")

        # Edge fill (start/end)
        df["value"] = df["value"].ffill().bfill()

    # 9. Final validation
    final_diffs = df.index.to_series().diff().value_counts()

    if verbose:
        print("Final time delta distribution:")
        print(final_diffs)

    if len(final_diffs) != 1:
        raise ValueError("Time series is still irregular after cleaning")

    if verbose:
        print(f"Final shape: {df.shape}")
        print("=== CLEANING COMPLETE ===")

    # 10. Reset index (optional but useful downstream)
    df = df.reset_index().rename(columns={"index": "timestamp"})

    return df