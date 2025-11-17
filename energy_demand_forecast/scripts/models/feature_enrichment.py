import numpy as np
import pandas as pd
from pathlib import Path

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add recommended engineered features to a half-hourly dataset.

    Requires columns:
        - 'startTime' (datetime)
        - 'actual' (float)
        - 'forecast' (float)

    Adds columns:
        - lag_1, lag_48, lag_96
        - roll7d_same_time, roll7d_std
        - hour_sin, hour_cos
        - dow, is_weekend
        - forecast_error_lag1
    """

    req = {"startTime", "actual", "forecast"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    out = df.copy()

    # Ensure chronological order
    if "startTime" in out.columns:
        out["startTime"] = pd.to_datetime(out["startTime"], utc=True, errors="coerce")
        out = out.sort_values("startTime")

    # Use a time index for safe shifting/rolling
    out = out.set_index("startTime")

    # -------- LAGS (no leakage) --------
    out["lag_1"]   = out["actual"].shift(1)     # previous 30-min
    out["lag_48"]  = out["actual"].shift(48)    # same time yesterday
    out["lag_96"]  = out["actual"].shift(96)    # same time two days ago

    # -------- ROLLING (past only) --------
    # Weekly same-time mean: use a 48*7-window over the yesterday-shifted series
    out["roll7d_same_time"] = (
        out["actual"]
        .shift(48)  # ensure only past info
        .rolling(window=48*7, min_periods=48*3)  # need at least ~3 days to start
        .mean()
    )

    # Weekly same-time std (variability context)
    out["roll7d_std"] = (
        out["actual"]
        .shift(48)
        .rolling(window=48*7, min_periods=48*3)
        .std()
    )

    # -------- CALENDAR / CYCLIC --------
    # Bring back a time index view for hour/dow
    idx = out.index
    hour = idx.hour
    dow = idx.dayofweek

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow"] = dow.astype(np.int16)
    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)

    # -------- BIAS / ERROR FEATURES --------
    # Yesterday's forecast error at the same half-hour
    out["forecast_error_lag1"] = out["actual"].shift(48) - out["forecast"].shift(48)

    # Reset index so downstream code sees 'startTime' as a column again
    out = out.reset_index()

    return out

def engineer_df(series: str = "demand"):
    path_parquet = Path("data/processed/tables/{series}_wide.parquet".format(series=series))
    if path_parquet.exists():
        df = pd.read_parquet(path_parquet)
    else:
        csv = Path(str(path_parquet).replace(".parquet", ".csv"))
        if not csv.exists():
            raise FileNotFoundError(f"Neither {path_parquet} nor {csv} exists.")
        df = pd.read_csv(csv)
    
    df = add_engineered_features(df)
    out_path = Path("data/processed/tables/{series}_wide_engineered.parquet".format(series=series))
    df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    engineer_df("demand")
    engineer_df("supply")