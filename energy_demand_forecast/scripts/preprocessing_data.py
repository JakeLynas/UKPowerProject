from pathlib import Path
import pandas as pd
import numpy as np

IN_PATH = Path("data/processed/combined.parquet")  # or combined.csv (auto-detected)
OUT_DIR = Path("data/processed"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_combined(path_parquet=IN_PATH) -> pd.DataFrame:
    if path_parquet.exists():
        df = pd.read_parquet(path_parquet)
    else:
        csv = Path(str(path_parquet).replace(".parquet", ".csv"))
        if not csv.exists():
            raise FileNotFoundError(f"Neither {path_parquet} nor {csv} exists.")
        df = pd.read_csv(csv)
    return df

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # expected columns: publishTime, startTime, value, series, kind
    for col in ("publishTime","startTime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    if "series" in df:
        df["series"] = df["series"].astype("category")
    if "kind" in df:
        df["kind"] = df["kind"].astype("category")
    return df

def add_sp_and_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["settlement_date"] = df["startTime"].dt.date
    mins = df["startTime"].dt.hour * 60 + df["startTime"].dt.minute
    df["settlement_period"] = (mins // 30 + 1).astype("int16")
    return df

def dedupe_latest_publication(df: pd.DataFrame) -> pd.DataFrame:
    # keep the latest publication per (series, kind, startTime)
    return (df.sort_values("publishTime")
              .dropna(subset=["startTime"])
              .drop_duplicates(subset=["series","kind","startTime"], keep="last"))

def validate_day_counts(df: pd.DataFrame, series="demand"):
    sub = df[df["series"] == series]
    counts = sub.groupby(["kind","settlement_date"])["startTime"].nunique().rename("n")
    bad = counts[~counts.isin([46,48,50])]
    if not bad.empty:
        print(f"⚠️ {series}: unexpected period counts detected (showing up to 10):")
        print(bad.head(10))

def pivot_demand_wide(df: pd.DataFrame) -> pd.DataFrame:
    # demand only → wide (columns: actual, forecast)
    d = df[df["series"] == "demand"][["startTime","kind","value"]]
    wide = d.pivot(index="startTime", columns="kind", values="value").reset_index()
    # normalize column names if categories carry order
    wide.columns.name = None
    # ensure both columns exist
    if "actual" not in wide:  wide["actual"] = np.nan
    if "forecast" not in wide: wide["forecast"] = np.nan
    return wide.sort_values("startTime")

def add_features(wide: pd.DataFrame) -> pd.DataFrame:
    # index on time for lag-safe ops
    df = wide.sort_values("startTime").set_index("startTime")
    # lags of actual
    df["lag_48"] = df["actual"].shift(48)
    df["lag_96"] = df["actual"].shift(96)
    # weekly same-time rolling mean (past only)
    df["roll7d_same_time"] = (
        df["actual"].shift(48).rolling(window=48*7, min_periods=48*3).mean()
    )
    # calendar
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    # back to columns
    df = df.reset_index()
    # simple QC: drop impossible/negatives
    for col in ("actual","forecast"):
        df.loc[df[col] < 0, col] = np.nan
    # drop rows missing target or core feature(s)
    df = df.dropna(subset=["actual","forecast","lag_48"])
    return df

def temporal_split(df: pd.DataFrame, valid_frac=0.2):
    df = df.sort_values("startTime")
    cut = int(len(df) * (1 - valid_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def save_outputs(demand_clean: pd.DataFrame,
                 demand_wide: pd.DataFrame,
                 supply_clean: pd.DataFrame | None = None):
    (OUT_DIR / "tables").mkdir(exist_ok=True, parents=True)
    demand_clean.to_parquet(OUT_DIR / "tables" / "demand_clean.parquet", index=False)
    demand_wide.to_parquet(OUT_DIR / "tables" / "demand_wide.parquet", index=False)
    if supply_clean is not None and not supply_clean.empty:
        supply_clean.to_parquet(OUT_DIR / "tables" / "supply_clean.parquet", index=False)

def run_preprocess_combined(in_path: Path = IN_PATH, valid_frac=0.2):
    # 1) load + normalize
    combined = enforce_schema(load_combined(in_path))
    # 2) dedupe on latest publication per series/kind/period
    combined = dedupe_latest_publication(combined)
    # 3) add date & SP
    combined = add_sp_and_date(combined)
    # 4) quick sanity checks
    validate_day_counts(combined, series="demand")
    validate_day_counts(combined, series="supply")

    # 5) supply/demand clean copies (optional save for later)
    supply_clean = combined[combined["series"] == "supply"].copy()

    # 6) demand → wide (actual vs forecast)
    demand_wide = pivot_demand_wide(combined)

    # 7) features for modeling
    demand_feats = add_features(demand_wide)

    # 8) split
    train, valid = temporal_split(demand_feats, valid_frac=valid_frac)

    # 9) save results
    save_outputs(demand_feats, demand_wide, supply_clean)
    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    valid.to_parquet(OUT_DIR / "valid.parquet", index=False)

    print(f"Saved: train={len(train)}, valid={len(valid)}, total={len(demand_feats)} rows")
    return train, valid

if __name__ == "__main__":
    run_preprocess_combined()


def combine_series_data(
    base_dir: str | Path = "data/test_out",
    series_list=("demand", "supply"),
    kind_list=("actual", "forecast"),
    filename="data.csv"
) -> pd.DataFrame:
    """
    Combine demand/supply × actual/forecast CSV files into one DataFrame.

    Parameters
    ----------
    base_dir : str or Path
        Root directory containing series/kind folders.
    series_list : iterable
        E.g., ("demand", "supply").
    kind_list : iterable
        E.g., ("actual", "forecast").
    filename : str
        Name of CSV file stored in each leaf folder.

    Returns
    -------
    pd.DataFrame
        Combined tidy DataFrame with `series`, `kind`, timestamps, and values.
    """
    base_dir = Path(base_dir)
    dfs = []

    for series in series_list:
        for kind in kind_list:
            file_path = base_dir / series / kind / filename
            if not file_path.exists():
                print(f"⚠️  Skipping missing file: {file_path}")
                continue

            df = pd.read_csv(file_path)
            # normalize timestamps
            for col in ("publishTime", "startTime"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            # rename value column if needed
            if "value" not in df.columns:
                raise ValueError(f"Expected 'value' column in {file_path}")
            df["series"] = series
            df["kind"] = kind
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {base_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    # enforce schema
    combined["series"] = combined["series"].astype("category")
    combined["kind"] = combined["kind"].astype("category")
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce").astype("float64")

    return combined


# if __name__ == "__main__":
#     #run_preprocess()
#     combined = combine_series_data("data/test_out")
#     print(combined.head())
#     print(combined["series"].value_counts())
#     print(combined["kind"].value_counts())

#     # Save for easy reuse
#     out_path = Path("data/processed/combined")
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     combined.to_parquet(f"{out_path}.parquet", index=False)
#     combined.to_csv(f"{out_path}.csv", index=False)
