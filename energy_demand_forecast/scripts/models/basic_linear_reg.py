# train_linear.py
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")

FEATURES_NUM = ["forecast", "lag_48", "lag_96", "roll7d_same_time"]
FEATURES_CAT = ["hour", "dow"]   # is_weekend we’ll treat as numeric binary
FEATURES_BIN = ["is_weekend"]

TARGET = "actual"

def load_sets(train_path=DATA_DIR/"train.parquet", valid_path=DATA_DIR/"valid.parquet"):
    train = pd.read_parquet(train_path)
    valid = pd.read_parquet(valid_path)

    # ensure dtypes are sane
    for c in ["startTime"]:
        if c in train.columns:
            train[c] = pd.to_datetime(train[c], utc=True, errors="coerce")
        if c in valid.columns:
            valid[c] = pd.to_datetime(valid[c], utc=True, errors="coerce")

    # drop any remaining NA in required cols
    needed = FEATURES_NUM + FEATURES_CAT + FEATURES_BIN + [TARGET]
    train = train.dropna(subset=needed).copy()
    valid = valid.dropna(subset=needed).copy()

    # cast ints for OHE categories
    for c in FEATURES_CAT:
        train[c] = train[c].astype(int)
        valid[c] = valid[c].astype(int)
    for c in FEATURES_BIN:
        train[c] = train[c].astype(int)
        valid[c] = valid[c].astype(int)

    return train, valid

def build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURES_NUM + FEATURES_BIN),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("linreg", LinearRegression())
        ]
    )
    return model

def metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label:>12}  MAE={mae:,.1f}   RMSE={rmse:,.1f}")
    return mae, rmse

def plot_actual_vs_pred(df_valid, y_pred, out_path=OUT_DIR/"actual_vs_pred.png", n_points=2000):
    # sample last n_points for a readable plot
    sub = df_valid.tail(n_points).copy()
    sub["pred"] = y_pred[-len(sub):]

    plt.figure(figsize=(12, 5))
    plt.plot(sub["startTime"], sub[TARGET], label="actual")
    plt.plot(sub["startTime"], sub["pred"], label="predicted")
    plt.title("Actual vs Predicted (validation slice)")
    plt.xlabel("time")
    plt.ylabel("MW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main(args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train, valid = load_sets()

    X_train = train[FEATURES_NUM + FEATURES_CAT + FEATURES_BIN]
    y_train = train[TARGET].values
    X_valid = valid[FEATURES_NUM + FEATURES_CAT + FEATURES_BIN]
    y_valid = valid[TARGET].values

    # Baseline: persistence ŷ = lag_48
    y_pred_base = valid["lag_48"].values
    metrics(y_valid, y_pred_base, "baseline")

    # Linear Regression
    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    metrics(y_valid, y_pred, "linreg")

    # Save artifacts
    joblib.dump(model, MODEL_DIR / "linear_reg.pkl")

    # Save predictions
    pred_df = valid[["startTime", TARGET]].copy()
    pred_df["pred_linreg"] = y_pred
    pred_df["pred_baseline"] = y_pred_base
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

    # Quick plot
    plot_actual_vs_pred(valid, y_pred, OUT_DIR / "actual_vs_pred.png")

    print(f"\nSaved model → {MODEL_DIR / 'linear_reg.pkl'}")
    print(f"Saved predictions → {OUT_DIR / 'predictions.csv'}")
    print(f"Saved plot → {OUT_DIR / 'actual_vs_pred.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train linear regression for UK demand forecasting.")
    # hooks to add switches later (e.g., different feature sets)
    args = parser.parse_args()
    main(args)
