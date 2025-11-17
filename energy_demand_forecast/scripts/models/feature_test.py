from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass
class EvalResult:
    feature: str
    # correlation
    spearman_rho: float
    spearman_p: float
    # OLS
    ols_coef: float
    ols_pvalue: float
    adj_r2_base: float
    adj_r2_ext: float
    delta_adj_r2: float
    # collinearity
    vif_candidate: float
    vif_max: float
    cond_number: float
    # metrics
    mae_base: float
    mae_ext: float
    delta_mae: float
    rmse_base: float
    rmse_ext: float
    delta_rmse: float
    # simple recommendation
    recommend: str

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])


def _chronological_split(
    df: pd.DataFrame, frac_valid: float = 0.2, time_col: Optional[str] = "startTime"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if time_col in df.columns:
        df = df.sort_values(time_col)
    else:
        df = df.sort_index()
    cut = max(1, int(len(df) * (1 - frac_valid)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _vif_table(X: np.ndarray, cols: List[str]) -> pd.DataFrame:
    vif = pd.DataFrame({
        "feature": ["const"] + cols,
        "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    })
    return vif


def evaluate_feature(
    df: pd.DataFrame,
    base_features: List[str],
    candidate_feature: str,
    target: str = "actual",
    frac_valid: float = 0.2,
    time_col: str = "startTime",
) -> pd.DataFrame:
    """
    Evaluate a single candidate feature against a base feature set.
    Assumes the candidate feature is already present in `df`.

    Returns a one-row DataFrame with:
      - Spearman rho/p
      - OLS coef & p-value for candidate (with adjusted R^2 delta)
      - VIF for candidate & max VIF, condition number
      - MAE/RMSE (base vs extended) and deltas
      - A simple textual recommendation
    """

    needed_cols = set(base_features + [candidate_feature, target])
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    # Keep only rows without NaNs in required cols
    work = df[list(needed_cols) + ([time_col] if time_col in df.columns else [])].copy()
    work = work.dropna()

    # -------------------------
    # Spearman (univariate)
    # -------------------------
    rho, p_spear = spearmanr(work[candidate_feature], work[target])

    # -------------------------
    # OLS (with/without candidate) on the TRAIN split
    # -------------------------
    train, valid = _chronological_split(work, frac_valid, time_col=time_col)

    Xb = train[base_features]
    Xe = train[base_features + [candidate_feature]]
    y  = train[target]

    # add intercepts
    Xb_sm = sm.add_constant(Xb, has_constant="add")
    Xe_sm = sm.add_constant(Xe, has_constant="add")

    ols_base = sm.OLS(y, Xb_sm).fit()
    ols_ext  = sm.OLS(y, Xe_sm).fit()

    # candidate stats (from extended model)
    if candidate_feature in ols_ext.params.index:
        coef = float(ols_ext.params[candidate_feature])
        pval = float(ols_ext.pvalues[candidate_feature])
    else:
        coef, pval = np.nan, np.nan  # should not happen, but be safe

    adj_r2_base = float(ols_base.rsquared_adj)
    adj_r2_ext  = float(ols_ext.rsquared_adj)
    delta_adj_r2 = adj_r2_ext - adj_r2_base

    # -------------------------
    # Collinearity: VIF & condition number (extended)
    # -------------------------
    # statsmodels VIF expects a plain ndarray
    Xe_mat = Xe_sm.values  # includes constant as first column
    vif_tbl = _vif_table(Xe_mat, cols= Xe.columns.tolist())
    vif_cand = float(vif_tbl.loc[vif_tbl["feature"] == candidate_feature, "VIF"].values[0])
    vif_max = float(vif_tbl["VIF"].max())
    cond_number = float(np.linalg.cond(Xe_mat))

    # -------------------------
    # Practical performance: MAE/RMSE on VALID split (sklearn)
    # -------------------------
    Xb_tr, y_tr = train[base_features].values, train[target].values
    Xe_tr       = train[base_features + [candidate_feature]].values
    Xb_va, y_va = valid[base_features].values, valid[target].values
    Xe_va       = valid[base_features + [candidate_feature]].values

    m_base = LinearRegression().fit(Xb_tr, y_tr)
    m_ext  = LinearRegression().fit(Xe_tr, y_tr)

    yhat_b = m_base.predict(Xb_va)
    yhat_e = m_ext.predict(Xe_va)

    mae_b  = mean_absolute_error(y_va, yhat_b)
    rmse_b = np.sqrt(mean_squared_error(y_va, yhat_b))
    mae_e  = mean_absolute_error(y_va, yhat_e)
    rmse_e = np.sqrt(mean_squared_error(y_va, yhat_e))

    d_mae  = mae_e - mae_b   # negative = improvement
    d_rmse = rmse_e - rmse_b # negative = improvement

    # -------------------------
    # Simple recommendation
    # -------------------------
    good_sig   = (pval < 0.05) if not np.isnan(pval) else False
    good_spear = (abs(rho) >= 0.2) and (p_spear < 0.05)
    good_vif   = (vif_cand < 5.0)
    good_perf  = (d_rmse < 0) or (d_mae < 0)
    good_adjr2 = (delta_adj_r2 > 0)

    if good_perf and good_sig and good_vif:
        rec = "KEEP"
    elif good_perf and (good_sig or good_spear) and (vif_cand < 10):
        rec = "KEEP (watch collinearity)"
    elif good_sig and good_adjr2 and not good_perf:
        rec = "BORDERLINE (statistically significant, but no error gain)"
    else:
        rec = "DROP"

    result = EvalResult(
        feature=candidate_feature,
        spearman_rho=float(rho),
        spearman_p=float(p_spear),
        ols_coef=float(coef),
        ols_pvalue=float(pval),
        adj_r2_base=adj_r2_base,
        adj_r2_ext=adj_r2_ext,
        delta_adj_r2=delta_adj_r2,
        vif_candidate=vif_cand,
        vif_max=vif_max,
        cond_number=cond_number,
        mae_base=float(mae_b),
        mae_ext=float(mae_e),
        delta_mae=float(d_mae),
        rmse_base=float(rmse_b),
        rmse_ext=float(rmse_e),
        delta_rmse=float(d_rmse),
        recommend=rec,
    )

    return result.as_dataframe()

def evaluate_features(df, base_features=None, candidate_features=None):
    base_features = base_features or ["forecast"]
    candidate_features = candidate_features or [
        "lag_1", "lag_48", "lag_96",
        "roll7d_same_time", "roll7d_std",
        "hour_sin", "hour_cos",
        "dow", "is_weekend",
        "forecast_error_lag1"
    ]

    results = []
    current_features = base_features.copy()

    for cand in candidate_features:
        res = evaluate_feature(df, current_features, cand, target="actual")
        results.append(res)
        if res["recommend"].iloc[0].startswith("KEEP"):
            current_features.append(cand)

    results_table = pd.concat(results, ignore_index=True)
    print(results_table)