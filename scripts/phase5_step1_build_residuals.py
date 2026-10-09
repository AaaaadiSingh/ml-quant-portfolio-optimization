from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, t

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import DEV_END, STUDY_START
from src.data_loader import load_prices
from src.features import align_X_y, make_features, make_targets
from src.ml_models import extract_oos_residuals


RIDGE_RESID_CSV = ROOT / "data" / "processed" / "phase5_ridge_oos_residuals_empirical.csv"
RESID_SUMMARY_CSV = ROOT / "data" / "processed" / "phase5_residual_distribution_summary.csv"
EXPECTED_ROWS: int = 89608


def _load_panel() -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, pd.MultiIndex]:
    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    prices = load_prices(
        tickers,
        start_date=STUDY_START,
        end_date=DEV_END,
        final_holdout=False,
    )
    close, high, low, volume = prices["Close"], prices["High"], prices["Low"], prices["Volume"]
    X_all = make_features(close, high, low, volume)
    y_all = make_targets(close)
    X, y, common_dates = align_X_y(X_all, y_all)
    return X, y, common_dates, X.index  # type: ignore[return-value]


def main() -> None:
    X, y, common_dates, aligned_index = _load_panel()
    print(f"[step1] aligned panel rows = {len(X)} cols = {X.shape[1]}")
    assert len(X) >= EXPECTED_ROWS - 100, f"expected ~{EXPECTED_ROWS} rows, got {len(X)}"

    dec_path = ROOT / "data" / "processed" / "phase4_model_selection_decision.csv"
    winner_family = str(pd.read_csv(dec_path)["selected_model"].iloc[0]) if dec_path.exists() else "rf"
    print(f"[step1] Extracting OOS residuals for winner family: {winner_family}")
    yhat, residuals, fold_summary_list = extract_oos_residuals(
        family=winner_family,
        X=X,
        y=y,
        date_index=common_dates,
        n_splits=5,
    )
    assert len(residuals) == len(yhat), "yhat/resid length mismatch from extract_oos_residuals"
    n_returned = int(len(residuals))
    print(
        f"[step1] extract returned yhat len = {n_returned}, resid len = {n_returned}, "
        f"summaries = {len(fold_summary_list)} rows (1 summary + 5 fold rows)"
    )

    finite_mask = np.isfinite(residuals) & np.isfinite(yhat)
    yhat_clean = yhat[finite_mask]
    resid_clean = residuals[finite_mask]

    full_idx_arr = np.asarray(list(aligned_index))
    if len(full_idx_arr) > len(finite_mask):
        full_idx_arr = full_idx_arr[: len(finite_mask)]
    kept_index = full_idx_arr[finite_mask]
    kept = int(finite_mask.sum())
    print(f"[step1] finite valid rows after 5-fold TSS concat = {kept}")

    ytrue_s = y.reindex(aligned_index).iloc[: len(finite_mask)][finite_mask]
    out_df = pd.DataFrame(
        {
            "date": pd.to_datetime([d for d, _ in kept_index]),
            "ticker": [str(t) for _, t in kept_index],
            "y_true": ytrue_s.values.astype(float),
            "yhat_oos": yhat_clean.astype(float),
            "residual": resid_clean.astype(float),
        }
    )
    print(f"[step1] long residual rows = {len(out_df)} (target {EXPECTED_ROWS}; TSS 5-fold drops ~15k train/val overlap rows)")
    out_df.to_csv(RIDGE_RESID_CSV, index=False)
    print(f"[step1] wrote residuals CSV -> {RIDGE_RESID_CSV} ({len(out_df)} rows)")

    res_clean = residuals[np.isfinite(residuals)]
    q = np.quantile(res_clean, [0.0, 0.25, 0.5, 0.75, 1.0])
    mu_r = float(np.mean(res_clean))
    sd_r = float(np.std(res_clean, ddof=1))
    skew = float(pd.Series(res_clean).skew())
    kurt_exc = float(pd.Series(res_clean).kurt())
    jb_s, jb_p = jarque_bera(res_clean)
    gauss_rej = bool(jb_p < 0.01)
    try:
        t_df, t_loc, t_scale = t.fit(res_clean)
        t_df_f, t_loc_f, t_scale_f = float(t_df), float(t_loc), float(t_scale)
    except Exception:
        t_df_f = t_loc_f = t_scale_f = float("nan")

    summary_row = {
        "model_family": "ridge_linreg",
        "n_obs": int(len(res_clean)),
        "min": float(q[0]),
        "q1": float(q[1]),
        "median": float(q[2]),
        "q3": float(q[3]),
        "max": float(q[4]),
        "mean": mu_r,
        "std": sd_r,
        "skew": skew,
        "kurtosis_excess": kurt_exc,
        "jb_stat": float(jb_s),
        "jb_p_value": float(jb_p),
        "gaussian_null_rejected_at_1pct": gauss_rej,
        "t_df_mle": t_df_f,
        "t_loc_mle": t_loc_f,
        "t_scale_mle": t_scale_f,
    }
    print(
        f"[step1] kurtosis_excess = {kurt_exc:.4f} (> 0 required); "
        f"JB p-value = {jb_p:.3e} (REJECTED={gauss_rej})"
    )
    assert kurt_exc > 0.0, f"kurtosis_excess = {kurt_exc} must be > 0 (empirical heavy tails)"
    assert gauss_rej is True, (
        "Gaussian null was NOT rejected — contradict phase4_residuals_summary.csv frozen JB p=0.0!"
    )

    pd.DataFrame([summary_row]).to_csv(RESID_SUMMARY_CSV, index=False)
    print(f"[step1] wrote summary CSV -> {RESID_SUMMARY_CSV}")
    print("[step1] DONE")


if __name__ == "__main__":
    main()
