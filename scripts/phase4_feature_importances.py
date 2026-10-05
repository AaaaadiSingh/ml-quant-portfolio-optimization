from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import DEV_END, STUDY_START
from src.data_loader import load_prices
from src.features import align_X_y, make_features, make_targets
from src.ml_models import (
    feature_family_of,
    predict_1d,
    train_rf,
    train_ridge_linreg,
    train_xgb,
)

RANDOM_SEED = 7
PERMUTATION_N_REPEATS = 10


def _holdout_temporal_chunk(X: pd.DataFrame, y: pd.Series, year: int = 2021):
    date_arr = X.index.get_level_values("date")
    mask = (date_arr >= pd.Timestamp(f"{year}-01-01")) & (date_arr < pd.Timestamp(f"{year+1}-01-01"))
    idx = np.where(mask.values if hasattr(mask, "values") else np.asarray(mask, dtype=bool))[0]
    return X.iloc[idx], y.iloc[idx]


def _train_split_before_year(X: pd.DataFrame, y: pd.Series, year: int = 2021):
    date_arr = X.index.get_level_values("date")
    end = pd.Timestamp(f"{year}-01-01")
    mask = date_arr < end
    idx = np.where(mask.values if hasattr(mask, "values") else np.asarray(mask, dtype=bool))
    return X.iloc[idx], y.iloc[idx]


def main() -> int:
    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    prices = load_prices(tickers, start_date=STUDY_START, end_date=DEV_END, final_holdout=False)
    close, high, low, volume = prices["Close"], prices["High"], prices["Low"], prices["Volume"]
    X_all = make_features(close, high, low, volume)
    y_all = make_targets(close)
    X, y, common_dates = align_X_y(X_all, y_all)
    feature_names = list(X.columns)
    print(f"[load] X={X.shape}, features={len(feature_names)}")
    X_tr, y_tr = _train_split_before_year(X, y, 2021)
    X_va, y_va = _holdout_temporal_chunk(X, y, 2021)
    print(f"[split] train={len(X_tr)} (before 2021), val={len(X_va)} (2021 calendar)")
    if len(X_tr) == 0 or len(X_va) == 0:
        raise RuntimeError(f"bad temporal split: train={len(X_tr)}, val={len(X_va)}")

    rows: list[dict] = []

    # Ridge: standardized coefficient magnitudes
    print("[ridge] training + coefs")
    ridge = train_ridge_linreg(X_tr, y_tr)
    scaler = ridge.named_steps["scaler"]
    model = ridge.named_steps["model"]
    std = np.asarray(scaler.scale_, dtype=float)
    raw_coefs = np.asarray(model.coef_, dtype=float).flatten()
    standardized = np.abs(raw_coefs * std)
    order = np.argsort(-standardized)
    for rank, j in enumerate(order, start=1):
        fn = feature_names[j]
        rows.append({
            "model_family": "ridge_linreg",
            "feature_name": fn,
            "family_group": feature_family_of(fn),
            "importance_mean": float(standardized[j]),
            "importance_std": 0.0,
            "rank_by_mean": int(rank),
        })

    # RF + XGB permutation importance
    for factory, name in [
        (train_rf, "rf"),
        (lambda Xt, yt: train_xgb(Xt, yt, n_estimators=300), "xgb"),
    ]:
        print(f"[{name}] training...")
        m = factory(X_tr, y_tr)
        print(f"[{name}] predicting baseline val MSE...")
        base_preds = predict_1d(m, X_va)
        base_err = (y_va.values - base_preds) ** 2
        base_mse = float(np.mean(base_err))
        print(f"[{name}] val MSE baseline = {base_mse:.5f}. permuting {len(feature_names)} features n={PERMUTATION_N_REPEATS}x")
        rng = np.random.default_rng(RANDOM_SEED)
        importances = np.zeros(len(feature_names), dtype=float)
        imp_std = np.zeros(len(feature_names), dtype=float)
        X_va_values = X_va.values.copy()
        yva = y_va.values.copy()
        for j, fn in enumerate(feature_names):
            deltas = []
            col_orig = X_va_values[:, j].copy()
            for rep in range(PERMUTATION_N_REPEATS):
                permuted = rng.permutation(col_orig)
                X_va_values[:, j] = permuted
                X_va_tmp = pd.DataFrame(X_va_values.copy(), columns=feature_names, index=X_va.index)
                p = predict_1d(m, X_va_tmp)
                perm_err = (yva - p) ** 2
                delta = float(np.mean(perm_err) - base_mse)
                deltas.append(delta)
            X_va_values[:, j] = col_orig
            importances[j] = float(np.mean(deltas))
            imp_std[j] = float(np.std(deltas, ddof=0))
        order = np.argsort(-importances)
        for rank, j in enumerate(order, start=1):
            fn = feature_names[j]
            rows.append({
                "model_family": name,
                "feature_name": fn,
                "family_group": feature_family_of(fn),
                "importance_mean": float(importances[j]),
                "importance_std": float(imp_std[j]),
                "rank_by_mean": int(rank),
            })

    df = pd.DataFrame(rows)
    out = ROOT / "data" / "processed" / "phase4_feature_importances.csv"
    df.to_csv(out, index=False)
    print(f"[write] {out.name}: rows={len(df)} cols={list(df.columns)}")

    print("\n[summary] top-5 by family (importance_mean):")
    for fam in ["ridge_linreg", "rf", "xgb"]:
        sub = df[df["model_family"] == fam].sort_values("importance_mean", ascending=False).head(5)
        print(f"  {fam:15s}:")
        for r in sub.itertuples(index=False):
            print(f"    {int(r.rank_by_mean):>2}. {r.feature_name[:36]:<36} fam={r.family_group[:18]:<18} mean={r.importance_mean:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
