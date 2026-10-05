from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import DEV_END, STUDY_START
from src.data_loader import load_prices
from src.features import FORWARD_TARGET_HORIZON_DAYS, align_X_y, make_features, make_targets
from src.ml_models import (
    ModelFamily,
    extract_oos_residuals,
    predict_1d,
    train_rf,
    train_ridge_linreg,
    train_xgb,
)

QUARTERLY_REBAL_HORIZON_D = 63
FAMILIES: list[ModelFamily] = ["ridge_linreg", "rf", "xgb"]


def _load_panel() -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, list[str], pd.DataFrame, pd.DataFrame]:
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
    return X, y, common_dates, tickers, close, universe


def _rebalance_dates_from_phase3(n_expected: int = 37) -> pd.DatetimeIndex:
    w_path = ROOT / "data" / "processed" / "phase3_baseline_weights.csv"
    if not w_path.exists():
        raise FileNotFoundError(f"Phase3 weights not found at {w_path}; re-run phase3_run_5baselines.py")
    wdf = pd.read_csv(w_path)
    if "rebal_date" not in wdf.columns and "rebalance_date" in wdf.columns:
        col = "rebalance_date"
    elif "rebal_date" in wdf.columns:
        col = "rebal_date"
    else:
        raise KeyError(
            f"phase3_baseline_weights.csv missing 'rebal_date'|'rebalance_date' col, cols={wdf.columns.tolist()[:5]}"
        )
    rd = pd.DatetimeIndex(pd.to_datetime(wdf[col]).unique()).sort_values()
    if len(rd) != n_expected:
        raise AssertionError(
            f"expected {n_expected} rebalance dates matching phase3, got {len(rd)}: "
            f"{rd.tolist()[:5]}...{rd.tolist()[-3:]}"
        )
    return rd


def _train_factory(family: ModelFamily, X_tr: pd.DataFrame, y_tr: pd.Series) -> object:
    if family == "ridge_linreg":
        return train_ridge_linreg(X_tr, y_tr)
    if family == "rf":
        return train_rf(X_tr, y_tr)
    if family == "xgb":
        n_tr = len(X_tr)
        if n_tr < 50:
            return train_xgb(X_tr, y_tr, n_estimators=100)
        order = np.argsort(X_tr.index.get_level_values("date").values)
        X_sorted = X_tr.iloc[order]
        y_sorted = y_tr.iloc[order]
        cut = max(1, int(n_tr * 85 // 100))
        X_fit = X_sorted.iloc[:cut]
        y_fit = y_sorted.iloc[:cut]
        X_tv = X_sorted.iloc[cut:]
        y_tv = y_sorted.iloc[cut:]
        return train_xgb(X_fit, y_fit, early_stopping_eval_set=(X_tv, y_tv))
    raise ValueError(family)


def _build_latest_X_per_ticker(
    X: pd.DataFrame,
    tickers: list[str],
    rd: pd.Timestamp,
) -> pd.DataFrame:
    date_arr = X.index.get_level_values("date")
    tkr_arr = X.index.get_level_values("ticker")
    mask_before = date_arr < rd
    frames: list[pd.DataFrame] = []
    tkrs_out: list[str] = []
    for t in tickers:
        mask_t = tkr_arr == t
        sub_idx = np.where(np.asarray(mask_before, dtype=bool) & np.asarray(mask_t, dtype=bool))[0]
        if len(sub_idx) == 0:
            raise ValueError(f"no features for ticker {t} before {rd}")
        last_i = int(sub_idx[-1])
        frames.append(X.iloc[[last_i]])
        tkrs_out.append(t)
    out = pd.concat(frames, axis=0)
    actual = out.index.get_level_values("ticker").tolist()
    if actual != tkrs_out:
        order = [actual.index(t) for t in tkrs_out]
        out = out.iloc[order]
    return out


def main() -> int:
    X, y, common_dates, tickers, close, universe_df = _load_panel()
    print(f"[load] X={X.shape}, dates_n={len(common_dates)}, N_tickers={len(tickers)}")

    rebal_dts = _rebalance_dates_from_phase3(37)
    print(f"[rd] {len(rebal_dts)} rebalance dates, first={rebal_dts[0].date()} last={rebal_dts[-1].date()}")

    target_dates_all = X.index.get_level_values("date")
    tickers_all = X.index.get_level_values("ticker")
    target_dates_unique = np.asarray(X.index.get_level_values("date").unique().sort_values())

    HOR = int(FORWARD_TARGET_HORIZON_DAYS)
    SCALE = QUARTERLY_REBAL_HORIZON_D / HOR
    assert abs(SCALE - 3.0) < 1e-12, f"63/21={63/21}!=3"

    forecast_rows: dict[ModelFamily, list[dict]] = {fam: [] for fam in FAMILIES}

    train_dates_series = X.index.get_level_values("date")

    for rd_i, rd in enumerate(rebal_dts):
        print(f"  [{rd_i+1:02d}/{len(rebal_dts)}] rd={rd.date()}")
        train_mask = np.asarray(
            train_dates_series < (rd - pd.Timedelta(days=HOR)), dtype=bool
        )
        row_idx = np.where(train_mask)[0]
        if len(row_idx) < 500:
            print(f"    SKIP (too few train rows {len(row_idx)} < 500 at rd={rd.date()})")
            continue
        X_tr = X.iloc[row_idx]
        y_tr = y.iloc[row_idx]
        tr_max_date = X_tr.index.get_level_values("date").max()
        assert tr_max_date < rd - pd.Timedelta(days=HOR - 1), (
            f"LEAK: train_max={tr_max_date} vs rd={rd} horizon={HOR}"
        )

        X_latest = _build_latest_X_per_ticker(X, tickers, rd)
        latest_max_date = X_latest.index.get_level_values("date").max()
        assert latest_max_date < rd, f"LEAK: latest={latest_max_date} >= rd={rd}"
        assert not X_latest.isna().any().any(), f"NaN in X_latest at rd={rd}"

        for fam_i, fam in enumerate(FAMILIES):
            try:
                model = _train_factory(fam, X_tr, y_tr)
                p21 = predict_1d(model, X_latest)
            except Exception as e:
                print(f"    !! {fam} failed at rd={rd.date()}: {e}")
                p21 = np.full(len(tickers), np.nan, dtype=float)
            mu_63 = p21 * SCALE
            for t_i, tkr in enumerate(tickers):
                forecast_rows[fam].append({
                    "rebalance_date": rd,
                    "ticker": tkr,
                    "model_family": fam,
                    "mu_hat_logret_63d": float(mu_63[t_i]),
                    "yhat_21d_point": float(p21[t_i]),
                    "train_start_date": X_tr.index.get_level_values("date").min(),
                    "train_end_date": tr_max_date,
                    "n_train_rows": int(len(X_tr)),
                })

    for fam in FAMILIES:
        df = pd.DataFrame(forecast_rows[fam])
        out_csv = ROOT / "data" / "processed" / f"phase4_{fam}_forecasts.csv"
        df.to_csv(out_csv, index=False)
        print(f"[write] {out_csv.name}: rows={len(df)} cols={list(df.columns)}")
        n_unique_rd = df["rebalance_date"].nunique()
        assert n_unique_rd <= len(rebal_dts)
        print(f"        unique_rd={n_unique_rd}, unique_tickers={df['ticker'].nunique()}")

    print("[phase4.2 + 4.3/4.4: residuals + diagnostics")
    residuals_rows = []
    diagnostics_rows = []
    for fam in FAMILIES:
        yhat, resid, meta = extract_oos_residuals(fam, X, y, n_splits=5)
        summary = meta[0]
        folds = meta[1:]
        residuals_rows.append(summary)
        diagnostics_rows.extend(folds)
        print(f"[resid {fam}] rmse={summary['oos_rmse']:.5f} r2={summary['oos_r2']:.4f} jb_p={summary['jb_p_value']:.3g} rej={summary['gaussian_null_rejected_at_1pct']}")

    pd.DataFrame(residuals_rows).to_csv(
        ROOT / "data" / "processed" / "phase4_residuals_summary.csv", index=False
    )
    diag_df = pd.DataFrame(diagnostics_rows)
    diag_df.to_csv(
        ROOT / "data" / "processed" / "phase4_oos_pred_diagnostics.csv", index=False
    )
    print(f"[diag] rows={len(diag_df)} cols={list(diag_df.columns)}")
    print()
    print("[Phase 4 diag note] RMSE/R² above are MODEL-LEVEL diagnostics ONLY. "
          "Real success metric = walk-forward portfolio tx-Sharpe vs baselines. "
          "Low/negative R² here does NOT mean Phase 4 failed (low S/N domain by construction).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
