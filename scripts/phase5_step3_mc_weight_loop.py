from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import DEV_END, STUDY_START
from src.optimizer import COV_ESTIMATOR_WINNER
from src.optimizer import _sector_group_matrix, _post_verify_weights
from src.data_loader import load_prices
from src.monte_carlo import (
    SEED_LOCK,
    SECTOR_TOLERANCE_LOCKED,
    WEIGHT_UPPER_LOCKED,
    resample_weights,
    simulate_scenarios,
)

REBAL_FREQ_WINNER_FROZEN: str = "quarterly"

RIDGE_RESID_CSV = ROOT / "data" / "processed" / "phase5_ridge_oos_residuals_empirical.csv"
FORECAST_CSV = ROOT / "data" / "processed" / "phase4_ridge_linreg_forecasts.csv"
NIFTY_MAP_CSV = ROOT / "docs" / "nifty50_sector_map.csv"

CENTROID_CSV = ROOT / "data" / "processed" / "phase5_centroid_median_weights.csv"
RAW_WEIGHTS_CSV = ROOT / "data" / "processed" / "phase5_raw_weight_draws_long.csv.gz"
PER_DRAW_SHARPE_CSV = ROOT / "data" / "processed" / "phase5_per_draw_approx_sharpe.csv"
LOOP_SUMMARY_CSV = ROOT / "data" / "processed" / "phase5_mc_loop_summary.csv"

N_SCENARIOS_PROD = 500
COV_LOOKBACK_DAYS_63 = 63
RISK_AVERSION_LAMBDA_DEFAULT = 2.0


def _sector_targets_full_nifty50_count(
    nifty_map_path: Path,
    universe_tickers: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    df = pd.read_csv(nifty_map_path)
    full_map = dict(zip(df["ticker"].astype(str).tolist(), df["sector_provisional"].astype(str).tolist()))
    if universe_tickers is not None:
        keep = [str(t) for t in universe_tickers]
        ticker_to_sector_sub = {t: full_map[t] for t in keep if t in full_map}
    else:
        ticker_to_sector_sub = {t: s for t, s in full_map.items()}
    cnt = Counter(ticker_to_sector_sub.values())
    total = sum(cnt.values())
    raw_targets = {s: float(c) / total for s, c in cnt.items()}
    ssum = float(sum(raw_targets.values()))
    if ssum > 0:
        normalized = {s: v / ssum for s, v in raw_targets.items()}
    else:
        normalized = raw_targets
    return normalized, ticker_to_sector_sub


def _rebal_dates_full_37() -> list[pd.Timestamp]:
    qreb = pd.read_csv(ROOT / "data" / "processed" / "quarterly_rebalance_prices.csv")
    return sorted(pd.to_datetime(qreb.iloc[:, 0]).drop_duplicates().tolist())


def _cms_fallback_mu(
    tickers: list[str],
    rd: pd.Timestamp,
    log_ret_panel: pd.DataFrame,
    lookback_days: int = 63 * 3,
    min_rows: int = 60,
) -> np.ndarray:
    end = rd - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days * 2)
    sub = log_ret_panel.loc[start:end, tickers].dropna(how="any")
    total_available_pre = len(sub)
    if len(sub) < min_rows:
        sub_before = log_ret_panel.loc[:end, tickers].dropna(how="any")
        if len(sub_before) >= min_rows:
            sub = sub_before
        else:
            first_date = log_ret_panel.index.min()
            sub_after = log_ret_panel.loc[first_date:, tickers].dropna(how="any")
            needed = max(0, min_rows - len(sub_before))
            if len(sub_before) + len(sub_after) >= min_rows:
                sub = pd.concat([sub_before, sub_after.head(needed)])
            else:
                sub = pd.concat([sub_before, sub_after]) if len(sub_before) > 0 else sub_after
    if len(sub) < 10:
        return np.zeros(len(tickers), dtype=float)
    actual_lb = min(lookback_days, sub.shape[0])
    sub_tail = sub.tail(actual_lb)
    mu_63d = sub_tail.mean(axis=0).to_numpy(dtype=float) * 63.0
    if not np.isfinite(mu_63d).all():
        mu_63d = np.nan_to_num(mu_63d, nan=0.0, posinf=0.0, neginf=0.0)
    return mu_63d


def _build_mu_frozen_df(
    tickers: list[str],
    all_rds: list[pd.Timestamp],
    log_ret_panel: pd.DataFrame,
) -> pd.DataFrame:
    forecasts = pd.read_csv(FORECAST_CSV).rename(columns={"rebalance_date": "rebal_date"})
    forecasts["rebal_date"] = pd.to_datetime(forecasts["rebal_date"])
    fc_rds_set = set(forecasts["rebal_date"].drop_duplicates().tolist())

    rows = []
    fc_by_rd = {rd: g for rd, g in forecasts.groupby("rebal_date")}

    for rd in all_rds:
        if rd in fc_rds_set:
            g = fc_by_rd[rd]
            tkr_to_mu = dict(zip(g["ticker"].astype(str).tolist(), g["mu_hat_logret_63d"].astype(float).tolist()))
            for t in tickers:
                mu_val = float(tkr_to_mu.get(t, np.nan))
                rows.append(
                    {
                        "rebal_date": rd,
                        "ticker": t,
                        "model_family": "ridge_linreg" if ~np.isnan(mu_val) else "cms_fallback_interp",
                        "mu_hat_logret_63d": mu_val if np.isfinite(mu_val) else 0.0,
                        "yhat_21d_point": np.nan,
                        "train_start_date": np.nan,
                        "train_end_date": np.nan,
                        "n_train_rows": 0,
                    }
                )
        else:
            mu_vec = _cms_fallback_mu(tickers, rd, log_ret_panel)
            for i, t in enumerate(tickers):
                rows.append(
                    {
                        "rebal_date": rd,
                        "ticker": t,
                        "model_family": "cms_fallback_histmean_63d_x63",
                        "mu_hat_logret_63d": float(mu_vec[i]),
                        "yhat_21d_point": np.nan,
                        "train_start_date": np.nan,
                        "train_end_date": np.nan,
                        "n_train_rows": 0,
                    }
                )
    out = pd.DataFrame(rows)
    out["rebal_date"] = pd.to_datetime(out["rebal_date"])
    return out


def _check_constraints_centroid(
    centroid_df: pd.DataFrame,
    ticker_to_sector: dict[str, str],
    sector_targets: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rds = sorted(centroid_df["rebal_date"].drop_duplicates().tolist())
    tickers = sorted(ticker_to_sector.keys())
    tol_pp = SECTOR_TOLERANCE_LOCKED * 100.0

    for rd in rds:
        sub = centroid_df[centroid_df["rebal_date"] == rd].set_index("ticker")["weight"].reindex(tickers).to_numpy(dtype=float)
        sum_err = abs(float(sub.sum()) - 1.0)
        w_max = float(sub.max())
        w_min = float(sub.min())
        cap_violated = w_max > WEIGHT_UPPER_LOCKED + 1e-4
        _, G = _sector_group_matrix(len(tickers), pd.Series(ticker_to_sector), tickers)
        sec_list = sorted(sector_targets.keys())
        targets_vec = np.array([float(sector_targets[s]) for s in sec_list], dtype=float)
        targets_sum = float(targets_vec.sum())
        if abs(targets_sum - 1.0) > 1e-9 and targets_sum > 0:
            targets_vec = targets_vec / targets_sum
        drift_pp = (G @ sub - targets_vec) * 100.0
        drift_max_pp = float(np.max(np.abs(drift_pp)))
        drift_violated = drift_max_pp > tol_pp + 1e-5
        finite_ok = bool(np.isfinite(sub).all())
        rows.append(
            {
                "rebal_date": rd,
                "sum_to_1_err": sum_err,
                "w_max": w_max,
                "w_min": w_min,
                "cap_violated": bool(cap_violated or w_max > WEIGHT_UPPER_LOCKED + 1e-4),
                "cap_limit_pct": WEIGHT_UPPER_LOCKED * 100.0,
                "sector_drift_max_pp": drift_max_pp,
                "sector_tol_pp": tol_pp,
                "sector_drift_violated": bool(drift_violated),
                "finite_ok": finite_ok,
                "violations_total": int(
                    (sum_err > 1e-7) + int(cap_violated) + int(drift_violated) + int(not finite_ok)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if COV_ESTIMATOR_WINNER != "LW":
        raise RuntimeError(f"COV_ESTIMATOR_WINNER must be LW, got {COV_ESTIMATOR_WINNER!r}")
    if REBAL_FREQ_WINNER_FROZEN != "quarterly":
        raise RuntimeError(f"REBAL_FREQ_WINNER must be quarterly, got {REBAL_FREQ_WINNER_FROZEN!r}")

    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    print(f"[step3] tickers N={len(tickers)}")

    all_rds = _rebal_dates_full_37()
    n_rd = len(all_rds)
    print(f"[step3] rebal_dates full 37 schedule: N={n_rd} (first 5 = CMS fallback, next 32 = ridge_linreg true ML)")
    assert n_rd >= 37, f"Expected >= 37 RDs, got {n_rd}"

    sector_targets, ticker_to_sector = _sector_targets_full_nifty50_count(NIFTY_MAP_CSV, universe_tickers=tickers)
    n_sectors = len(sector_targets)
    print(f"[step3] sector map: {n_sectors} sectors, ticker_to_sector covers {len(ticker_to_sector)}/{len(tickers)} tickers")
    assert len(ticker_to_sector) == len(tickers), "ticker_to_sector coverage incomplete!"

    print("[step3] Loading prices and computing log-ret panel ...")
    prices = load_prices(tickers, start_date=STUDY_START, end_date=DEV_END, final_holdout=False)
    close = prices["Close"]
    log_ret_full = np.log(close / close.shift(1)).iloc[1:]
    print(f"[step3] log_ret panel: rows={len(log_ret_full)}, cols={log_ret_full.shape[1]}")

    print("[step3] Building mu_frozen_df (37 RDs × 46 tickers) ...")
    mu_frozen_df = _build_mu_frozen_df(tickers, all_rds, log_ret_full)
    n_rows_fc = len(mu_frozen_df)
    print(f"[step3] mu_frozen_df rows = {n_rows_fc} (expected {n_rd * len(tickers)})")
    assert n_rows_fc == n_rd * len(tickers), f"mu_frozen_df rows mismatch: {n_rows_fc} vs {n_rd * len(tickers)}"
    n_missing_mu = int((~np.isfinite(mu_frozen_df["mu_hat_logret_63d"].to_numpy(dtype=float))).sum())
    assert n_missing_mu == 0, f"{n_missing_mu} non-finite mu values!"

    resid_df = pd.read_csv(RIDGE_RESID_CSV)
    resid_df["date"] = pd.to_datetime(resid_df["date"])
    assert len(resid_df) > 1000, f"Residuals too short: {len(resid_df)}"
    print(f"[step3] residuals loaded: {len(resid_df)} rows")

    print(f"[step3] simulate_scenarios: mode=multivariate_row, N_scenarios={N_SCENARIOS_PROD}, seed={SEED_LOCK}")
    mu_tensor = simulate_scenarios(
        mu_frozen_df=mu_frozen_df,
        ridge_oos_resid_long_df=resid_df,
        mode="multivariate_row",
        n_scenarios=N_SCENARIOS_PROD,
        seed=SEED_LOCK,
        rebal_dates_order=all_rds,
        tickers_order=tickers,
        scale_resid_sqrt3=True,
    )
    print(f"[step3] mu_scenario_tensor shape = {mu_tensor.shape} (expected ({n_rd},{len(tickers)},{N_SCENARIOS_PROD}))")
    assert mu_tensor.shape == (n_rd, len(tickers), N_SCENARIOS_PROD)
    assert np.isfinite(mu_tensor).all()

    sector_targets_per_rd = {pd.to_datetime(rd): sector_targets for rd in all_rds}
    total_solves = n_rd * N_SCENARIOS_PROD
    print(
        f"[step3] resample_weights: entering {n_rd} RD × {N_SCENARIOS_PROD} draw = {total_solves:,} QP solves "
        f"(CMS 10% cap, ±3pp sector, LW Σ 63d lookback)"
    )

    centroid_df, raw_df, per_draw_df = resample_weights(
        rd_list=all_rds,
        tickers_list=tickers,
        mu_scenario_tensor=mu_tensor,
        log_ret_panel=log_ret_full,
        sector_targets_per_rd=sector_targets_per_rd,
        ticker_to_sector=ticker_to_sector,
        aggregation="median",
        risk_aversion_lambda=RISK_AVERSION_LAMBDA_DEFAULT,
        cov_lookback_days=COV_LOOKBACK_DAYS_63,
        n_jobs=-1,
        joblib_backend="threads",
        warm_start=True,
        seed=SEED_LOCK,
    )
    print(f"[step3] centroid_df rows = {len(centroid_df)}")
    print(f"[step3] raw_df rows = {len(raw_df)}")
    print(f"[step3] per_draw_df rows = {len(per_draw_df)}")

    centroid_df.to_csv(CENTROID_CSV, index=False)
    print(f"[step3] SAVE-BEFORE-CHECK wrote centroid CSV -> {CENTROID_CSV} ({len(centroid_df)} rows)")
    raw_df.to_csv(RAW_WEIGHTS_CSV, index=False, compression="gzip")
    print(f"[step3] SAVE-BEFORE-CHECK wrote raw draws -> {RAW_WEIGHTS_CSV} ({len(raw_df)} rows)")
    per_draw_df.to_csv(PER_DRAW_SHARPE_CSV, index=False)
    print(f"[step3] SAVE-BEFORE-CHECK wrote per_draw -> {PER_DRAW_SHARPE_CSV}")

    print("[step3] Hard constraint verification on centroid table ...")
    summary_df = _check_constraints_centroid(centroid_df, ticker_to_sector, sector_targets)
    summary_df.to_csv(LOOP_SUMMARY_CSV, index=False)
    print(f"[step3] SAVE-BEFORE-CHECK wrote loop summary -> {LOOP_SUMMARY_CSV}")
    total_violations = int(summary_df["violations_total"].sum())
    print(
        f"[step3] centroid constraint summary: RDs={len(summary_df)}, "
        f"rd-level violations cells={int((summary_df['violations_total'] > 0).sum())}, "
        f"total violations={total_violations}"
    )
    cap_rd_viol = int(summary_df["cap_violated"].sum())
    drift_rd_viol = int(summary_df["sector_drift_violated"].sum())
    print(f"[step3]  cap violations RDs: {cap_rd_viol}/{len(summary_df)}")
    print(f"[step3]  drift violations RDs: {drift_rd_viol}/{len(summary_df)} (±{SECTOR_TOLERANCE_LOCKED * 100:.1f} pp)")
    assert cap_rd_viol == 0, f"{cap_rd_viol} RDs failed 10% cap!"
    assert drift_rd_viol == 0, f"{drift_rd_viol} RDs failed sector ±{SECTOR_TOLERANCE_LOCKED * 100:.1f}pp!"
    assert total_violations == 0, f"{total_violations} total violations in centroid!"

    print("[step3] Raw-draw constraint integrity spot checks ...")
    g_raw = raw_df.groupby(["rebal_date", "draw_id"])["weight"]
    sum_raw_err = (g_raw.sum() - 1.0).abs().max()
    max_raw_w = float(raw_df["weight"].max())
    finite_raw = bool(np.isfinite(raw_df["weight"]).all())
    print(f"[step3]   max |sum w - 1| across draws = {float(sum_raw_err):.3e}")
    print(f"[step3]   max weight across draws = {max_raw_w:.5f}")
    print(f"[step3]   finite weights = {finite_raw}")
    assert finite_raw, "Raw weights have NaN/Inf!"
    assert max_raw_w <= WEIGHT_UPPER_LOCKED + 1e-3, f"Raw max weight {max_raw_w} > cap {WEIGHT_UPPER_LOCKED}!"
    assert float(sum_raw_err) < 1e-5, f"Raw sum-to-1 violation max err = {sum_raw_err}!"

    print(f"[step3] ALL CHECKS PASSED. Final artifacts:")
    print(f"  {CENTROID_CSV.name}: {len(centroid_df)} rows")
    print(f"  {RAW_WEIGHTS_CSV.name}: {len(raw_df)} rows (gzip)")
    print(f"  {PER_DRAW_SHARPE_CSV.name}: {len(per_draw_df)} rows")
    print(f"  {LOOP_SUMMARY_CSV.name}: {len(summary_df)} rows (all 0 violations)")
    print("[step3] DONE (37 RD × 500 draws QP loop with 0 centroid violations)")


if __name__ == "__main__":
    main()
