from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import (
    DEV_END,
    QUARTERLY_STEP_D,
    STUDY_START,
    _raw_sharpe,
    _rebalance_dates,
)
from phase3_run_5baselines import (
    BASELINE_ORDER,
    COV_LOOKBACK_D,
    MU_WINDOW_D,
    RISK_FREE_ANNUAL,
    SECTOR_TOLERANCE_PP,
    WEIGHT_UPPER,
    _nb2_consistent_metrics,
    _monthly_returns,
    _rolling_max_drawdown,
    _sector_targets_full_nifty50_count,
)
from src.covariance import ledoit_wolf_cov, log_returns
from src.data_loader import FinalHoldoutDate, load_prices
from src.optimizer import (
    COV_ESTIMATOR_WINNER,
    TRADING_DAYS_PER_YEAR,
    classic_max_sharpe_weights,
    equal_weight,
)
from stage1a_apply_txcost import (
    TURNOVER_ONE_SIDED_BPS,
    apply_tx_costs_to_equity,
    turnover_from_equities_and_weights,
)

ML_FAMILIES = ["ridge_linreg", "rf", "xgb"]
ML_FAMILY_PRETTY = {
    "ridge_linreg": "Ridge μ̂ (LW Σ, ±3pp)",
    "rf": "RF μ̂ (LW Σ, ±3pp)",
    "xgb": "XGB μ̂ (LW Σ, ±3pp)",
}

MU_HAT_LOWER_BOUND = -0.35
MU_HAT_UPPER_BOUND = 0.60


def _load_forecast_map(family: str) -> dict[tuple[pd.Timestamp, str], float]:
    p = ROOT / "data" / "processed" / f"phase4_{family}_forecasts.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing phase4 forecast CSV: {p.name}. Run phase4_build_forecasts.py first.")
    df = pd.read_csv(p)
    if len(df) == 0:
        raise ValueError(f"{p.name} has 0 rows.")
    m: dict[tuple[pd.Timestamp, str], float] = {}
    for row in df.itertuples(index=False):
        rd = pd.Timestamp(row.rebalance_date)
        tkr = str(row.ticker)
        mu = float(row.mu_hat_logret_63d)
        m[(rd, tkr)] = mu
    return m


def _fill_missing_mu_hat_with_cms_fallback(
    close: pd.DataFrame,
    rebal_dts_valid: pd.DatetimeIndex,
    tickers_order: list[str],
    forecast_map: dict[tuple[pd.Timestamp, str], float],
) -> dict[pd.Timestamp, np.ndarray]:
    n = len(tickers_order)
    daily_idx = close.index.sort_values()
    log_ret_all = np.log(close / close.shift(1)).iloc[1:]
    out: dict[pd.Timestamp, np.ndarray] = {}
    for rd in rebal_dts_valid:
        mu_vec = np.zeros(n, dtype=float)
        any_filled_from_ml = False
        for j, t in enumerate(tickers_order):
            v = forecast_map.get((rd, t), None)
            if v is not None and np.isfinite(v):
                mu_vec[j] = float(v)
                any_filled_from_ml = True
        if not any_filled_from_ml:
            pos = int(daily_idx.get_loc(rd)) if rd in daily_idx else -1
            if pos >= 0:
                start_i = max(0, int(pos) - int(MU_WINDOW_D))
                slice_ = log_ret_all.iloc[start_i : int(pos) + 1]
                if len(slice_) >= 10:
                    mu_daily = slice_.mean(axis=0).values.astype(float)
                    tkr_ord_sub = [t for t in tickers_order if t in slice_.columns]
                    if len(tkr_ord_sub) == n:
                        mu_vec = mu_daily * TRADING_DAYS_PER_YEAR
                    else:
                        c2v = dict(zip(slice_.columns, mu_daily))
                        mu_vec = np.array([c2v.get(t, 0.0) for t in tickers_order], dtype=float) * TRADING_DAYS_PER_YEAR
        lo = float(MU_HAT_LOWER_BOUND)
        hi = float(MU_HAT_UPPER_BOUND)
        in_bounds = np.isfinite(mu_vec).all() and bool(((mu_vec >= lo) & (mu_vec <= hi)).all())
        if not in_bounds:
            clipped = np.clip(np.nan_to_num(mu_vec, nan=0.0, posinf=hi, neginf=lo), lo, hi)
            out[rd] = clipped
        else:
            out[rd] = mu_vec
    return out


def _wf_one_ml_family(
    close: pd.DataFrame,
    rebal_dts: pd.DatetimeIndex,
    tickers_order: list[str],
    cms_sector_info: dict,
    mu_by_rd: dict[pd.Timestamp, np.ndarray],
) -> tuple[pd.Series, dict[pd.Timestamp, np.ndarray]]:
    if COV_ESTIMATOR_WINNER != "LW":
        raise RuntimeError(
            f"COV_ESTIMATOR_WINNER={COV_ESTIMATOR_WINNER!r} locked in optimizer.py. Phase4 script requires LW."
        )
    n = len(tickers_order)
    daily_idx = pd.DatetimeIndex(close.index.sort_values())
    rebal_pos = [int(p) for p in daily_idx.get_indexer(rebal_dts) if p >= 0]
    valid_dts = pd.DatetimeIndex([daily_idx[p] for p in rebal_pos])
    weights_by_rebal: dict[pd.Timestamp, np.ndarray] = {}
    ticker_to_sector = cms_sector_info["ticker_to_sector"]
    sector_targets = cms_sector_info["sector_targets"]
    sec_constraints = {
        "sector_targets": sector_targets,
        "ticker_to_sector": pd.Series(ticker_to_sector),
        "sector_tolerance": SECTOR_TOLERANCE_PP,
    }
    for rd, pos in zip(valid_dts, rebal_pos):
        lb_start = max(0, int(pos) - COV_LOOKBACK_D + 1)
        look_slice = close.iloc[lb_start : int(pos) + 1]
        log_ret = log_returns(look_slice)
        if len(log_ret) < 5:
            weights_by_rebal[rd] = equal_weight(n)
            continue
        mu_vec = mu_by_rd.get(rd, np.zeros(n, dtype=float))
        assert np.isfinite(mu_vec).all(), f"non-finite μ̂ at rd={rd}"
        assert bool(((mu_vec >= MU_HAT_LOWER_BOUND) & (mu_vec <= MU_HAT_UPPER_BOUND)).all()), (
            f"μ̂ implausible at rd={rd}: min={mu_vec.min():.3f} max={mu_vec.max():.3f}"
        )
        cov_df, _ = ledoit_wolf_cov(log_ret)
        w = classic_max_sharpe_weights(
            mu=mu_vec,
            cov=cov_df.values.astype(float),
            risk_free_rate=RISK_FREE_ANNUAL,
            w_upper=WEIGHT_UPPER,
            sector_constraints=sec_constraints,
            tickers_order=tickers_order,
        )
        weights_by_rebal[rd] = np.asarray(w, dtype=float)
    rebal_keys_sorted = sorted(weights_by_rebal.keys(), key=lambda d: daily_idx.get_loc(d))
    equity_arr = np.ones(len(daily_idx), dtype=float)
    for k, rd in enumerate(rebal_keys_sorted):
        w = weights_by_rebal[rd]
        pos0 = daily_idx.get_loc(rd)
        if k + 1 < len(rebal_keys_sorted):
            pos1 = daily_idx.get_loc(rebal_keys_sorted[k + 1])
        else:
            pos1 = len(daily_idx)
        sub = close.iloc[pos0:pos1]
        if len(sub) < 1:
            continue
        sub_close = sub.values.astype(float)
        init_row = sub_close[0, :]
        safe = np.where(np.isfinite(init_row) & (init_row > 0), init_row, 1.0)
        sub_ret = sub_close / safe
        port_curve = sub_ret @ w
        equity_arr[pos0:pos1] = equity_arr[pos0] * port_curve
    equity_s = pd.Series(equity_arr, index=daily_idx).sort_index()
    return equity_s, weights_by_rebal


def _cms_panel5_check(
    w_by_rd: dict[pd.Timestamp, np.ndarray],
    tickers_order: list[str],
    sector_info: dict,
) -> dict[str, float]:
    ticker_to_sector = sector_info["ticker_to_sector"]
    sectors_sorted = sorted({ticker_to_sector[t] for t in tickers_order})
    targets = sector_info["sector_targets"]
    tol = float(SECTOR_TOLERANCE_PP)
    max_pos = 0.0
    max_neg = 0.0
    n_tol_violations = 0
    for rd, w in w_by_rd.items():
        wmap = dict(zip(tickers_order, w.astype(float).tolist()))
        actual_by_sec: dict[str, float] = {s: 0.0 for s in sectors_sorted}
        for t, v in wmap.items():
            actual_by_sec[ticker_to_sector[t]] = actual_by_sec.get(ticker_to_sector[t], 0.0) + float(v)
        for s in sectors_sorted:
            tgt = float(targets.get(s, 0.0))
            drift = float(actual_by_sec.get(s, 0.0) - tgt)
            if drift > max_pos:
                max_pos = drift
            if drift < max_neg:
                max_neg = drift
            if abs(drift) > tol + 1e-6:
                n_tol_violations += 1
    return {
        "max_pos_drift_pp": max_pos * 100.0,
        "max_neg_drift_pp": max_neg * 100.0,
        "violations_n": n_tol_violations,
        "violated_flag": bool(n_tol_violations > 0),
    }


def _sortino_ratio(eq: pd.Series, rf_annual: float = RISK_FREE_ANNUAL, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    e = eq.sort_index().dropna()
    if len(e) < 2:
        return float("nan")
    r = e.pct_change().fillna(0.0)
    rf_d = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0
    excess = r - rf_d
    downside = np.where(excess < 0, excess, 0.0)
    sigma_d = float(np.sqrt(np.mean(downside**2)))
    if sigma_d <= 1e-16:
        return float("nan")
    return float(excess.mean() / sigma_d * np.sqrt(trading_days))


def main() -> int:
    print("[Phase 4 Work Group 4] ML μ̂ strategies through frozen quarterly+LW pipeline")
    assert COV_ESTIMATOR_WINNER == "LW", f"COV_ESTIMATOR_WINNER must be LW, got {COV_ESTIMATOR_WINNER!r}"
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    universe_df = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers_order = sorted(universe_df["ticker"].astype(str).tolist())
    n = len(tickers_order)
    prices = load_prices(
        tickers_order,
        start_date=STUDY_START,
        end_date=DEV_END,
        final_holdout=False,
    )
    assert prices.index.max() < FinalHoldoutDate, f"holdout leak max date {prices.index.max()}"
    close = prices["Close"].ffill().bfill().sort_index().loc[:, tickers_order]
    daily_idx = pd.DatetimeIndex(close.index.sort_values())
    rebal_dts = _rebalance_dates(daily_idx, QUARTERLY_STEP_D)
    rebal_pos = [int(p) for p in daily_idx.get_indexer(rebal_dts) if p >= 0]
    rebal_dts_valid = pd.DatetimeIndex([daily_idx[p] for p in rebal_pos])
    print(f"[info] N={n} tickers, n_rebal={len(rebal_dts_valid)} (quarterly)")
    assert len(rebal_dts_valid) == 37, f"expected 37, got {len(rebal_dts_valid)}"

    sector_targets, ticker_to_sector = _sector_targets_full_nifty50_count(
        ROOT / "docs" / "nifty50_sector_map.csv",
        universe_tickers=tickers_order,
    )
    cms_sector_info = {
        "sector_targets": sector_targets,
        "ticker_to_sector": ticker_to_sector,
        "tickers_order": tickers_order,
        "sector_tolerance": 0.03,
    }

    summary_rows: list[dict] = []
    weight_long_rows: list[dict] = []
    raw_equity_frames: dict[str, pd.Series] = {}
    txadj_equity_frames: dict[str, pd.Series] = {}
    panel5_per_family: dict[str, dict] = {}

    for family in ML_FAMILIES:
        print(f"\n[---] ML family = {family}")
        forecast_map = _load_forecast_map(family)
        print(f"      forecast_map entries = {len(forecast_map)}")
        mu_by_rd = _fill_missing_mu_hat_with_cms_fallback(close, rebal_dts_valid, tickers_order, forecast_map)
        n_ml = sum(1 for rd in rebal_dts_valid if (rd, tickers_order[0]) in forecast_map)
        print(f"      n_RD_w_ML_forecast = {n_ml}/37")
        equ_raw, w_by_rd = _wf_one_ml_family(
            close=close,
            rebal_dts=rebal_dts_valid,
            tickers_order=tickers_order,
            cms_sector_info=cms_sector_info,
            mu_by_rd=mu_by_rd,
        )
        raw_equity_frames[family] = equ_raw
        p5 = _cms_panel5_check(w_by_rd, tickers_order, cms_sector_info)
        panel5_per_family[family] = p5
        print(f"      Panel 5 sector drift: +{p5['max_pos_drift_pp']:.3f} / {p5['max_neg_drift_pp']:.3f} pp "
              f"violations={p5['violations_n']} violated_flag={p5['violated_flag']}")
        assert not p5["violated_flag"], f"CMS ±3pp sector drift violated for {family}: +{p5['max_pos_drift_pp']:.3f}/{p5['max_neg_drift_pp']:.3f}"
        for rd in sorted(w_by_rd.keys()):
            w = w_by_rd[rd]
            max_w = float(np.nanmax(w))
            assert max_w <= WEIGHT_UPPER + 1e-7, f"single-name cap breach for {family} rd={rd}: max={max_w}"
            for t_i, t in enumerate(tickers_order):
                weight_long_rows.append({
                    "rebal_date": rd,
                    "model_family": family,
                    "ticker": t,
                    "weight_pp": float(w[t_i]),
                })
        t = turnover_from_equities_and_weights(equ_raw.index, w_by_rd, close)
        equ_tx = apply_tx_costs_to_equity(equ_raw, t, TURNOVER_ONE_SIDED_BPS)
        txadj_equity_frames[family] = equ_tx.rename(family + "__txadj")
        nb2_tx = _nb2_consistent_metrics(equ_tx, RISK_FREE_ANNUAL, TRADING_DAYS_PER_YEAR)
        nb2_raw = _nb2_consistent_metrics(equ_raw, RISK_FREE_ANNUAL, TRADING_DAYS_PER_YEAR)
        raw_sharpe = _raw_sharpe(equ_raw)
        calmar = float("nan")
        if nb2_tx["max_dd_pct"] < 0 and abs(nb2_tx["max_dd_pct"]) > 1e-8:
            calmar = float(nb2_tx["ann_return_pct"] / 100.0 / (-nb2_tx["max_dd_pct"] / 100.0))
        sortino = _sortino_ratio(equ_tx)
        worst_month_pct = best_month_pct = float("nan")
        m_ret = _monthly_returns(equ_tx)
        if len(m_ret) >= 1:
            worst_month_pct = float(m_ret.min() * 100.0)
            best_month_pct = float(m_ret.max() * 100.0)
        mdd_abs, mdd_date = _rolling_max_drawdown(equ_tx)
        baseline_ew_tx_csv = out_dir / "phase3_baseline_equities_txadj.csv"
        beat_1n = float("nan")
        if baseline_ew_tx_csv.exists():
            ew_series = pd.read_csv(baseline_ew_tx_csv, index_col=0, parse_dates=True).get("equal_weight__txadj")
            if ew_series is not None:
                ew_nb2 = _nb2_consistent_metrics(ew_series.sort_index(), RISK_FREE_ANNUAL, TRADING_DAYS_PER_YEAR)
                beat_1n = (nb2_tx["sharpe_txadj"] - ew_nb2["sharpe_txadj"]) * 10000.0
        ann_turnover_bps = float("nan")
        if len(t.dropna()) >= 2:
            ann_turnover_bps = float(t.mean() * 252.0 * 10000.0)
        summary_rows.append({
            "strategy": ML_FAMILY_PRETTY.get(family, family),
            "strategy_key": family,
            "ann_return_pct": nb2_tx["ann_return_pct"],
            "ann_vol": nb2_tx["ann_vol"],
            "sharpe_txadj": nb2_tx["sharpe_txadj"],
            "sharpe_raw": nb2_raw["sharpe_txadj"] if np.isfinite(nb2_raw["sharpe_txadj"]) else float(raw_sharpe),
            "max_dd_pct": nb2_tx["max_dd_pct"],
            "total_turnover_ann_bps": ann_turnover_bps,
            "calmar": calmar,
            "sortino": sortino,
            "n_rebalance": len(rebal_dts_valid),
            "n_days": int(len(equ_raw)),
            "beat_1n_sharpe_bps": beat_1n,
            "worst_month_pct": worst_month_pct,
            "best_month_pct": best_month_pct,
        })
        print(f"      OK ann={nb2_tx['ann_return_pct']:+.2f}% vol={nb2_tx['ann_vol']:.3f} "
              f"S_tx={nb2_tx['sharpe_txadj']:+.3f} MDD={nb2_tx['max_dd_pct']:+.2f}% "
              f"turnover_bps_ann={ann_turnover_bps:.0f}")

        pd.Series(equ_raw.values, index=equ_raw.index, name=family).to_csv(
            out_dir / f"phase4_{family}_equity_raw.csv"
        )
        pd.Series(equ_tx.values, index=equ_tx.index, name=family).to_csv(
            out_dir / f"phase4_{family}_equity_txadj.csv"
        )

    pd.DataFrame(weight_long_rows).to_csv(out_dir / "phase4_ml_weights.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "phase4_ml_summary.csv", index=False)
    print(f"\n[ok  ] 6 equity CSVs (raw+txadj × 3 families), weights rows={len(weight_long_rows)}, summary rows={len(summary_df)}")

    # Step 4.6 verdict
    baseline_summary_csv = out_dir / "phase3_baseline_summary.csv"
    if baseline_summary_csv.exists():
        p3 = pd.read_csv(baseline_summary_csv)
        strategy_map_pretty = {
            "equal_weight": "Equal Weight (1/N)",
            "freefloat_proxy": "FreeFloat Proxy",
            "min_variance": "Min Variance (LW Σ)",
            "risk_parity": "Risk Parity (LW Σ)",
            "classic_max_sharpe": "Classic Max Sharpe (LW Σ, ±3pp)",
        }
        rows = []
        ew_sharpe_tx = float("nan")
        cms_sharpe_tx = float("nan")
        for r in p3.itertuples(index=False):
            s = str(r.strategy)
            st = strategy_map_pretty.get(s, s)
            shtx = float(r.sharpe_txadj) if "sharpe_txadj" in p3.columns else float("nan")
            if s == "equal_weight":
                ew_sharpe_tx = shtx
            if s == "classic_max_sharpe":
                cms_sharpe_tx = shtx
            rows.append({"strategy": st, "strategy_key": s, "sharpe_txadj": shtx, "source": "baseline"})
        for r in summary_df.itertuples(index=False):
            shtx = float(r.sharpe_txadj)
            rows.append({
                "strategy": r.strategy,
                "strategy_key": r.strategy_key,
                "sharpe_txadj": shtx,
                "source": "ml",
            })
        verdict_df = pd.DataFrame(rows)
        verdict_df = verdict_df.sort_values("sharpe_txadj", ascending=False).reset_index(drop=True)
        verdict_df["rank"] = np.arange(1, len(verdict_df) + 1)
        if np.isfinite(ew_sharpe_tx):
            verdict_df["Δ_sharpe_vs_1n_bps"] = (verdict_df["sharpe_txadj"] - ew_sharpe_tx) * 10000.0
        else:
            verdict_df["Δ_sharpe_vs_1n_bps"] = float("nan")
        if np.isfinite(cms_sharpe_tx):
            verdict_df["Δ_sharpe_vs_cms_bps"] = (verdict_df["sharpe_txadj"] - cms_sharpe_tx) * 10000.0
        else:
            verdict_df["Δ_sharpe_vs_cms_bps"] = float("nan")
        verdict_df["beat_50bps_threshold_flag"] = verdict_df["Δ_sharpe_vs_1n_bps"].apply(
            lambda v: bool(np.isfinite(v) and v >= 50.0)
        )
        verdict_df.to_csv(out_dir / "phase4_ml_vs_baseline_verdict.csv", index=False)

        print("\n===== PHASE 4 ML vs BASELINE VERDICT =====")
        print(f"{'Rank':<5} {'Strategy':<40} {'Sharpe(tx)':>10} {'Δ vs 1/N (bps)':>15}")
        for r in verdict_df.itertuples(index=False):
            st = str(r.strategy)
            print(f"{int(r.rank):<5} {st[:40]:<40} {r.sharpe_txadj:>10.4f} {r.Δ_sharpe_vs_1n_bps:>15.1f}")
        any_flag = bool(verdict_df["beat_50bps_threshold_flag"].any())
        print()
        if any_flag:
            print("[POSITIVE SIGNAL → Phase 5 MC investment justified] "
                  f"(at least 1 ML strategy beats 1/N by ≥ +50 bps-Sharpe)")
        else:
            print("[FLAT → Phase 5 MC still mandatory, per CONTEXT 3-layer defense]")
        print("[DSR/PBO deferred to Phase 6: point-estimate deltas ARE NOT the final claim.]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
