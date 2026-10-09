from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, objective_functions

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import (
    QUARTERLY_STEP_D,
    DEV_END,
    STUDY_START,
    _raw_sharpe,
    _rebalance_dates,
)
from src.backtest_engine import WalkForwardBacktestEngine
from src.covariance import ledoit_wolf_cov, pca_factor_cov
from src.data_loader import FinalHoldoutDate, load_prices
from src.optimizer import min_variance_weights, risk_parity_weights

from stage1a_apply_txcost import (
    TURNOVER_ONE_SIDED_BPS,
    apply_tx_costs_to_equity,
    turnover_from_equities_and_weights,
    _annualized_metrics,
)

TIE_THRESHOLD_SHARPE = 0.02
CMS_MU_WINDOW_D = 63
CMS_RISK_FREE_ANNUAL = 0.04
TRADING_DAYS_PER_YEAR = 252
WEIGHT_UPPER = 0.10
COV_LOOKBACK_D = 63
ESTIMATORS = [
    ("ledoit_wolf", {}),
    ("pca_factor", {"min_explained": 0.85, "max_k": 15}),
]
STRATEGIES = ["min_variance", "risk_parity", "classic_max_sharpe"]


def _cms_weights_pypfopt(
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: list[str],
    w_upper: float = WEIGHT_UPPER,
    rf_annual: float = CMS_RISK_FREE_ANNUAL,
) -> np.ndarray:
    n = len(tickers)
    cov_df = pd.DataFrame(cov, index=tickers, columns=tickers)
    mu_series = pd.Series(mu.astype(float), index=tickers)
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    ef = EfficientFrontier(mu_series - rf_daily, cov_df, weight_bounds=(0.0, w_upper))
    ef.add_objective(objective_functions.L2_reg, gamma=1e-6)
    try:
        raw = ef.max_sharpe(risk_free_rate=0.0)
    except Exception:
        return np.full(n, 1.0 / n, dtype=float)
    w = np.array([float(raw[t]) for t in tickers], dtype=float)
    if n * w_upper < 1.0 - 1e-9:
        pass
    s = float(w.sum())
    if abs(s) < 1e-12:
        return np.full(n, 1.0 / n, dtype=float)
    w = w / s
    w = np.clip(w, 0.0, w_upper + 1e-9)
    s2 = float(w.sum())
    if abs(s2) < 1e-12:
        return np.full(n, 1.0 / n, dtype=float)
    w = w / s2
    return w


def _rolling_mu_annualized(
    close_df: pd.DataFrame,
    rebal_positions: list[int],
    window_d: int = CMS_MU_WINDOW_D,
) -> dict[pd.Timestamp, np.ndarray]:
    tickers = list(close_df.columns)
    n = len(tickers)
    log_ret = np.log(close_df / close_df.shift(1)).iloc[1:]
    mu_by_rebal: dict[pd.Timestamp, np.ndarray] = {}
    for rd, pos in zip(close_df.index[rebal_positions].tolist(), rebal_positions):
        start = max(1, int(pos) - int(window_d))
        slice_ = log_ret.iloc[start : int(pos) + 1]
        if len(slice_) < max(10, window_d // 4):
            mu_by_rebal[rd] = np.zeros(n, dtype=float)
            continue
        mu_daily = slice_.mean(axis=0).values.astype(float)
        mu_ann = mu_daily * TRADING_DAYS_PER_YEAR
        if not np.isfinite(mu_ann).all():
            mu_ann = np.where(np.isfinite(mu_ann), mu_ann, 0.0)
        mu_by_rebal[rd] = mu_ann
    return mu_by_rebal


def _wf_one_estimator_one_strategy(
    estimator_name: str,
    estimator_kwargs: dict,
    strategy: str,
    close: pd.DataFrame,
    rebal_dts: pd.DatetimeIndex,
    mu_by_rebal: dict[pd.Timestamp, np.ndarray] | None,
) -> tuple[pd.Series, dict[pd.Timestamp, np.ndarray], list[dict]]:
    tickers = list(close.columns)
    n = len(tickers)
    daily_idx = pd.DatetimeIndex(close.index.sort_values())
    rebal_pos = daily_idx.get_indexer(rebal_dts)
    valid_idx = [int(p) for p in rebal_pos if p >= 0]
    valid_dts = daily_idx[valid_idx]
    weights_by_rebal: dict[pd.Timestamp, np.ndarray] = {}
    meta_records: list[dict] = []
    for rd, pos in zip(valid_dts, valid_idx):
        lb_start = max(0, int(pos) - COV_LOOKBACK_D + 1)
        look_slice = close.iloc[lb_start : int(pos) + 1]
        log_ret = np.log(look_slice / look_slice.shift(1)).iloc[1:]
        if len(log_ret) < 5:
            weights_by_rebal[rd] = np.full(n, 1.0 / n)
            continue
        try:
            if estimator_name == "ledoit_wolf":
                cov_df, meta = ledoit_wolf_cov(log_ret, **estimator_kwargs)
            elif estimator_name == "pca_factor":
                cov_df, meta = pca_factor_cov(log_ret, **estimator_kwargs)
            else:
                raise ValueError(estimator_name)
        except Exception:
            weights_by_rebal[rd] = np.full(n, 1.0 / n)
            continue
        meta_records.append(meta)
        cov_np = cov_df.values.astype(float)
        if strategy == "min_variance":
            try:
                w = min_variance_weights(cov_np, w_upper=WEIGHT_UPPER, solver_backend="pypfopt")
            except Exception:
                w = np.full(n, 1.0 / n)
        elif strategy == "risk_parity":
            try:
                w = risk_parity_weights(cov_np)
            except Exception:
                w = np.full(n, 1.0 / n)
        elif strategy == "classic_max_sharpe":
            mu = (mu_by_rebal or {}).get(rd)
            if mu is None:
                mu = np.zeros(n, dtype=float)
            try:
                w = _cms_weights_pypfopt(mu, cov_np, tickers)
            except Exception:
                w = np.full(n, 1.0 / n)
        else:
            raise ValueError(strategy)
        weights_by_rebal[rd] = np.asarray(w, dtype=float)
    rebal_keys_sorted = sorted(weights_by_rebal.keys(), key=lambda d: daily_idx.get_loc(d) if d in daily_idx else -1)
    if not rebal_keys_sorted:
        return pd.Series(np.ones(len(daily_idx)), index=daily_idx, name=f"{estimator_name}_{strategy}"), weights_by_rebal, meta_records
    engine = WalkForwardBacktestEngine(close, bps_per_turnover=TURNOVER_ONE_SIDED_BPS)
    equity_s, _, _, _ = engine.simulate(weights_by_rebal)
    equity_s.name = f"{estimator_name}_{strategy}"
    return equity_s.sort_index(), weights_by_rebal, meta_records


def _pca_K_average(meta_list: list[dict]) -> float:
    vals = [m["K"] for m in meta_list if m.get("method") == "pca_factor" and "K" in m]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    print("[Stage 1B] Ledoit-Wolf vs PCA-factor covariance-estimator decision")
    print(f"[info] Stage 1A winner frequency = QUARTERLY ({QUARTERLY_STEP_D} BD rebal)")
    print(f"[info] Σ lookback = {COV_LOOKBACK_D} BD; μ CMS window = {CMS_MU_WINDOW_D} BD; r_f = {CMS_RISK_FREE_ANNUAL:.2%} annual")
    print(f"[info] 3 cov-sensitive strategies: {STRATEGIES} (Equal Weight EXCLUDED per D10 — cov-agnostic)")
    print(f"[info] tx-cost rule: {TURNOVER_ONE_SIDED_BPS:.0f} bps / one-sided turnover")

    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    n = len(tickers)
    print(f"[info] N = {n} tickers")

    prices = load_prices(
        tickers,
        start_date=STUDY_START,
        end_date=DEV_END,
        final_holdout=False,
    )
    assert not prices.empty
    assert prices.index.max() < FinalHoldoutDate, "holdout leak detected"
    close = prices["Close"].ffill().bfill().sort_index()
    print(f"[info] Close shape = {close.shape}; {close.index.min().date()} -> {close.index.max().date()}")

    daily_idx = pd.DatetimeIndex(close.index.sort_values())
    rebal_dts = _rebalance_dates(daily_idx, QUARTERLY_STEP_D)
    rebal_pos = [int(p) for p in daily_idx.get_indexer(rebal_dts) if p >= 0]
    rebal_dts = pd.DatetimeIndex([daily_idx[p] for p in rebal_pos])
    print(f"[info] n_rebal = {len(rebal_dts)} (quarterly)")

    mu_by_rebal = _rolling_mu_annualized(close, rebal_pos, window_d=CMS_MU_WINDOW_D)
    print(f"[ok  ] CMS rolling μ: {len(mu_by_rebal)} rebal μ vectors captured")

    results_rows: list[dict] = []
    equity_frames: dict[str, pd.Series] = {}
    pca_metas: list[dict] = []
    lw_metas: list[dict] = []

    print("\n===== §16: ONE-TIME SYNTHETIC SOLVER CROSS-CHECK (MANDATORY) =====")
    rng = np.random.default_rng(20240104)
    T_synth = 252
    true_corr_np = rng.uniform(0.03, 0.35, size=(n, n))
    true_corr_np = (true_corr_np + true_corr_np.T) / 2.0
    np.fill_diagonal(true_corr_np, 1.0)
    ev, Q = np.linalg.eigh(true_corr_np)
    ev = np.where(ev > 1e-8, ev, 1e-8)
    true_corr_np = Q @ np.diag(ev) @ Q.T
    d = 1.0 / np.sqrt(np.diag(true_corr_np))
    true_corr_np = (true_corr_np * d[:, None]) * d[None, :]
    vols = rng.uniform(0.008, 0.022, size=n)
    true_cov_np = np.diag(vols) @ true_corr_np @ np.diag(vols)
    synth_rets = rng.multivariate_normal(np.zeros(n), true_cov_np, size=T_synth)
    synth_df = pd.DataFrame(synth_rets, columns=tickers)
    synth_df.index = pd.date_range("2020-01-01", periods=T_synth, freq="B")
    try:
        cov_lw, _ = ledoit_wolf_cov(synth_df)
        cov_pca, _ = pca_factor_cov(synth_df, min_explained=0.85, max_k=15)
    except Exception as exc:
        print(f"[FAIL] synthetic cov estimator crashed: {exc!s}")
        return 3
    for est_label, cov_ in [("ledoit_wolf", cov_lw.values), ("pca_factor", cov_pca.values)]:
        w_pypfopt = min_variance_weights(cov_, w_upper=WEIGHT_UPPER, solver_backend="pypfopt")
        w_cvxpy = min_variance_weights(cov_, w_upper=WEIGHT_UPPER, solver_backend="cvxpy")
        rms = float(np.sqrt(np.mean((w_pypfopt - w_cvxpy) ** 2)))
        status = "PASS" if rms <= 5e-4 else "FAIL"
        print(f"[ok  ] Σ={est_label:12s}  N=252 syn rets  pypfopt-vs-cvxpy RMS(Δw) = {rms:.2e}  [{status}]")
        if status == "FAIL":
            print("[STOP] Solver cross-check FAILED on synthetic data — debug solvers BEFORE touching empirical data (per spec §16).")
            return 4
    print("[ok  ] §16 synthetic cross-check: PASSED — proceeding to empirical WF.\n")

    for estimator_name, estimator_kwargs in ESTIMATORS:
        print(f"\n[---] Σ ESTIMATOR = {estimator_name}")
        for strategy in STRATEGIES:
            print(f"[run ]   strategy = {strategy:22s} WF ...", end=" ", flush=True)
            equ, w_by_rd, metas_out = _wf_one_estimator_one_strategy(
                estimator_name=estimator_name,
                estimator_kwargs=estimator_kwargs,
                strategy=strategy,
                close=close,
                rebal_dts=rebal_dts,
                mu_by_rebal=mu_by_rebal,
            )
            if estimator_name == "ledoit_wolf":
                lw_metas.extend(metas_out)
            elif estimator_name == "pca_factor":
                pca_metas.extend(metas_out)
            print(f"OK equity {equ.shape[0]} points")
            print(f"[calc]                          turnover + tx-cost drag ...", end=" ", flush=True)
            turnover = turnover_from_equities_and_weights(
                equity_index=equ.index,
                weights_by_rebal_date=w_by_rd,
                close_prices_df=close,
            )
            equ_txadj = apply_tx_costs_to_equity(equ, turnover, TURNOVER_ONE_SIDED_BPS)
            met = _annualized_metrics(equ_txadj, turnover)
            raw_s = _raw_sharpe(equ)
            col_key = f"{estimator_name}__{strategy}"
            equity_frames[col_key + "__raw"] = equ
            equity_frames[col_key + "__txadj"] = equ_txadj
            results_rows.append(
                {
                    "estimator": estimator_name,
                    "strategy": strategy,
                    "raw_sharpe": float(raw_s) if np.isfinite(raw_s) else float("nan"),
                    "tx_sharpe": met["sharpe"],
                    "ann_turnover_bps": met["ann_turnover_bps_mean_daily"],
                    "tx_drag_ann_pct": met["tx_drag_pct_annualized"],
                }
            )
            print(
                f"OK raw_S={raw_s:+.3f} tx_S={met['sharpe']:+.3f} drag_ann={met['tx_drag_pct_annualized']:+.2f}% "
                f"ann_turn={met['ann_turnover_bps_mean_daily']:.1f} bps"
            )

    res_df = pd.DataFrame(results_rows)
    print("\n===== STAGE 1B 3-STRATEGY RESULTS =====")
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:+.4f}".format)
    print(res_df.to_string(index=False))

    by_est: dict[str, dict] = {}
    for est, grp in res_df.groupby("estimator"):
        s_tx = grp["tx_sharpe"].astype(float).values
        s_turn = grp["ann_turnover_bps"].astype(float).values
        by_est[est] = {
            "mean_3_strat_sharpe": float(np.nanmean(s_tx)),
            "mean_ann_turnover_bps": float(np.nanmean(s_turn)),
            "sharpe_by_strat": dict(zip(grp["strategy"].tolist(), s_tx.tolist())),
        }
    est_labels = list(by_est.keys())
    if len(est_labels) != 2:
        print(f"[FAIL] expected 2 estimators, got {len(est_labels)}")
        return 5
    e1, e2 = est_labels
    m1 = by_est[e1]["mean_3_strat_sharpe"]
    m2 = by_est[e2]["mean_3_strat_sharpe"]
    margin = abs(m1 - m2)
    tie_reasons_used: list[str] = []
    winner = e1 if m1 >= m2 else e2
    if margin < TIE_THRESHOLD_SHARPE:
        tie_reasons_used.append(
            f"Δmean_3_strat={margin:+.4f} < {TIE_THRESHOLD_SHARPE:.2f} → tiebreaker 1 (lower mean ann turnover bps)"
        )
        t1 = by_est[e1]["mean_ann_turnover_bps"]
        t2 = by_est[e2]["mean_ann_turnover_bps"]
        cheaper = e1 if t1 <= t2 else e2
        if cheaper != winner:
            winner = cheaper
            tie_reasons_used.append(
                f"tiebreak 1 winner → {cheaper} (mean_turn_bps={by_est[cheaper]['mean_ann_turnover_bps']:.1f} vs alt {by_est[e2 if cheaper==e1 else e1]['mean_ann_turnover_bps']:.1f})"
            )
        else:
            tie_reasons_used.append("tiebreak 1 confirms current winner (already cheaper or tied)")
    if len(tie_reasons_used) == 0:
        tie_reasons_used.append("direct win (no tiebreaker fired)")

    margin_bps_sharpe = float(margin * 10_000.0)
    print("\n===== STAGE 1B SELECTION =====")
    for e, v in by_est.items():
        print(f"  mean tx-constrained Sharpe (3 cov-sensitive strat) | {e:12s} = {v['mean_3_strat_sharpe']:+.4f}   (mean ann turn={v['mean_ann_turnover_bps']:.1f} bps)")
    print(f"  margin = {margin:+.4f} Sharpe = {margin_bps_sharpe:+.1f} bps-Sharpe")
    for reason in tie_reasons_used:
        print(f"  → {reason}")
    print(f"  *** WINNER Σ ESTIMATOR = {winner.upper()} ***")

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_rows: list[dict] = []
    for est in est_labels:
        v = by_est[est]
        sb = v["sharpe_by_strat"]
        decision_rows.append(
            {
                "estimator": est,
                "mean_3_strat_sharpe": v["mean_3_strat_sharpe"],
                "minvar_sharpe": sb.get("min_variance", float("nan")),
                "cms_sharpe": sb.get("classic_max_sharpe", float("nan")),
                "rp_sharpe": sb.get("risk_parity", float("nan")),
                "mean_ann_turnover_bps": v["mean_ann_turnover_bps"],
                "pca_K_if_applicable": float("nan")
                if est != "pca_factor"
                else _pca_K_average(pca_metas)
                if pca_metas
                else float("nan"),
                "winner_flag": bool(est == winner),
            }
        )
    decision_df = pd.DataFrame(decision_rows)
    decision_out = out_dir / "stage1b_decision.csv"
    decision_df.to_csv(decision_out, index=False)
    print(f"\n[ok  ] Stage 1B decision table (2 rows) → {decision_out}")

    stage1_w1 = {
        "stage": "Stage_1A_rebalance_frequency",
        "winner": "QUARTERLY (63 BD)",
        "margin_bps_sharpe": 47.7,
    }
    stage1_w2 = {
        "stage": "Stage_1B_covariance_estimator",
        "winner": "LW" if winner == "ledoit_wolf" else f"PCA(K≈{np.nanmean([d.get('K',0) for d in pca_metas]) if (winner=='pca_factor' and pca_metas) else '?'})",
        "margin_bps_sharpe": float(margin_bps_sharpe),
    }
    authoritative = pd.DataFrame([stage1_w1, stage1_w2])
    auth_out = out_dir / "phase3_stage1_winners.csv"
    authoritative.to_csv(auth_out, index=False)
    print(f"[ok  ] Authoritative 2-row Stage 1 winners → {auth_out}")

    eq_df = pd.DataFrame(equity_frames).sort_index()
    eq_out = out_dir / "stage1b_wf_equities.csv"
    eq_df.to_csv(eq_out)
    print(f"[ok  ] 12 equity curves (2 est × 3 strat × 2 modes raw/txadj) → {eq_out}")

    print("\n=== Stage 1B — 2-estimator summary ===")
    pd.set_option("display.float_format", "{:+.4f}".format)
    print(decision_df.to_string(index=False))
    print("\n=== Stage 1 authoritative winners (for Phase 4+) ===")
    print(authoritative.to_string(index=False))
    print("\n[DONE] Stage 1B decision finalized and frozen. Next: update optimizer.py COV_ESTIMATOR_WINNER constant, implement CMS (baseline 5) non-stub, then Work Group 4 full 5-baseline WF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
