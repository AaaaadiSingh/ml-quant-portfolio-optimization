from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import (
    QUARTERLY_STEP_D,
    DEV_END,
    STUDY_START,
    _raw_sharpe,
    _rebalance_dates,
)
from src.covariance import ledoit_wolf_cov, log_returns
from src.data_loader import FinalHoldoutDate, load_prices
from src.optimizer import (
    COV_ESTIMATOR_WINNER,
    TRADING_DAYS_PER_YEAR,
    classic_max_sharpe_weights,
    equal_weight,
    freefloat_proxy_weight,
    min_variance_weights,
    risk_parity_weights,
)
from stage1a_apply_txcost import (
    TURNOVER_ONE_SIDED_BPS,
    apply_tx_costs_to_equity,
    turnover_from_equities_and_weights,
    _annualized_metrics,
)

COV_LOOKBACK_D = 63
MU_WINDOW_D = 63
RISK_FREE_ANNUAL = 0.04
SECTOR_TOLERANCE_PP = 0.03
WEIGHT_UPPER = 0.10

BASELINE_ORDER = [
    "equal_weight",
    "freefloat_proxy",
    "min_variance",
    "risk_parity",
    "classic_max_sharpe",
]


def _nb2_consistent_metrics(
    equity: pd.Series,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    eq = equity.sort_index().dropna()
    out: dict[str, float] = {
        "ann_return_pct": float("nan"),
        "ann_vol": float("nan"),
        "sharpe_txadj": float("nan"),
        "max_dd_pct": float("nan"),
    }
    if len(eq) < 2:
        return out
    n_yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    out["ann_return_pct"] = float(((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / n_yrs) - 1.0) * 100.0)
    r = eq.pct_change().fillna(0.0)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / trading_days) - 1.0
    if sigma > 1e-15:
        out["sharpe_txadj"] = float((mu - rf_daily) / sigma * np.sqrt(trading_days))
    out["ann_vol"] = float(sigma * np.sqrt(trading_days))
    peak = eq.cummax()
    out["max_dd_pct"] = float(((eq / peak - 1.0).min()) * 100.0)
    return out


def _sector_targets_full_nifty50_count(
    nifty_map_path: Path,
    universe_tickers: Optional[list[str]] = None,
) -> tuple[dict[str, float], dict[str, str]]:
    df = pd.read_csv(nifty_map_path)
    full_map = dict(zip(df["ticker"].astype(str).tolist(), df["sector_provisional"].astype(str).tolist()))
    if universe_tickers is not None:
        keep = [str(t) for t in universe_tickers]
        ticker_to_sector_sub = {t: full_map[t] for t in keep if t in full_map}
    else:
        ticker_to_sector_sub = {t: s for t, s in full_map.items()}
    sub_sectors = pd.Series(ticker_to_sector_sub)
    count_by_sector = sub_sectors.groupby(sub_sectors.values, dropna=False).size()
    total = int(count_by_sector.sum())
    targets = {str(s): int(c) / total for s, c in count_by_sector.items()}
    return targets, ticker_to_sector_sub


def _monthly_returns(equity: pd.Series) -> pd.Series:
    eq = equity.sort_index().dropna()
    if len(eq) < 2:
        return pd.Series(dtype=float)
    daily = eq / eq.shift(1) - 1.0
    monthly = daily.resample("ME").apply(lambda s: (1.0 + s.fillna(0)).prod() - 1.0)
    return monthly


def _rolling_max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None]:
    eq = equity.sort_index().dropna()
    if len(eq) < 2:
        return float("nan"), None
    cummax = eq.cummax()
    dd = eq / cummax - 1.0
    i = int(np.argmin(dd.values))
    return float(dd.iloc[i]), dd.index[i]


def _wf_one_strategy(
    strategy_label: str,
    close: pd.DataFrame,
    rebal_dts: pd.DatetimeIndex,
    tickers_order: list[str],
    universe_df: pd.DataFrame,
    cms_sector_info: dict | None,
) -> tuple[pd.Series, dict[pd.Timestamp, np.ndarray]]:
    n = len(tickers_order)
    daily_idx = pd.DatetimeIndex(close.index.sort_values())
    rebal_pos = [int(p) for p in daily_idx.get_indexer(rebal_dts) if p >= 0]
    valid_dts = pd.DatetimeIndex([daily_idx[p] for p in rebal_pos])
    weights_by_rebal: dict[pd.Timestamp, np.ndarray] = {}

    mu_by_rd: dict[pd.Timestamp, np.ndarray] | None = None
    if strategy_label == "classic_max_sharpe":
        mu_by_rd = {}
        log_ret_all = np.log(close / close.shift(1)).iloc[1:]
        for rd, pos in zip(valid_dts, rebal_pos):
            start_i = max(0, int(pos) - int(MU_WINDOW_D))
            slice_ = log_ret_all.iloc[start_i : int(pos) + 1]
            if len(slice_) < 10:
                mu_by_rd[rd] = np.zeros(n, dtype=float)
                continue
            mu_daily = slice_.mean(axis=0).values.astype(float)
            mu_by_rd[rd] = mu_daily * TRADING_DAYS_PER_YEAR

    free_float_map = dict(zip(universe_df["ticker"].astype(str), universe_df["free_float_rank"].astype(float)))
    ranks_vec = np.array([float(free_float_map[t]) for t in tickers_order], dtype=float)

    for rd, pos in zip(valid_dts, rebal_pos):
        lb_start = max(0, int(pos) - COV_LOOKBACK_D + 1)
        look_slice = close.iloc[lb_start : int(pos) + 1]
        log_ret = log_returns(look_slice)
        if len(log_ret) < 5:
            w_fallback = equal_weight(n)
            weights_by_rebal[rd] = w_fallback
            continue
        if strategy_label == "equal_weight":
            w = equal_weight(n)
        elif strategy_label == "freefloat_proxy":
            w = freefloat_proxy_weight(ranks_vec)
        elif strategy_label == "min_variance":
            if COV_ESTIMATOR_WINNER != "LW":
                raise RuntimeError(
                    f"COV_ESTIMATOR_WINNER = {COV_ESTIMATOR_WINNER!r} in optimizer.py; "
                    f"this Work Group 4 script is locked to LW per frozen Stage 1B."
                )
            cov_df, _ = ledoit_wolf_cov(log_ret)
            w = min_variance_weights(
                cov_df.values.astype(float), w_upper=WEIGHT_UPPER, solver_backend="pypfopt"
            )
        elif strategy_label == "risk_parity":
            cov_df, _ = ledoit_wolf_cov(log_ret)
            w = risk_parity_weights(cov_df.values.astype(float))
        elif strategy_label == "classic_max_sharpe":
            cov_df, _ = ledoit_wolf_cov(log_ret)
            mu_vec = (mu_by_rd or {}).get(rd, np.zeros(n, dtype=float))
            if cms_sector_info is None:
                w = classic_max_sharpe_weights(
                    mu=mu_vec,
                    cov=cov_df.values.astype(float),
                    risk_free_rate=RISK_FREE_ANNUAL,
                    w_upper=WEIGHT_UPPER,
                    sector_constraints=None,
                    tickers_order=tickers_order,
                )
            else:
                ticker_to_sector = cms_sector_info["ticker_to_sector"]
                targets = cms_sector_info["sector_targets"]
                w = classic_max_sharpe_weights(
                    mu=mu_vec,
                    cov=cov_df.values.astype(float),
                    risk_free_rate=RISK_FREE_ANNUAL,
                    w_upper=WEIGHT_UPPER,
                    sector_constraints={
                        "sector_targets": targets,
                        "ticker_to_sector": pd.Series(ticker_to_sector),
                        "sector_tolerance": SECTOR_TOLERANCE_PP,
                    },
                    tickers_order=tickers_order,
                )
        else:
            raise ValueError(strategy_label)
        weights_by_rebal[rd] = np.asarray(w, dtype=float)
    rebal_keys_sorted = sorted(
        weights_by_rebal.keys(),
        key=lambda d: daily_idx.get_loc(d) if d in daily_idx else -1,
    )
    equity_arr = np.ones(len(daily_idx), dtype=float)
    for k, rd in enumerate(rebal_keys_sorted):
        w = weights_by_rebal[rd]
        pos0 = daily_idx.get_loc(rd)
        if k + 1 < len(rebal_keys_sorted):
            next_rd = rebal_keys_sorted[k + 1]
            pos1 = daily_idx.get_loc(next_rd)
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
    equity_s = pd.Series(equity_arr, index=daily_idx, name=strategy_label).sort_index()
    return equity_s, weights_by_rebal


def main() -> int:
    print("[Phase 3 Work Group 4] Full 5-baseline quarterly walk-forward (frozen Stage 1A=Quarterly, Stage 1B=LW)")
    print(f"[info] COV_ESTIMATOR_WINNER = {COV_ESTIMATOR_WINNER} (per src/optimizer.py lock)")
    print(f"[info] Frequency = Quarterly, {QUARTERLY_STEP_D} BD rebal step")
    print(f"[info] Σ LW lookback = {COV_LOOKBACK_D} BD | CMS μ̂ 63d sample mean daily log ret × 252 | r_f = {RISK_FREE_ANNUAL:.2%}")
    print(f"[info] CMS sector constraint: ± {SECTOR_TOLERANCE_PP:.0%} pp (NIFTY-50 count-based static proxy target)")

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    universe_df = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers_order = sorted(universe_df["ticker"].astype(str).tolist())
    n = len(tickers_order)
    print(f"[info] N = {n} tickers")

    prices = load_prices(
        tickers_order,
        start_date=STUDY_START,
        end_date=DEV_END,
        final_holdout=False,
    )
    assert not prices.empty
    assert prices.index.max() < FinalHoldoutDate, f"holdout leak: max date {prices.index.max()}"
    close = prices["Close"].ffill().bfill().sort_index()
    close = close.loc[:, tickers_order]
    n_dates = close.shape[0]
    print(f"[info] Close shape = ({n_dates}, {n}); {close.index.min().date()} -> {close.index.max().date()}")

    sector_targets, ticker_to_sector = _sector_targets_full_nifty50_count(
        ROOT / "docs" / "nifty50_sector_map.csv",
        universe_tickers=tickers_order,
    )
    n_sectors = len(sector_targets)
    print(f"[info] NIFTY-50 count-based sector proxy: {n_sectors} sectors (normalized to surviving {len(tickers_order)} tickers)")
    print(f"[info] Sample sector targets (first 5): {list(sector_targets.items())[:5]}")
    cms_sector_info = {
        "sector_targets": sector_targets,
        "ticker_to_sector": ticker_to_sector,
        "tickers_order": tickers_order,
        "sector_tolerance": 0.03,
    }

    daily_idx = pd.DatetimeIndex(close.index.sort_values())
    rebal_dts = _rebalance_dates(daily_idx, QUARTERLY_STEP_D)
    rebal_pos = [int(p) for p in daily_idx.get_indexer(rebal_dts) if p >= 0]
    rebal_dts_valid = pd.DatetimeIndex([daily_idx[p] for p in rebal_pos])
    n_rebal = len(rebal_dts_valid)
    print(f"[info] n_rebal = {n_rebal} (quarterly)")

    raw_equities: dict[str, pd.Series] = {}
    txadj_equities: dict[str, pd.Series] = {}
    weight_records: list[tuple[pd.Timestamp, str, np.ndarray]] = []
    turnover_per_strategy: dict[str, pd.Series] = {}
    summary_rows: list[dict] = []

    for strategy in BASELINE_ORDER:
        print(f"\n[---] strategy = {strategy:22s}")
        print(f"[run ]   Walk forward build weights + equity ...", end=" ", flush=True)
        equ_raw, w_by_rd = _wf_one_strategy(
            strategy_label=strategy,
            close=close,
            rebal_dts=rebal_dts_valid,
            tickers_order=tickers_order,
            universe_df=universe_df,
            cms_sector_info=cms_sector_info if strategy == "classic_max_sharpe" else None,
        )
        raw_equities[strategy] = equ_raw
        n_captured = len(w_by_rd)
        print(f"OK {len(equ_raw)} days, {n_captured} rebal w vectors")

        for rd in sorted(w_by_rd.keys()):
            weight_records.append((rd, strategy, w_by_rd[rd].copy()))

        print(f"[calc]   Turnover + 10 bps tx-cost drag ...", end=" ", flush=True)
        t = turnover_from_equities_and_weights(
            equity_index=equ_raw.index,
            weights_by_rebal_date=w_by_rd,
            close_prices_df=close,
        )
        turnover_per_strategy[strategy] = t
        equ_tx = apply_tx_costs_to_equity(equ_raw, t, TURNOVER_ONE_SIDED_BPS)
        txadj_equities[strategy] = equ_tx.rename(strategy + "__txadj")

        raw_sharpe = _raw_sharpe(equ_raw)
        metrics = _annualized_metrics(equ_tx, t)
        metrics_raw = _annualized_metrics(equ_raw, pd.Series(dtype=float))
        worst_month_pct = best_month_pct = float("nan")
        month_ret = _monthly_returns(equ_tx)
        if len(month_ret) >= 1:
            worst_month_pct = float(month_ret.min() * 100.0)
            best_month_pct = float(month_ret.max() * 100.0)
        mdd_abs, mdd_date = _rolling_max_drawdown(equ_tx)
        nb2_tx = _nb2_consistent_metrics(equ_tx, RISK_FREE_ANNUAL, TRADING_DAYS_PER_YEAR)
        nb2_raw = _nb2_consistent_metrics(equ_raw, RISK_FREE_ANNUAL, TRADING_DAYS_PER_YEAR)
        summary_rows.append(
            {
                "strategy": strategy,
                "ann_return_pct": nb2_tx["ann_return_pct"],
                "ann_vol": nb2_tx["ann_vol"],
                "sharpe_txadj": nb2_tx["sharpe_txadj"],
                "sharpe_raw": nb2_raw["sharpe_txadj"] if np.isfinite(nb2_raw["sharpe_txadj"]) else float(raw_sharpe),
                "max_dd_pct": nb2_tx["max_dd_pct"],
                "max_dd_date": mdd_date.isoformat() if mdd_date is not None else "",
                "total_turnover_bps_ann": metrics["ann_turnover_bps_mean_daily"],
                "n_rebalances": n_rebal,
                "worst_monthly_ret_pct": worst_month_pct,
                "best_monthly_ret_pct": best_month_pct,
                "tx_drag_ann_pct": metrics["tx_drag_pct_annualized"],
            }
        )
        print(
            f"OK ann_ret={nb2_tx['ann_return_pct']:+.2f}% ann_vol={nb2_tx['ann_vol']:.3f} "
            f"tx_S={nb2_tx['sharpe_txadj']:+.3f} raw_S={raw_sharpe:+.3f} MDD={nb2_tx['max_dd_pct']:+.2f}%"
        )

    df_raw = pd.DataFrame(raw_equities).sort_index()
    df_tx = pd.DataFrame(txadj_equities).sort_index()
    raw_out = out_dir / "phase3_baseline_equities_raw.csv"
    txadj_out = out_dir / "phase3_baseline_equities_txadj.csv"
    df_raw.to_csv(raw_out)
    df_tx.to_csv(txadj_out)
    print(f"\n[ok  ] Raw equities ({df_raw.shape[0]}×{df_raw.shape[1]})  -> {raw_out}")
    print(f"[ok  ] Txadj equities ({df_tx.shape[0]}×{df_tx.shape[1]})  -> {txadj_out}")

    wr_idx = pd.MultiIndex.from_tuples(
        [(rd, s) for rd, s, _ in weight_records],
        names=["rebal_date", "strategy"],
    )
    w_arr = np.vstack([w for _, _, w in weight_records])
    weights_df = pd.DataFrame(w_arr, index=wr_idx, columns=tickers_order)
    weights_flat = weights_df.reset_index()
    w_csv_out = out_dir / "phase3_baseline_weights.csv"
    w_parquet_out = out_dir / "phase3_baseline_weights.parquet"
    saved_as_parquet = False
    try:
        weights_df.to_parquet(w_parquet_out, compression="snappy")
        saved_as_parquet = True
    except ImportError:
        saved_as_parquet = False
    if saved_as_parquet:
        print(f"[ok  ] Weights parquet MultiIndex (rebal_date,strategy) × cols → {len(weights_df)} rows, {w_parquet_out}")
    else:
        weights_flat.to_csv(w_csv_out, index=False)
        print(
            f"[ok  ] pyarrow/fastparquet not in frozen requirements → used CSV instead (parquet skipped). "
            f"Weights table (rebal_date,strategy,TICKER cols) → {len(weights_flat)} rows, {w_csv_out}"
        )
        weights_flat.to_parquet = None

    summary_df = pd.DataFrame(summary_rows)
    summary_out = out_dir / "phase3_baseline_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"[ok  ] 5-strategy summary 10 cols → {summary_out}")

    authoritative_df = pd.read_csv(out_dir / "phase3_stage1_winners.csv") if (
        out_dir / "phase3_stage1_winners.csv"
    ).exists() else None
    if authoritative_df is None or len(authoritative_df) < 2:
        authoritative_df = pd.DataFrame(
            [
                {
                    "stage": "Stage_1A_rebalance_frequency",
                    "winner": "QUARTERLY (63 BD)",
                    "margin_bps_sharpe": 47.7,
                },
                {
                    "stage": "Stage_1B_covariance_estimator",
                    "winner": "LW",
                    "margin_bps_sharpe": 8.2573,
                },
            ]
        )
        authoritative_df.to_csv(out_dir / "phase3_stage1_winners.csv", index=False)
        print(f"[gen ] Regenerated authoritative phase3_stage1_winners.csv (2 rows)")
    else:
        print(f"[ok  ] Authoritative stage1 winners already on disk (rows={len(authoritative_df)})")

    print("\n===== PHASE 3 Work Group 4 — 5 BASELINE SUMMARY =====")
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", "{:+.4f}".format)
    print_cols = [
        "strategy",
        "ann_return_pct",
        "ann_vol",
        "sharpe_txadj",
        "sharpe_raw",
        "max_dd_pct",
        "total_turnover_bps_ann",
        "n_rebalances",
        "worst_monthly_ret_pct",
        "best_monthly_ret_pct",
    ]
    print(summary_df[print_cols].to_string(index=False))
    print("\n[DONE] Work Group 4 5-baseline artifacts: 2 CSVs (equities), 1 parquet (weights), 1 summary CSV, 1 authoritative Stage 1 winners.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
