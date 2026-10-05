from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import (
    COV_WINDOW_D,
    MONTHLY_STEP_D,
    QUARTERLY_STEP_D,
    STUDY_START,
    VOL_WINDOW_D,
    DEV_END,
    _equal_weight,
    _free_float_proxy_weight,
    _rebalance_dates,
    _risk_parity_weights,
    _raw_sharpe,
    _solve_min_variance,
)
from src.data_loader import FinalHoldoutDate, load_prices

TURNOVER_ONE_SIDED_BPS = 10.0
TRADING_DAYS_PER_YEAR = 252

FREQUENCIES: dict[str, int] = {
    "monthly": MONTHLY_STEP_D,
    "quarterly": QUARTERLY_STEP_D,
}

BASELINES: list[str] = ["equal", "freefloat_proxy", "minvar", "riskparity"]

TIE_THRESHOLD_SHARPE = 0.02


def _build_weights_by_rebal(
    close: pd.DataFrame,
    rebal_dts: pd.DatetimeIndex,
    baseline: str,
    universe: pd.DataFrame,
) -> dict[pd.Timestamp, np.ndarray]:
    tickers = list(close.columns)
    n = len(tickers)
    daily_idx_sorted = pd.DatetimeIndex(close.index.sort_values())
    pos = daily_idx_sorted.get_indexer(rebal_dts)
    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    for rd, idx_pos in zip(rebal_dts, pos):
        look_start = max(0, int(idx_pos) - max(COV_WINDOW_D, VOL_WINDOW_D) + 1)
        look_end = int(idx_pos) + 1
        look = close.iloc[look_start:look_end]
        if baseline == "equal":
            w = _equal_weight(n)
        elif baseline == "freefloat_proxy":
            w = _free_float_proxy_weight(tickers, universe)
        elif baseline == "minvar":
            if len(look) < COV_WINDOW_D:
                w = _equal_weight(n)
            else:
                ret = np.log(look / look.shift(1)).dropna()
                if len(ret) < 2 or ret.shape[1] != n:
                    w = _equal_weight(n)
                else:
                    cov = np.cov(ret.values.T, ddof=0)
                    w = _solve_min_variance(cov, n)
        elif baseline == "riskparity":
            if len(look) < VOL_WINDOW_D:
                w = _equal_weight(n)
            else:
                ret = np.log(look / look.shift(1)).dropna()
                if len(ret) < 2 or ret.shape[1] != n:
                    w = _equal_weight(n)
                else:
                    cov_diag = np.var(ret.values, axis=0, ddof=0)
                    w = _risk_parity_weights(cov_diag, n)
        else:
            raise ValueError(f"unknown baseline: {baseline}")
        weights_by_date[rd] = w
    return weights_by_date


def turnover_from_equities_and_weights(
    equity_index: pd.DatetimeIndex,
    weights_by_rebal_date: dict[pd.Timestamp, np.ndarray],
    close_prices_df: pd.DataFrame,
) -> pd.Series:
    sorted_idx = pd.DatetimeIndex(equity_index.sort_values())
    rebal_sorted = sorted(
        [rd for rd in weights_by_rebal_date.keys() if rd in sorted_idx]
    )
    if len(rebal_sorted) < 2:
        idx = pd.DatetimeIndex(rebal_sorted)
        return pd.Series(np.full(len(rebal_sorted), np.nan), index=idx, dtype=float)
    tickers = list(close_prices_df.columns)
    close_aligned = close_prices_df.loc[sorted_idx, tickers].ffill().bfill()
    rebal_positions = sorted_idx.get_indexer(rebal_sorted)
    turnovers: dict[pd.Timestamp, float] = {}
    for k in range(len(rebal_sorted)):
        rd = rebal_sorted[k]
        pos_rd = int(rebal_positions[k])
        close_new = close_aligned.iloc[pos_rd].values.astype(float)
        w_new = np.asarray(weights_by_rebal_date[rd], dtype=float)
        if k == 0:
            turnovers[rd] = 0.0
            continue
        prev_rd = rebal_sorted[k - 1]
        prev_pos = int(rebal_positions[k - 1])
        close_prev = close_aligned.iloc[prev_pos].values.astype(float)
        w_prev = np.asarray(weights_by_rebal_date[prev_rd], dtype=float)
        safe_close = np.where(np.isfinite(close_prev) & (close_prev > 0), close_prev, 1.0)
        per_asset_ret = close_new / safe_close - 1.0
        port_ret = float(np.dot(w_prev, per_asset_ret))
        denom = 1.0 + port_ret
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            denom = 1.0
        w_drifted = w_prev * (1.0 + per_asset_ret) / denom
        w_drifted = np.where(np.isfinite(w_drifted), w_drifted, 0.0)
        w_new_c = np.where(np.isfinite(w_new), w_new, 0.0)
        turn = 0.5 * float(np.sum(np.abs(w_new_c - w_drifted)))
        turnovers[rd] = max(0.0, turn)
    idx = pd.DatetimeIndex(list(turnovers.keys()))
    return pd.Series(list(turnovers.values()), index=idx, name="turnover_one_sided").sort_index()


def apply_tx_costs_to_equity(
    raw_equity_series: pd.Series,
    turnover_series: pd.Series,
    bps_per_turnover: float = TURNOVER_ONE_SIDED_BPS,
) -> pd.Series:
    eq = raw_equity_series.sort_index().copy()
    if len(eq) == 0:
        return eq
    drag_per_rebal = pd.Series(0.0, index=eq.index, dtype=float)
    for rd, turn in turnover_series.items():
        if rd in drag_per_rebal.index and np.isfinite(turn) and turn >= 0.0:
            cost_rate = (bps_per_turnover / 10_000.0) * turn
            drag_per_rebal.loc[rd] = cost_rate
    cumulative_log_drag = (-drag_per_rebal).cumsum()
    log_eq = np.log(np.where(eq.values > 0, eq.values, np.nan))
    adjusted_log = log_eq + cumulative_log_drag.values
    result = pd.Series(
        np.exp(adjusted_log),
        index=eq.index,
        name=f"{raw_equity_series.name}_txadj",
    ).sort_index()
    first_valid = result.first_valid_index()
    if first_valid is not None:
        result.loc[:first_valid] = raw_equity_series.loc[:first_valid].values
    return result


def _annualized_metrics(
    equity: pd.Series,
    turnover: pd.Series,
) -> dict[str, float]:
    eq = equity.dropna().sort_index()
    if len(eq) < 2:
        return {
            "sharpe": float("nan"),
            "ann_turnover_bps_mean_daily": float("nan"),
            "tx_drag_pct_annualized": float("nan"),
            "ann_return": float("nan"),
        }
    sharpe = _raw_sharpe(eq)
    turn_clean = turnover[turnover.index.isin(eq.index)].dropna()
    n_years = max(
        (eq.index.max() - eq.index.min()).days / 365.25,
        1.0 / TRADING_DAYS_PER_YEAR,
    )
    total_turn = float(turn_clean.sum())
    ann_turn_mean_daily_bps = (total_turn / n_years) * 10_000.0
    total_cost = float((turn_clean * (TURNOVER_ONE_SIDED_BPS / 10_000.0)).sum())
    tx_drag_ann_pct = (np.exp(-total_cost / max(n_years, 1e-9)) - 1.0) * 100.0
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    ann_return = (1.0 + total_ret) ** (1.0 / n_years) - 1.0
    return {
        "sharpe": float(sharpe) if np.isfinite(sharpe) else float("nan"),
        "ann_turnover_bps_mean_daily": float(ann_turn_mean_daily_bps)
        if np.isfinite(ann_turn_mean_daily_bps)
        else float("nan"),
        "tx_drag_pct_annualized": float(tx_drag_ann_pct)
        if np.isfinite(tx_drag_ann_pct)
        else float("nan"),
        "ann_return": float(ann_return) * 100.0 if np.isfinite(ann_return) else float("nan"),
    }


def _stage1a_winner(
    rows: list[dict],
) -> tuple[str, float, list[str]]:
    by_freq: dict[str, list[float]] = {}
    for r in rows:
        by_freq.setdefault(r["frequency"], []).append(float(r["tx_sharpe"]))
    means = {f: float(np.nanmean(v)) if len(v) else float("nan") for f, v in by_freq.items()}
    freqs = list(means.keys())
    if len(freqs) != 2:
        raise RuntimeError(f"expected 2 frequencies, got {len(freqs)}: {freqs}")
    f_a, f_b = freqs[0], freqs[1]
    m_a, m_b = means[f_a], means[f_b]
    tie_reasons: list[str] = []
    if np.isnan(m_a) or np.isnan(m_b):
        raise RuntimeError(f"NaN mean Sharpe: {means}")
    winner = f_a if m_a >= m_b else f_b
    margin = abs(m_a - m_b)
    if margin < TIE_THRESHOLD_SHARPE:
        tie_reasons.append(
            f"Δmean={margin:+.4f} < {TIE_THRESHOLD_SHARPE:.2f} → tiebreaker 1 (lower total turnover)"
        )
        tot_by_freq: dict[str, float] = {}
        for r in rows:
            tot_by_freq.setdefault(r["frequency"], 0.0)
            tot_by_freq[r["frequency"]] += float(r["ann_turnover_bps_mean_daily"] or 0.0)
        cheaper = f_a if tot_by_freq[f_a] <= tot_by_freq[f_b] else f_b
        if cheaper != winner:
            winner = cheaper
            tie_reasons.append(
                f"tiebreak 1 winner: {cheaper} mean_turn_bps={tot_by_freq[cheaper]:.1f} vs alt {tot_by_freq[f_b if cheaper==f_a else f_a]:.1f}"
            )
        else:
            tie_reasons.append(
                f"tiebreak 1 confirms: {winner} already cheaper or tied mean_turn_bps"
            )
    if len(tie_reasons) == 0:
        tie_reasons.append("direct win (no tiebreaker fired)")
    return winner, margin, tie_reasons


def main() -> int:
    print("[Stage 1A] Apply 10 bps/turn tx-costs → pick rebalance frequency")
    print(f"[info] tx-cost rule: {TURNOVER_ONE_SIDED_BPS:.0f} bps / one-sided turnover (buy+sell combined)")

    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    n_tickers = len(tickers)
    print(f"[info] N = {n_tickers} tickers")

    prices = load_prices(
        tickers,
        start_date=STUDY_START,
        end_date=DEV_END,
        final_holdout=False,
    )
    assert not prices.empty, "prices empty"
    assert prices.index.max() < FinalHoldoutDate, "holdout leak"
    close = prices["Close"].ffill().bfill()
    print(f"[info] Close shape = {close.shape}; {close.index.min().date()} -> {close.index.max().date()}")

    equities_raw_path = ROOT / "data" / "processed" / "stage1a_baseline_equities.csv"
    if not equities_raw_path.exists():
        print(f"[FAIL] missing {equities_raw_path} — run build_stage1a_baselines.py first")
        return 2
    equities_raw_df = pd.read_csv(equities_raw_path, index_col=0, parse_dates=True).sort_index()
    sharpes_raw_df = pd.read_csv(
        ROOT / "data" / "processed" / "stage1a_baseline_raw_sharpes.csv"
    )
    raw_sharpe_lookup = {
        (r["frequency"], r["baseline"]): float(r["raw_sharpe_no_txcost"])
        for _, r in sharpes_raw_df.iterrows()
    }

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    equities_txadj_frames: dict[str, pd.Series] = {}
    turnovers_for_print: dict[tuple[str, str], pd.Series] = {}

    for freq_label, step_d in FREQUENCIES.items():
        rebal_dts = _rebalance_dates(pd.DatetimeIndex(close.index), step_d)
        print(f"\n[---] frequency={freq_label:9s} step={step_d:2d}d n_rebal={len(rebal_dts)}")
        for baseline in BASELINES:
            key = f"{freq_label}__{baseline}"
            print(f"[run ] {freq_label:9s} {baseline:15s}  rebuild weights ...", end=" ")
            w_by_rd = _build_weights_by_rebal(close, rebal_dts, baseline, universe)
            print(f"OK ({len(w_by_rd)} rebal weight vectors captured)")

            print(f"[calc]                         turnover ...", end=" ")
            turnover = turnover_from_equities_and_weights(
                equity_index=pd.DatetimeIndex(equities_raw_df.index),
                weights_by_rebal_date=w_by_rd,
                close_prices_df=close,
            )
            turnovers_for_print[(freq_label, baseline)] = turnover
            n_nonzero_turn = int(((turnover.fillna(0.0) > 1e-12).sum()))
            print(f"OK mean={turnover.fillna(0).mean()*10_000:.1f} bps/rebal over {n_nonzero_turn} events")

            raw_eq = equities_raw_df[key].dropna().sort_index()
            eq_tx = apply_tx_costs_to_equity(
                raw_equity_series=raw_eq,
                turnover_series=turnover,
                bps_per_turnover=TURNOVER_ONE_SIDED_BPS,
            )
            equities_txadj_frames[key] = eq_tx.rename(key + "__txadj")

            raw_s = raw_sharpe_lookup.get((freq_label, baseline), float("nan"))
            met = _annualized_metrics(eq_tx, turnover)
            rows.append(
                {
                    "frequency": freq_label,
                    "baseline": baseline,
                    "raw_sharpe": float(raw_s) if np.isfinite(raw_s) else float("nan"),
                    "tx_sharpe": met["sharpe"],
                    "ann_turnover_bps_mean_daily": met["ann_turnover_bps_mean_daily"],
                    "tx_drag_pct_annualized": met["tx_drag_pct_annualized"],
                    "winner_flag": False,
                }
            )
            print(
                f"[ok  ]                         raw_S={raw_s:+.3f}  "
                f"tx_S={met['sharpe']:+.3f}  drag_ann={met['tx_drag_pct_annualized']:+.2f}%  "
                f"ann_turn={met['ann_turnover_bps_mean_daily']:.1f} bps"
            )

    winner_freq, margin_sharpe, tie_reasons_used = _stage1a_winner(rows)
    print("\n===== STAGE 1A SELECTION =====")
    by_freq_mean: dict[str, float] = {}
    for r in rows:
        by_freq_mean.setdefault(r["frequency"], [])
        by_freq_mean[r["frequency"]].append(r["tx_sharpe"])
    for f, vals in by_freq_mean.items():
        print(f"  mean tx-Sharpe (4 baselines) | {f:9s} = {float(np.nanmean(vals)):+.4f}")
    print(f"  margin: {margin_sharpe:+.4f} Sharpe")
    for reason in tie_reasons_used:
        print(f"  → {reason}")
    print(f"  *** WINNER FREQUENCY = {winner_freq.upper()} ({FREQUENCIES[winner_freq]} d rebalance) ***")

    for r in rows:
        if r["frequency"] == winner_freq:
            r["winner_flag"] = True
    decision_row = {
        "frequency": winner_freq,
        "baseline": "DECISION",
        "raw_sharpe": float("nan"),
        "tx_sharpe": float(
            np.nanmean([r["tx_sharpe"] for r in rows if r["frequency"] == winner_freq])
        ),
        "ann_turnover_bps_mean_daily": float(
            np.nanmean(
                [r["ann_turnover_bps_mean_daily"] for r in rows if r["frequency"] == winner_freq]
            )
        ),
        "tx_drag_pct_annualized": float(
            np.nanmean([r["tx_drag_pct_annualized"] for r in rows if r["frequency"] == winner_freq])
        ),
        "winner_flag": True,
    }
    rows_with_decision = list(rows) + [decision_row]
    decision_df = pd.DataFrame(rows_with_decision)
    decision_out = out_dir / "stage1a_decision.csv"
    decision_df.to_csv(decision_out, index=False)
    print(f"\n[ok  ] 9-row decision table written to {decision_out}")

    txadj_out = out_dir / "stage1a_baseline_equities_txadj.csv"
    pd.DataFrame(equities_txadj_frames).sort_index().to_csv(txadj_out)
    print(f"[ok  ] 8 txadj equity curves written to {txadj_out}")

    print("\n=== Stage 1A — 4-baseline tx-cost-adjusted Sharpes ===")
    pd.set_option("display.float_format", "{:+.4f}".format)
    pd.set_option("display.width", 220)
    show_cols = [
        "frequency",
        "baseline",
        "raw_sharpe",
        "tx_sharpe",
        "ann_turnover_bps_mean_daily",
        "tx_drag_pct_annualized",
        "winner_flag",
    ]
    print(decision_df[show_cols].to_string(index=False))

    print("\n[DONE] Stage 1A decision finalized and frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
