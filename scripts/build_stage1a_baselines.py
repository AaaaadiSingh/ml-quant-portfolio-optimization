from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import FinalHoldoutDate, load_prices

STUDY_START = pd.Timestamp("2015-01-01")
DEV_END = pd.Timestamp("2023-12-31")

MONTHLY_STEP_D = 21
QUARTERLY_STEP_D = 63

COV_WINDOW_D = 63
VOL_WINDOW_D = 63

WEIGHT_SUM_TOL = 1e-8


def _equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _free_float_proxy_weight(tickers: list[str], universe: pd.DataFrame) -> np.ndarray:
    ranks = []
    for t in tickers:
        row = universe.loc[universe["ticker"] == t]
        ranks.append(float(row["free_float_rank"].iloc[0]) if len(row) == 1 else np.nan)
    r = np.array(ranks, dtype=float)
    inv = 1.0 / np.clip(r, 1e-6, None)
    return inv / inv.sum()


def _solve_min_variance(cov: np.ndarray, n: int) -> np.ndarray:
    try:
        inv = np.linalg.pinv(cov)
        ones = np.ones((n, 1))
        w_unscaled = inv @ ones
        denom_arr = ones.T @ w_unscaled
        denom = float(denom_arr.item() if hasattr(denom_arr, "item") else denom_arr)
        if abs(denom) < 1e-12:
            return _equal_weight(n)
        w = w_unscaled.flatten() / denom
    except np.linalg.LinAlgError:
        return _equal_weight(n)
    w = np.clip(w, -2.0, 2.0)
    s = w.sum()
    if abs(s) < WEIGHT_SUM_TOL:
        return _equal_weight(n)
    return w / s


def _risk_parity_weights(cov_diag: np.ndarray, n: int) -> np.ndarray:
    vols = np.sqrt(np.clip(cov_diag, 1e-16, None))
    inv = 1.0 / vols
    return inv / inv.sum()


def _rebalance_dates(daily_idx: pd.DatetimeIndex, step_d: int) -> pd.DatetimeIndex:
    daily_idx_sorted = pd.DatetimeIndex(daily_idx.sort_values())
    if len(daily_idx_sorted) == 0:
        return pd.DatetimeIndex([])
    rebal_dts = [daily_idx_sorted[0]]
    last_pos = 0
    while True:
        last_dt = daily_idx_sorted[last_pos]
        target_dt = last_dt + pd.offsets.BDay(step_d)
        candidates = daily_idx_sorted[daily_idx_sorted > target_dt]
        if len(candidates) == 0:
            break
        next_dt = candidates[0]
        pos = daily_idx_sorted.get_loc(next_dt)
        rebal_dts.append(next_dt)
        last_pos = int(pos)
    return pd.DatetimeIndex(rebal_dts)


def _walk_forward_equity(
    close: pd.DataFrame,
    step_d: int,
    baseline: str,
    universe: pd.DataFrame,
) -> pd.Series:
    tickers = list(close.columns)
    n = len(tickers)
    rebal_dts = _rebalance_dates(pd.DatetimeIndex(close.index), step_d)
    if len(rebal_dts) < 2:
        return pd.Series(dtype=float)

    daily_idx = pd.DatetimeIndex(close.index.sort_values())
    pos = close.index.get_indexer(rebal_dts)
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

    equity_idx = daily_idx
    equity_vals = np.full(len(equity_idx), np.nan, dtype=float)
    cum_value = 1.0
    prev_close = None
    w_curr = np.array([])

    for i_dt, dt in enumerate(equity_idx):
        c = close.loc[dt].values.astype(float)
        if dt in weights_by_date:
            if prev_close is None:
                w_curr = weights_by_date[dt]
                prev_close = c
                equity_vals[i_dt] = cum_value
                continue
            period_ret = np.log(c / prev_close)
            portfolio_ret = float(np.dot(w_curr, period_ret))
            cum_value *= np.exp(portfolio_ret)
            w_curr = weights_by_date[dt]
            prev_close = c
            equity_vals[i_dt] = cum_value
        else:
            if prev_close is None:
                equity_vals[i_dt] = cum_value
                prev_close = c
                w_curr = _equal_weight(n) if w_curr.size == 0 else w_curr
                continue
            period_ret = np.log(c / prev_close)
            portfolio_ret = float(np.dot(w_curr, period_ret))
            cum_value *= np.exp(portfolio_ret)
            prev_close = c
            equity_vals[i_dt] = cum_value

    return pd.Series(equity_vals, index=equity_idx, name=f"{baseline}").dropna()


def _raw_sharpe(equity: pd.Series) -> float:
    if len(equity) < 10:
        return float("nan")
    lr = np.log(equity / equity.shift(1)).dropna()
    if lr.std(ddof=0) < 1e-12:
        return 0.0
    return float(lr.mean() / lr.std(ddof=0) * np.sqrt(252))


def main() -> int:
    print("[Stage 1A] Building 2 rebalance panels + 8 baseline equity curves")
    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    print(f"[info] N = {len(tickers)} tickers")

    prices = load_prices(tickers, start_date=STUDY_START, end_date=DEV_END, final_holdout=False)
    assert not prices.empty, "prices empty"
    assert prices.index.max() < FinalHoldoutDate, "holdout leak in stage1a"
    close = prices["Close"].ffill().bfill()
    print(f"[info] Close shape = {close.shape}; {close.index.min().date()} -> {close.index.max().date()}")

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly_dts = _rebalance_dates(pd.DatetimeIndex(close.index), MONTHLY_STEP_D)
    quarterly_dts = _rebalance_dates(pd.DatetimeIndex(close.index), QUARTERLY_STEP_D)
    monthly_panel = close.loc[monthly_dts]
    quarterly_panel = close.loc[quarterly_dts]
    monthly_panel.to_csv(out_dir / "monthly_rebalance_prices.csv")
    quarterly_panel.to_csv(out_dir / "quarterly_rebalance_prices.csv")
    print(f"[ok  ] monthly panel shape = {monthly_panel.shape}, saved to monthly_rebalance_prices.csv")
    print(f"[ok  ] quarterly panel shape = {quarterly_panel.shape}, saved to quarterly_rebalance_prices.csv")

    baselines = ["equal", "freefloat_proxy", "minvar", "riskparity"]
    steps = {
        "monthly": MONTHLY_STEP_D,
        "quarterly": QUARTERLY_STEP_D,
    }

    equity_frames: dict[str, pd.Series] = {}
    sharpe_rows = []
    for freq_label, step_d in steps.items():
        for b in baselines:
            print(f"[run ] {freq_label:9s} {b:15s} ...", end=" ")
            eq = _walk_forward_equity(close, step_d, b, universe)
            key = f"{freq_label}__{b}"
            eq_out = eq.rename(key)
            equity_frames[key] = eq_out
            s = _raw_sharpe(eq)
            ret_total = float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) >= 2 else float("nan")
            sharpe_rows.append({
                "frequency": freq_label,
                "baseline": b,
                "raw_sharpe_no_txcost": s,
                "total_return_decimal": ret_total,
                "n_rebalances": int((close.index.get_indexer(eq.index) >= 0).sum() // max(1, step_d)),
            })
            print(f"Sharpe(raw) = {s:+.3f} | total ret = {ret_total*100:+.1f}%")

    equities_df = pd.DataFrame(equity_frames).sort_index()
    equities_out = out_dir / "stage1a_baseline_equities.csv"
    equities_df.to_csv(equities_out)
    print(f"[ok  ] 8 equity curves written to {equities_out}")

    sharpe_table = pd.DataFrame(sharpe_rows)
    sharpe_out = out_dir / "stage1a_baseline_raw_sharpes.csv"
    sharpe_table.to_csv(sharpe_out, index=False)
    print(f"[ok  ] Sharpe table written to {sharpe_out}")
    print("\n=== Stage 1A Raw Sharpe Table (no transaction costs) ===")
    with pd.option_context("display.float_format", "{:+.3f}".format, "display.width", 200):
        print(sharpe_table[["frequency", "baseline", "raw_sharpe_no_txcost", "total_return_decimal"]].to_string(index=False))
    print("\n[DONE] Stage 1A data prep + 8 baseline equities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
