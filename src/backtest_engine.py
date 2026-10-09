from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

DEFAULT_TURNOVER_ONE_SIDED_BPS = 10.0


def apply_tx_costs_to_equity(
    raw_equity_series: pd.Series,
    turnover_series: pd.Series,
    bps_per_turnover: float = DEFAULT_TURNOVER_ONE_SIDED_BPS,
) -> pd.Series:
    """
    Applies proportional transaction cost drag to a daily portfolio equity curve.
    At each rebalance date, equity is reduced by (bps / 10,000) * turnover.
    """
    eq = raw_equity_series.sort_index().copy()
    if len(eq) == 0:
        return eq
    drag_per_rebal = pd.Series(0.0, index=eq.index, dtype=float)
    for rd, turn in turnover_series.items():
        if rd in drag_per_rebal.index and np.isfinite(turn) and turn >= 0.0:
            cost_rate = (bps_per_turnover / 10_000.0) * float(turn)
            drag_per_rebal.loc[rd] = cost_rate
    cumulative_log_drag = (-drag_per_rebal).cumsum()
    log_eq = np.log(np.where(eq.values > 0, eq.values, np.nan))
    adjusted_log = log_eq + cumulative_log_drag.values
    return pd.Series(np.exp(adjusted_log), index=eq.index, name=eq.name)


class WalkForwardBacktestEngine:
    """
    Modular walk-forward execution and portfolio accounting engine.
    Orchestrates continuous close-to-close daily portfolio evolution,
    daily asset return compounding, weight drift tracking, turnover calculation,
    and transaction cost drag modeling without period boundary disconnects.
    """

    def __init__(
        self,
        close_prices: pd.DataFrame,
        bps_per_turnover: float = DEFAULT_TURNOVER_ONE_SIDED_BPS,
    ) -> None:
        self.close_prices = close_prices.ffill().bfill().sort_index()
        self.tickers = sorted(self.close_prices.columns.tolist())
        self.daily_index = pd.DatetimeIndex(self.close_prices.index.sort_values())
        self.bps_per_turnover = bps_per_turnover

    def simulate(
        self,
        weights_by_rebal: Dict[pd.Timestamp, np.ndarray],
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """
        Simulates walk-forward portfolio equity curve with continuous daily compounding
        and weight drift.

        Returns:
            equity_raw: pd.Series continuously compounded without transaction costs
            equity_txadj: pd.Series with transaction cost drag
            turnover_series: pd.Series of one-sided turnover per rebalance date
            drift_audit_df: pd.DataFrame with daily drifted weights and sum-to-1 error
        """
        rebal_dates = sorted([rd for rd in weights_by_rebal.keys() if rd in self.daily_index])
        if len(rebal_dates) == 0:
            raise ValueError("No valid rebalance dates found in daily price index.")

        rebal_set = set(rebal_dates)
        n_days = len(self.daily_index)
        n_tickers = len(self.tickers)

        # Validate weight vectors
        for rd in rebal_dates:
            w = np.asarray(weights_by_rebal[rd], dtype=float)
            assert len(w) == n_tickers, f"Weight vector size {len(w)} != {n_tickers} at {rd}"
            assert abs(np.sum(w) - 1.0) < 1e-5, f"Weights at {rd} do not sum to 1: {np.sum(w)}"

        # Prepare daily close price matrix aligned to sorted tickers
        close_mat = self.close_prices[self.tickers].values.astype(float)
        # Daily simple returns: R[t] = close[t] / close[t-1] - 1 for t >= 1
        safe_prev_close = np.where(close_mat[:-1] > 0, close_mat[:-1], 1.0)
        daily_returns = close_mat[1:] / safe_prev_close - 1.0  # shape (n_days - 1, n_tickers)

        equity_raw = np.ones(n_days, dtype=float)
        turnover_dict: Dict[pd.Timestamp, float] = {}
        drift_rows: List[dict] = []

        # Find initial rebalance date
        first_rd = rebal_dates[0]
        first_loc = self.daily_index.get_loc(first_rd)
        w_current = np.asarray(weights_by_rebal[first_rd], dtype=float)
        turnover_dict[first_rd] = 0.0

        current_rebal_period = first_rd

        # Record day 0 drift audit
        drift_rows.append({
            "date": self.daily_index[0],
            "rebal_period": first_rd,
            "drift_sum": float(np.sum(w_current)),
            "max_weight": float(np.max(w_current)),
        })

        for t in range(1, n_days):
            curr_date = self.daily_index[t]
            r_t = daily_returns[t - 1]  # asset returns from t-1 to t

            # Portfolio return earned over day t using weights held entering day t
            port_ret_t = float(np.dot(w_current, r_t))
            equity_raw[t] = equity_raw[t - 1] * (1.0 + port_ret_t)

            # Drift weights to the end of day t
            denom = 1.0 + port_ret_t
            if denom > 1e-12:
                w_drift_t = (w_current * (1.0 + r_t)) / denom
            else:
                w_drift_t = np.copy(w_current)

            drift_sum = float(np.sum(w_drift_t))
            assert abs(drift_sum - 1.0) < 1e-4, f"Drift sum breach at {curr_date}: {drift_sum}"

            # Check if day t is a rebalance date
            if curr_date in rebal_set:
                w_target = np.asarray(weights_by_rebal[curr_date], dtype=float)
                # Turnover vs drifted weights at close of day t
                turn = 0.5 * float(np.sum(np.abs(w_target - w_drift_t)))
                turnover_dict[curr_date] = max(0.0, turn)
                current_rebal_period = curr_date
                # Rebalance at close: holding weights for next day are w_target
                w_current = np.copy(w_target)
            else:
                # Retain drifted weights for next day
                w_current = np.copy(w_drift_t)

            drift_rows.append({
                "date": curr_date,
                "rebal_period": current_rebal_period,
                "drift_sum": drift_sum,
                "max_weight": float(np.max(w_drift_t)),
            })

        raw_series = pd.Series(equity_raw, index=self.daily_index, name="equity_raw")
        turnover_series = pd.Series(turnover_dict, name="turnover_one_sided").sort_index()
        txadj_series = apply_tx_costs_to_equity(
            raw_series, turnover_series, bps_per_turnover=self.bps_per_turnover
        ).rename("equity_txadj")
        drift_df = pd.DataFrame(drift_rows)

        return raw_series, txadj_series, turnover_series, drift_df


def simulate_walk_forward(
    close_prices: pd.DataFrame,
    weights_by_rebal: Dict[pd.Timestamp, np.ndarray],
    bps_per_turnover: float = DEFAULT_TURNOVER_ONE_SIDED_BPS,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """
    Functional wrapper for WalkForwardBacktestEngine.simulate().
    """
    engine = WalkForwardBacktestEngine(close_prices, bps_per_turnover=bps_per_turnover)
    return engine.simulate(weights_by_rebal)
