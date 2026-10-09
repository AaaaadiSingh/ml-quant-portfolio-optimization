from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import WalkForwardBacktestEngine, simulate_walk_forward
from src.metrics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    compute_all_metrics,
    empirical_cvar,
    empirical_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def test_metrics_on_synthetic_series():
    # 252 business days
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    
    # 1. Flat equity curve
    flat_eq = pd.Series(1.0, index=dates)
    assert abs(cagr(flat_eq)) < 1e-6
    assert abs(annualized_volatility(flat_eq)) < 1e-6
    assert abs(max_drawdown(flat_eq)) < 1e-6

    # 2. Linearly growing equity curve: 1.0 -> 1.10 (+10% in 1 year)
    grow_eq = pd.Series(np.linspace(1.0, 1.10, len(dates)), index=dates)
    grow_cagr = cagr(grow_eq)
    assert grow_cagr > 9.0 and grow_cagr < 11.5
    assert max_drawdown(grow_eq) == 0.0

    # 3. Known return series for VaR / CVaR
    rets = np.array([-0.05, -0.04, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    var_5 = empirical_var(rets, alpha=0.10)
    cvar_5 = empirical_cvar(rets, alpha=0.10)
    assert var_5 <= -0.04
    assert cvar_5 <= var_5

    # 4. Consolidated metrics dictionary
    m = compute_all_metrics(grow_eq, risk_free_annual=0.04)
    assert "sharpe_txadj" in m
    assert "sortino" in m
    assert "calmar" in m
    assert "var_95_daily_pct" in m
    assert "cvar_95_daily_pct" in m


def test_backtest_engine_drift_and_turnover_simulation():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    tickers = ["AAPL", "GOOG", "MSFT"]
    
    # Random positive prices
    np.random.seed(42)
    daily_pct = np.random.normal(0.0005, 0.01, size=(len(dates), len(tickers)))
    price_mat = 100.0 * np.cumprod(1.0 + daily_pct, axis=0)
    close_df = pd.DataFrame(price_mat, index=dates, columns=tickers)

    rd0 = dates[0]
    rd1 = dates[25]
    
    weights = {
        rd0: np.array([0.4, 0.4, 0.2]),
        rd1: np.array([0.2, 0.3, 0.5]),
    }

    engine = WalkForwardBacktestEngine(close_df, bps_per_turnover=10.0)
    raw, tx, turnover, drift_df = engine.simulate(weights)

    # Asserts
    assert len(raw) == len(dates)
    assert len(tx) == len(dates)
    assert raw.iloc[0] == 1.0
    assert (raw > 0).all()
    assert (tx > 0).all()
    assert (tx <= raw + 1e-12).all()  # Transaction costs reduce equity

    # Turnover at initial is 0, at rd1 is positive
    assert turnover.loc[rd0] == 0.0
    assert turnover.loc[rd1] > 0.0

    # Drift sum across all dates must be strictly 1.0
    assert len(drift_df) == len(dates)
    assert np.max(np.abs(drift_df["drift_sum"] - 1.0)) < 1e-5


def test_backtest_engine_daily_return_identity():
    """
    Rigorously verifies the mathematical identity that daily portfolio return
    equals sum(w_held * r_asset) on EVERY day, boundary days included.
    """
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    tickers = ["T1", "T2", "T3", "T4"]
    np.random.seed(123)
    daily_pct = np.random.normal(0.001, 0.015, size=(len(dates), len(tickers)))
    price_mat = 100.0 * np.cumprod(1.0 + daily_pct, axis=0)
    close_df = pd.DataFrame(price_mat, index=dates, columns=tickers)

    # Multiple rebalance dates
    rebal_dates = [dates[0], dates[20], dates[50], dates[80]]
    weights = {
        rebal_dates[0]: np.array([0.25, 0.25, 0.25, 0.25]),
        rebal_dates[1]: np.array([0.40, 0.10, 0.30, 0.20]),
        rebal_dates[2]: np.array([0.10, 0.50, 0.20, 0.20]),
        rebal_dates[3]: np.array([0.30, 0.30, 0.20, 0.20]),
    }

    engine = WalkForwardBacktestEngine(close_df, bps_per_turnover=10.0)
    raw, tx, turnover, drift_df = engine.simulate(weights)

    # Verify no NaN values
    assert not raw.isna().any()
    assert not tx.isna().any()

    # Calculate actual realized daily returns of equity
    equity_daily_ret = raw.values[1:] / raw.values[:-1] - 1.0

    # Reconstruct expected daily returns from asset returns and weights held
    P = close_df.values
    asset_ret = P[1:] / P[:-1] - 1.0

    w_held = np.copy(weights[rebal_dates[0]])
    expected_daily_ret = np.zeros(len(asset_ret))

    for t in range(len(asset_ret)):
        r_t = asset_ret[t]
        expected_daily_ret[t] = float(np.dot(w_held, r_t))
        # Drift weights
        port_g = 1.0 + expected_daily_ret[t]
        w_drift = (w_held * (1.0 + r_t)) / port_g
        # Check rebalance for next day
        next_dt = dates[t + 1]
        if next_dt in weights:
            w_held = np.copy(weights[next_dt])
        else:
            w_held = np.copy(w_drift)

    max_delta = np.max(np.abs(equity_daily_ret - expected_daily_ret))
    assert max_delta < 1e-12, f"Daily return identity violated! Max delta = {max_delta}"


def test_backtest_engine_rebalance_boundary_continuity():
    """
    Verifies that on rebalance dates, the equity curve is strictly continuous
    and does NOT reset to 1.0 or create artificial drop cliffs.
    """
    dates = pd.date_range("2021-01-01", periods=60, freq="B")
    tickers = ["T1", "T2"]
    # Steady 1% daily gain in both assets
    close_df = pd.DataFrame(
        {
            "T1": 100.0 * (1.01 ** np.arange(len(dates))),
            "T2": 100.0 * (1.01 ** np.arange(len(dates))),
        },
        index=dates,
    )

    rebal_dates = [dates[0], dates[20], dates[40]]
    weights = {rd: np.array([0.5, 0.5]) for rd in rebal_dates}

    engine = WalkForwardBacktestEngine(close_df, bps_per_turnover=0.0)
    raw, tx, turnover, _ = engine.simulate(weights)

    # With 1% steady gain in both assets, equity on day t MUST equal 1.01^t
    expected_equity = 1.01 ** np.arange(len(dates))
    max_err = np.max(np.abs(raw.values - expected_equity))
    assert max_err < 1e-10, f"Compounding error! Max deviation from 1.01^t: {max_err}"

    # Specifically check rebalance dates: equity must NOT be 1.0 at day 20 or day 40!
    assert raw.loc[dates[20]] > 1.20
    assert raw.loc[dates[40]] > 1.48


def test_backtest_engine_independent_equal_weight_match():
    """
    Independent validation comparing WalkForwardBacktestEngine against
    an independent, minimal multi-period pandas simulation.
    """
    dates = pd.date_range("2022-01-01", periods=120, freq="B")
    tickers = ["A", "B", "C"]
    np.random.seed(99)
    pcts = np.random.normal(0.0005, 0.01, size=(len(dates), len(tickers)))
    close_df = pd.DataFrame(100.0 * np.cumprod(1.0 + pcts, axis=0), index=dates, columns=tickers)

    rebal_dates = [dates[0], dates[30], dates[60], dates[90]]
    weights = {rd: np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]) for rd in rebal_dates}

    engine = WalkForwardBacktestEngine(close_df, bps_per_turnover=0.0)
    raw, _, _, _ = engine.simulate(weights)

    # Independent piecewise rebalancing computation
    indep_equity = np.ones(len(dates))
    cur_base = 1.0
    for k in range(len(rebal_dates)):
        rd = rebal_dates[k]
        pos0 = dates.get_loc(rd)
        pos1 = dates.get_loc(rebal_dates[k + 1]) if k + 1 < len(rebal_dates) else len(dates) - 1

        # Holding period relative returns vs pos0
        sub_p = close_df.iloc[pos0 : pos1 + 1].values
        rel_p = sub_p / sub_p[0]
        sub_port = np.mean(rel_p, axis=1)

        indep_equity[pos0 : pos1 + 1] = cur_base * sub_port
        cur_base = indep_equity[pos1]

    max_diff = np.max(np.abs(raw.values - indep_equity))
    assert max_diff < 1e-12, f"Independent EW match failed! Max diff = {max_diff}"
