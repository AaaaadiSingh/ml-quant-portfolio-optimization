from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import pandas as pd

DEFAULT_RISK_FREE_ANNUAL = 0.04
DEFAULT_TRADING_DAYS_PER_YEAR = 252


def cagr(equity: pd.Series, trading_days: int = DEFAULT_TRADING_DAYS_PER_YEAR) -> float:
    """
    Geometric Compound Annual Growth Rate (CAGR) in percentage.
    Formula identical to NB2-consistent metrics:
    ((eq[-1] / eq[0]) ** (1 / n_years) - 1.0) * 100.0
    """
    eq = equity.sort_index().dropna()
    if len(eq) < 2:
        return float("nan")
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    if n_years <= 0:
        return float("nan")
    return float(((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / n_years) - 1.0) * 100.0)


def annualized_volatility(equity: pd.Series, trading_days: int = DEFAULT_TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualized daily return volatility: std(ddof=1) * sqrt(252).
    """
    eq = equity.sort_index().dropna()
    if len(eq) < 2:
        return float("nan")
    r = eq.pct_change().dropna()
    return float(r.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(
    equity: pd.Series,
    risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL,
    trading_days: int = DEFAULT_TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized excess Sharpe ratio matching NB2 formula exactly:
    (mean(daily_ret) - rf_daily) / std(daily_ret, ddof=1) * sqrt(trading_days)
    """
    eq = equity.sort_index().dropna()
    if len(eq) < 2:
        return float("nan")
    r = eq.pct_change().fillna(0.0)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / trading_days) - 1.0
    if sigma < 1e-15:
        return float("nan")
    return float((mu - rf_daily) / sigma * np.sqrt(trading_days))


def max_drawdown(equity: pd.Series) -> float:
    """
    Peak-to-trough maximum drawdown in percentage:
    ((eq / eq.cummax() - 1.0).min()) * 100.0
    """
    eq = equity.sort_index().dropna()
    if len(eq) < 2:
        return float("nan")
    peak = eq.cummax()
    return float(((eq / peak - 1.0).min()) * 100.0)


def sortino_ratio(
    equity: pd.Series,
    risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL,
    trading_days: int = DEFAULT_TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Sortino ratio penalizing only downside volatility below daily risk-free rate.
    """
    eq = equity.sort_index().dropna()
    if len(eq) < 2:
        return float("nan")
    daily_ret = eq.pct_change().dropna()
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / trading_days) - 1.0
    excess_ret = daily_ret - rf_daily
    downside = np.where(excess_ret < 0.0, excess_ret, 0.0)
    downside_vol = np.sqrt(np.mean(downside ** 2)) * np.sqrt(trading_days)
    if downside_vol < 1e-12:
        return float("nan")
    ann_excess = (eq.iloc[-1] / eq.iloc[0]) ** (trading_days / len(daily_ret)) - 1.0 - risk_free_annual
    return float(ann_excess / downside_vol)


def calmar_ratio(equity: pd.Series, risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL, trading_days: int = DEFAULT_TRADING_DAYS_PER_YEAR) -> float:
    """
    Calmar ratio = Annual Return / |Max Drawdown|.
    """
    cagr_val = cagr(equity, trading_days)
    mdd_val = max_drawdown(equity)
    if math.isnan(cagr_val) or math.isnan(mdd_val) or abs(mdd_val) < 1e-8:
        return float("nan")
    return float((cagr_val / 100.0) / abs(mdd_val / 100.0))


def empirical_var(daily_returns: Union[pd.Series, np.ndarray], alpha: float = 0.05) -> float:
    """
    Empirical Value at Risk (VaR) quantile (e.g. 5th percentile).
    """
    rets = np.asarray(daily_returns).ravel()
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return float("nan")
    return float(np.percentile(rets, alpha * 100.0))


def empirical_cvar(daily_returns: Union[pd.Series, np.ndarray], alpha: float = 0.05) -> float:
    """
    Empirical Conditional Value at Risk (CVaR / Expected Shortfall).
    Mean of losses beyond the VaR alpha threshold.
    """
    rets = np.asarray(daily_returns).ravel()
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return float("nan")
    var_q = empirical_var(rets, alpha=alpha)
    tail = rets[rets <= var_q]
    return float(np.mean(tail)) if len(tail) > 0 else var_q


def compute_all_metrics(
    equity: pd.Series,
    risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL,
    trading_days: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Consolidated performance metrics dictionary matching Phase 6 schema.
    """
    eq = equity.sort_index().dropna()
    daily_rets = eq.pct_change().dropna()
    return {
        "cagr_pct": cagr(eq, trading_days),
        "ann_vol": annualized_volatility(eq, trading_days),
        "sharpe_txadj": sharpe_ratio(eq, risk_free_annual, trading_days),
        "max_dd_pct": max_drawdown(eq),
        "sortino": sortino_ratio(eq, risk_free_annual, trading_days),
        "calmar": calmar_ratio(eq, risk_free_annual, trading_days),
        "var_95_daily_pct": empirical_var(daily_rets, alpha) * 100.0,
        "cvar_95_daily_pct": empirical_cvar(daily_rets, alpha) * 100.0,
    }
