"""
Dashboard Data Loading and Validation Utilities
===============================================
Institutional-grade cached data loading layer for the Streamlit dashboard.
All data paths are resolved relative to the project root.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Root is two levels up from dashboard/data_loader.py
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_RAW = ROOT_DIR / "data" / "raw"
RESULTS_DIR = ROOT_DIR / "results"
DOCS_DIR = ROOT_DIR / "docs"

# Standard strategy display names and colors
STRATEGY_DISPLAY = {
    "centroid": "MC Centroid (Mean, LW Σ, ±3pp)",
    "ridge_linreg": "Ridge LinReg Point Est.",
    "rf": "Random Forest Point Est.",
    "xgb": "XGBoost Point Est.",
    "equal_weight": "Equal Weight (1/N)",
    "min_variance": "Minimum Variance (LW Σ)",
    "risk_parity": "Risk Parity (LW Σ)",
    "classic_max_sharpe": "Classic Max Sharpe (LW Σ, ±3pp)",
}

STRATEGY_COLORS = {
    "centroid": "#2563EB",          # Royal blue (primary highlight)
    "ridge_linreg": "#0D9488",      # Teal
    "rf": "#16A34A",                # Emerald green
    "xgb": "#CA8A04",               # Amber
    "equal_weight": "#64748B",      # Slate grey (benchmark)
    "min_variance": "#0284C7",      # Sky blue
    "risk_parity": "#9333EA",       # Purple
    "classic_max_sharpe": "#DC2626",# Crimson
}


@st.cache_data(show_spinner=False)
def load_metrics_summary() -> Optional[pd.DataFrame]:
    """Load authoritative 8-strategy walk-forward performance summary."""
    p = RESULTS_DIR / "metrics_summary.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception as exc:
        st.error(f"Error loading metrics summary: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_holdout_eval() -> Optional[pd.DataFrame]:
    """Load strictly embargoed 18-month holdout evaluation results."""
    p = RESULTS_DIR / "holdout_eval.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception as exc:
        st.error(f"Error loading holdout evaluation: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_equity_curves() -> Optional[pd.DataFrame]:
    """
    Load daily continuous cumulative equity curves for all 8 strategies.
    Index is parsed DatetimeIndex; initial date 2015-01-01 starts at 1.0.
    """
    p = DATA_PROCESSED / "phase6_all_equities_txadj.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df
    except Exception as exc:
        st.error(f"Error loading equity curves: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_centroid_weights() -> Optional[pd.DataFrame]:
    """
    Load historical quarterly rebalance weights for the MC Centroid portfolio.
    Columns: rebal_date, ticker, weight
    """
    p = DATA_PROCESSED / "phase5_selected_centroid_weights.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["rebal_date"])
        return df
    except Exception as exc:
        st.error(f"Error loading centroid weights: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_sector_map() -> Optional[pd.DataFrame]:
    """Load sector and company name classification metadata."""
    p = DOCS_DIR / "nifty50_sector_map.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        cols_needed = ["ticker", "company_name", "sector_provisional", "free_float_rank"]
        avail = [c for c in cols_needed if c in df.columns]
        return df[avail]
    except Exception as exc:
        st.error(f"Error loading sector map: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_bootstrap_distributions() -> Optional[pd.DataFrame]:
    """
    Load B=200 stationary block bootstrap synthetic performance distributions.
    Columns: path_id, strategy, cagr, annualized_vol, sharpe, sortino, max_drawdown, calmar
    """
    p = DATA_PROCESSED / "phase7_bootstrap_distributions.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception as exc:
        st.error(f"Error loading bootstrap distributions: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_outperformance_probs() -> Optional[pd.DataFrame]:
    """Load bootstrap outperformance probabilities vs Equal Weight and competitors."""
    p = DATA_PROCESSED / "phase7_outperformance_probabilities.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception as exc:
        st.error(f"Error loading outperformance probabilities: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_regime_analysis() -> Optional[pd.DataFrame]:
    """Load performance across 6 historical macro regimes."""
    p = DATA_PROCESSED / "phase6_regime_analysis.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception as exc:
        st.error(f"Error loading regime analysis: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_txcost_sensitivity() -> Optional[pd.DataFrame]:
    """Load transaction cost sensitivity sweep across [0, 5, 10, 20, 30, 50] bps."""
    p = DATA_PROCESSED / "phase6_txcost_sensitivity.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception as exc:
        st.error(f"Error loading transaction cost sensitivity: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_factor_attribution() -> Optional[pd.DataFrame]:
    """Load Carhart 4-factor regression exposures (MKT, SMB, HML, MOM)."""
    p = DATA_PROCESSED / "phase6_factor_attribution.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception as exc:
        st.error(f"Error loading factor attribution: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_jk_pairwise_tests() -> Optional[pd.DataFrame]:
    """Load Jobson-Korkie Memmel (2003) pairwise hypothesis test results."""
    p = DATA_PROCESSED / "phase6_jk_pairwise_tests.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception as exc:
        st.error(f"Error loading Jobson-Korkie pairwise tests: {exc}")
        return None


def validate_weights(
    weights_series: pd.Series, tolerance: float = 0.01
) -> Tuple[bool, List[str]]:
    """
    Validates a series of portfolio weights against institutional convex constraints:
    1. Numeric and all finite
    2. Non-negative (long-only: w_i >= -1e-6)
    3. Sum to 1.0 within tolerance
    4. Single-name cap compliance (w_i <= 0.10 + tolerance)
    """
    messages = []
    is_valid = True

    if not pd.api.types.is_numeric_dtype(weights_series):
        return False, ["Weights series is not numeric."]

    if weights_series.isna().any() or np.isinf(weights_series).any():
        return False, ["Weights contain NaN or infinite values."]

    # Non-negative
    if (weights_series < -1e-5).any():
        neg_count = (weights_series < -1e-5).sum()
        messages.append(f"Contains {neg_count} negative weight(s).")
        is_valid = False

    # Sum to 1.0
    total = float(weights_series.sum())
    if abs(total - 1.0) > tolerance:
        messages.append(f"Weight sum = {total:.4f} (expected ~1.0 within {tolerance}).")
        is_valid = False

    # Max concentration cap (10%)
    max_w = float(weights_series.max())
    if max_w > 0.10 + tolerance:
        messages.append(f"Max single-name weight = {max_w*100:.2f}% exceeds 10% cap.")
        is_valid = False

    if is_valid:
        messages.append(f"Valid portfolio: sum={total*100:.2f}%, max_name={max_w*100:.2f}%, {len(weights_series)} assets.")

    return is_valid, messages
