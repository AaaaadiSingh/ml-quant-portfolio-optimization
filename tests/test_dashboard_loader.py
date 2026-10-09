"""
Unit Tests for Dashboard Data Loader and Weight Validators
==========================================================
Verifies that the dashboard presentation layer correctly loads precomputed artifacts,
safely handles missing or malformed data, and rigorously validates convex portfolio constraints.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from dashboard import data_loader as dl


def test_load_metrics_summary():
    df = dl.load_metrics_summary()
    assert df is not None, "metrics_summary.csv failed to load"
    assert len(df) == 8, f"Expected 8 strategies, got {len(df)}"
    assert "strategy" in df.columns
    assert "sharpe_txadj" in df.columns
    assert "centroid" in df["strategy"].values
    assert "equal_weight" in df["strategy"].values


def test_load_holdout_eval():
    df = dl.load_holdout_eval()
    assert df is not None, "holdout_eval.csv failed to load"
    assert len(df) == 8, f"Expected 8 strategies, got {len(df)}"
    assert "strategy" in df.columns
    assert "cagr_pct" in df.columns
    assert "sharpe_txadj" in df.columns


def test_load_equity_curves():
    df = dl.load_equity_curves()
    assert df is not None, "phase6_all_equities_txadj.csv failed to load"
    assert len(df) == 2222, f"Expected 2222 trading days, got {len(df)}"
    assert "centroid" in df.columns
    assert "equal_weight" in df.columns
    # Check start is 1.0
    assert np.isclose(df["centroid"].iloc[0], 1.0)
    assert np.isclose(df["equal_weight"].iloc[0], 1.0)


def test_load_centroid_weights():
    df = dl.load_centroid_weights()
    assert df is not None, "phase5_selected_centroid_weights.csv failed to load"
    assert "rebal_date" in df.columns
    assert "ticker" in df.columns
    assert "weight" in df.columns
    dates = df["rebal_date"].unique()
    assert len(dates) == 37, f"Expected 37 rebalance dates, got {len(dates)}"


def test_validate_weights_valid():
    # Construct a valid 10-asset equal-weight portfolio
    w = pd.Series([0.10] * 10)
    is_valid, msgs = dl.validate_weights(w)
    assert is_valid is True
    assert "Valid portfolio" in msgs[0]


def test_validate_weights_invalid_sum():
    # Weights sum to 0.80 instead of 1.0
    w = pd.Series([0.08] * 10)
    is_valid, msgs = dl.validate_weights(w)
    assert is_valid is False
    assert any("sum" in m.lower() for m in msgs)


def test_validate_weights_negative():
    # Contains negative weights (shorting)
    w = pd.Series([0.15, -0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15])
    is_valid, msgs = dl.validate_weights(w)
    assert is_valid is False
    assert any("negative" in m.lower() for m in msgs)


def test_validate_weights_cap_breach():
    # One stock exceeds 10% cap (e.g. 15%)
    w = pd.Series([0.15] + [0.85 / 9] * 9)
    is_valid, msgs = dl.validate_weights(w)
    assert is_valid is False
    assert any("cap" in m.lower() for m in msgs)


def test_validate_weights_with_nans():
    w = pd.Series([0.10] * 9 + [np.nan])
    is_valid, msgs = dl.validate_weights(w)
    assert is_valid is False
    assert any("nan" in m.lower() for m in msgs)


def test_validate_weights_non_numeric():
    w = pd.Series(["0.10"] * 10)
    is_valid, msgs = dl.validate_weights(w)
    assert is_valid is False
    assert any("numeric" in m.lower() for m in msgs)


def test_missing_file_handling(monkeypatch, tmp_path):
    # Clear Streamlit cache and temporarily point RESULTS_DIR to an empty temp dir
    dl.load_metrics_summary.clear()
    monkeypatch.setattr(dl, "RESULTS_DIR", tmp_path)
    try:
        res = dl.load_metrics_summary()
        assert res is None
    finally:
        dl.load_metrics_summary.clear()

