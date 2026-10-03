from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import FinalHoldoutDate, load_prices

SENTINEL = -999999.0
HOLDOUT_START = FinalHoldoutDate
PRE_START = pd.Timestamp("2015-01-01")
PRE_END = HOLDOUT_START - pd.Timedelta(days=1)
FULL_END = pd.Timestamp("2025-06-30")


def _build_synthetic_ticker(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(PRE_START, FULL_END, freq="B")
    n = len(dates)
    base = 100.0 + rng.normal(0, 0.5, n).cumsum()
    base = np.clip(base, 5.0, 10_000.0)
    df = pd.DataFrame(
        {
            "Open": base * (1 + rng.normal(0, 0.002, n)),
            "High": base * (1 + rng.normal(0, 0.01, n)),
            "Low": base * (1 - np.abs(rng.normal(0, 0.01, n))),
            "Close": base,
            "Adj Close": base,
            "Volume": np.floor(1e6 + rng.uniform(0, 1e6, n)).astype(float),
        },
        index=dates,
    )
    holdout_mask = df.index >= HOLDOUT_START
    n_hold = int(holdout_mask.sum())
    if n_hold == 0:
        raise AssertionError("synthetic data has zero holdout rows — date range broken")
    df.loc[holdout_mask, "Close"] = SENTINEL
    df.loc[holdout_mask, "Adj Close"] = SENTINEL
    df.loc[holdout_mask, "Open"] = SENTINEL
    df.loc[holdout_mask, "High"] = SENTINEL
    df.loc[holdout_mask, "Low"] = SENTINEL
    df.loc[holdout_mask, "Volume"] = float(int(SENTINEL))
    return df


@pytest.fixture
def cache_tmp():
    tmp = Path(tempfile.mkdtemp(prefix="qtest_no_lookahead_"))
    try:
        tickers = ["FAKEA.NS", "FAKEB.NS", "FAKEC.NS"]
        for i, t in enumerate(tickers):
            df = _build_synthetic_ticker(seed=i + 1)
            df.to_csv(tmp / f"{t}.csv", index=True)
        yield tmp, tickers
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_holdout_guard_blocks_sentinel(cache_tmp):
    cache_dir, tickers = cache_tmp
    df = load_prices(
        tickers,
        start_date=PRE_START,
        end_date=FULL_END,
        final_holdout=False,
        cache_dir=cache_dir,
    )
    assert not df.empty, "load_prices returned empty DataFrame"
    assert df.index.max() < HOLDOUT_START, (
        f"guard failed — max date {df.index.max()} >= holdout {HOLDOUT_START}"
    )
    sentinel_hits = int((df == SENTINEL).sum().sum())
    assert sentinel_hits == 0, (
        f"holdout sentinel LEAKED into final_holdout=False view: {sentinel_hits} cells"
    )
    sentinel_hits_abs = int((np.abs(df + SENTINEL) < 1e-6).sum().sum())
    assert sentinel_hits_abs == 0, (
        f"holdout sentinel variant LEAKED: {sentinel_hits_abs} cells"
    )


def test_holdout_guard_reveals_sentinel_when_enabled(cache_tmp):
    cache_dir, tickers = cache_tmp
    df = load_prices(
        tickers,
        start_date=PRE_START,
        end_date=FULL_END,
        final_holdout=True,
        cache_dir=cache_dir,
    )
    assert not df.empty, "load_prices(True) returned empty DataFrame"
    assert df.index.max() >= HOLDOUT_START, (
        f"guard over-restricted — max date {df.index.max()} < holdout {HOLDOUT_START}"
    )
    holdout_rows = df.index >= HOLDOUT_START
    n_hold_dates = int(holdout_rows.sum())
    assert n_hold_dates > 0, "no holdout dates present despite final_holdout=True"
    holdout_close = df.loc[holdout_rows, "Close"]
    sentinel_hits = int((np.abs(holdout_close - SENTINEL) < 1e-6).sum().sum())
    total_cells_holdout_close = int(holdout_close.shape[0] * holdout_close.shape[1])
    assert sentinel_hits == total_cells_holdout_close, (
        f"sentinel not found in holdout when enabled: "
        f"{sentinel_hits}/{total_cells_holdout_close} Close cells match"
    )


def test_holdout_guard_rejects_non_bool():
    with pytest.raises(ValueError, match="final_holdout must be a Python bool"):
        load_prices(
            ["RELIANCE.NS"],
            start_date=PRE_START,
            end_date=HOLDOUT_START,
            final_holdout=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="final_holdout must be a Python bool"):
        load_prices(
            ["RELIANCE.NS"],
            start_date=PRE_START,
            end_date=HOLDOUT_START,
            final_holdout=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="final_holdout must be a Python bool"):
        load_prices(
            ["RELIANCE.NS"],
            start_date=PRE_START,
            end_date=HOLDOUT_START,
            final_holdout=0,  # type: ignore[arg-type]
        )


def test_holdout_guard_is_keyword_only():
    import inspect

    sig = inspect.signature(load_prices)
    final_param = sig.parameters["final_holdout"]
    assert final_param.kind == inspect.Parameter.KEYWORD_ONLY, (
        "final_holdout must be KEYWORD_ONLY so callers cannot pass it positionally "
        "and accidentally set True without reading the warning"
    )
