from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import FinalHoldoutDate, load_prices
from src.features import (
    FORWARD_TARGET_HORIZON_DAYS,
    align_X_y,
    make_features,
    make_targets,
    MOM_WINDOWS_DAYS,
    SMA_WINDOWS_DAYS,
    TRADING_DAYS_PER_YEAR,
    VOL_WINDOWS_DAYS,
)

STUDY_START = pd.Timestamp("2015-01-01")
DEV_END = pd.Timestamp("2023-12-31")


def main() -> int:
    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    print(f"[info] universe size N = {len(tickers)}")

    prices = load_prices(
        tickers,
        start_date=STUDY_START,
        end_date=DEV_END,
        final_holdout=False,
    )
    if prices.empty:
        print("[FAIL] load_prices returned empty DataFrame")
        return 2
    print(f"[info] raw prices shape = {prices.shape}, date range {prices.index.min().date()} -> {prices.index.max().date()}")

    close = prices["Close"]
    high = prices["High"]
    low = prices["Low"]
    volume = prices["Volume"]
    n_dates_raw = len(close.index)

    max_rolling_lead = max(
        *VOL_WINDOWS_DAYS,
        *SMA_WINDOWS_DAYS,
        *MOM_WINDOWS_DAYS,
        63,
    )
    min_expected_lead = max_rolling_lead
    min_expected_rows = n_dates_raw - min_expected_lead - FORWARD_TARGET_HORIZON_DAYS
    min_expected_rows = max(min_expected_rows, 1)
    print(f"[info] max rolling lead-in = {max_rolling_lead} d; target horizon = {FORWARD_TARGET_HORIZON_DAYS} d; lower bound N_rows ~ {min_expected_rows * len(tickers)} cells")

    print("[run ] make_features() ...")
    X_wide = make_features(close, high, low, volume)
    if X_wide.empty:
        print("[FAIL] make_features returned empty DataFrame")
        return 3
    print(f"[ok  ] features shape (long, date x ticker) = {X_wide.shape}; N_feature_cols = {len(X_wide.columns)}")

    print("[run ] make_targets() ...")
    y_long = make_targets(close, horizon_days=FORWARD_TARGET_HORIZON_DAYS)
    if y_long.empty:
        print("[FAIL] make_targets returned empty Series")
        return 4
    print(f"[ok  ] target length = {len(y_long)}")

    print("[run ] align_X_y() ...")
    X, y, common_dates = align_X_y(X_wide, y_long)

    assert not X.empty, "X empty after align"
    assert not y.empty, "y empty after align"
    assert X.index.equals(y.index), "X and y indices diverged after align"
    print(f"[ok  ] X shape = {X.shape} | y shape = {y.shape} | distinct common dates = {len(common_dates)}")

    nan_x = int(X.isna().sum().sum())
    assert nan_x == 0, f"X still has {nan_x} NaN cells after align_X_y dropna"
    nan_y = int(y.isna().sum())
    assert nan_y == 0, f"y still has {nan_y} NaN after align_X_y dropna"
    print(f"[ok  ] NaN check: X={nan_x}, y={nan_y}")

    max_common_date = common_dates.max()
    safe_boundary = FinalHoldoutDate - pd.offsets.BDay(FORWARD_TARGET_HORIZON_DAYS + 1)
    assert max_common_date < safe_boundary, (
        f"holdout leak risk! common_dates.max()={max_common_date.date()} >= "
        f"safe boundary {safe_boundary.date()} (FinalHoldout minus target horizon+1)"
    )
    assert max_common_date < FinalHoldoutDate, (
        f"hard holdout leak! max date {max_common_date.date()} >= {FinalHoldoutDate.date()}"
    )
    print(f"[ok  ] holdout-safe: max common date = {max_common_date.date()} < {safe_boundary.date()} (safe edge)")

    min_common_date = common_dates.min()
    raw_start = close.index.min()
    expected_min = raw_start + pd.offsets.BDay(max_rolling_lead + 1)
    assert min_common_date >= expected_min - pd.Timedelta(days=7), (
        f"lead-in underflow! first usable date {min_common_date.date()} < "
        f"raw_start + {max_rolling_lead + 1} BD ~ {expected_min.date()} - rolling windows probably not filled correctly"
    )
    print(f"[ok  ] lead-in check: min common date = {min_common_date.date()} >= ~{expected_min.date()} (lead-in filled)")

    pearson = X.apply(lambda c: c.corr(y) if c.std() > 1e-12 else 0.0)
    pearson_sorted = pearson.sort_values(key=lambda s: s.abs(), ascending=False)
    print("[ok  ] top-5 |corr(feature, y)|:")
    for col, val in pearson_sorted.head(5).items():
        print(f"       {col:45s}  corr = {val:+.4f}")
    print("[ok  ] bot-3 |corr(feature, y)|:")
    for col, val in pearson_sorted.tail(3).items():
        print(f"       {col:45s}  corr = {val:+.4f}")

    stds = X.std(axis=0)
    dead = (stds < 1e-10).sum()
    assert dead == 0, f"{dead} dead-constant feature columns found (std < 1e-10)"
    print(f"[ok  ] dead-constant features = {int(dead)} / {len(stds)} total")

    n_trainable = len(y)
    n_expected_dates = len(common_dates)
    n_tkrs_per_date = n_trainable // max(n_expected_dates, 1)
    print(
        f"[summary] {n_trainable} trainable cells across "
        f"{n_expected_dates} dates x ~{n_tkrs_per_date} tickers/date | "
        f"{len(X.columns)} features | horizon y = fwd {FORWARD_TARGET_HORIZON_DAYS} d log return"
    )
    print("[DONE] all 7 sanity assertions PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
