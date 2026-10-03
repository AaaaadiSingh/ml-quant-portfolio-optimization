from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

TRADING_DAYS_PER_YEAR = 252

RETURN_HORIZONS_DAYS: tuple[int, ...] = (1, 5, 21, 63, 126)
VOL_WINDOWS_DAYS: tuple[int, ...] = (21, 63, 126)
SMA_WINDOWS_DAYS: tuple[int, ...] = (20, 50)
MOM_WINDOWS_DAYS: tuple[int, ...] = (63, 126, 252)
VOLUME_MEAN_WINDOW_DAYS = 63
ILLIQUID_VOLUME_THRESHOLD = 10_000
ILLIQUID_FRAC_WINDOW_DAYS = 21
RISK_WINDOW_DAYS = 21
VAR_CONFIDENCE = 0.95

FORWARD_TARGET_HORIZON_DAYS = 21


def _log_returns(close: pd.DataFrame, d: int) -> pd.DataFrame:
    if d <= 0:
        raise ValueError(f"horizon d must be > 0, got {d}")
    return np.log(close / close.shift(d))


def _sector_map() -> pd.DataFrame:
    p = DOCS / "nifty50_sector_map.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"sector map not found at {p}; required for sector dummy features"
        )
    sm = pd.read_csv(p)
    if "sector_provisional" not in sm.columns or "ticker" not in sm.columns:
        raise KeyError(
            f"{p} must contain columns 'ticker' and 'sector_provisional'; got {list(sm.columns)}"
        )
    sm = sm[["ticker", "sector_provisional"]].dropna().drop_duplicates(subset=["ticker"])
    return sm.set_index("ticker")["sector_provisional"]


def make_features(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    if not (
        close.shape == high.shape == low.shape == volume.shape
        and close.columns.equals(high.columns)
        and close.columns.equals(low.columns)
        and close.columns.equals(volume.columns)
        and close.index.equals(high.index)
        and close.index.equals(low.index)
        and close.index.equals(volume.index)
    ):
        raise ValueError(
            "close/high/low/volume must have identical shape, columns (tickers), "
            "and DatetimeIndex. Got shapes: "
            f"close={close.shape}, high={high.shape}, low={low.shape}, vol={volume.shape}"
        )
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("price frames must have a pd.DatetimeIndex")

    tickers = list(close.columns)
    dates = close.index

    records: list[tuple[pd.Timestamp, str, dict[str, float]]] = []

    for d in RETURN_HORIZONS_DAYS:
        r = _log_returns(close, d)
        for tkr in tickers:
            for t, v in r[tkr].items():
                records.append((t, tkr, {f"ret_{d}d": float(v)}))

    for w in VOL_WINDOWS_DAYS:
        vol = _log_returns(close, 1).rolling(window=w, min_periods=w).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
        for tkr in tickers:
            for t, v in vol[tkr].items():
                records.append((t, tkr, {f"vol_ann_{w}d": float(v)}))

    sma = {w: close.rolling(window=w, min_periods=w).mean() for w in SMA_WINDOWS_DAYS}
    for w in SMA_WINDOWS_DAYS:
        ratio = (close / sma[w]) - 1.0
        for tkr in tickers:
            for t, v in ratio[tkr].items():
                records.append((t, tkr, {f"sma_cross_{w}d": float(v)}))
    w_short, w_long = SMA_WINDOWS_DAYS[0], SMA_WINDOWS_DAYS[1]
    sma_fast_slow = (sma[w_short] / sma[w_long]) - 1.0
    for tkr in tickers:
        for t, v in sma_fast_slow[tkr].items():
            records.append(
                (t, tkr, {f"sma_fastslow_{w_short}_{w_long}": float(v)})
            )

    for w in MOM_WINDOWS_DAYS:
        mom = _log_returns(close, w)
        for tkr in tickers:
            for t, v in mom[tkr].items():
                records.append((t, tkr, {f"mom_{w}d": float(v)}))
    long_w = MOM_WINDOWS_DAYS[-1]
    short_recent = 21
    mom12_1 = _log_returns(close, long_w) - _log_returns(close, short_recent)
    for tkr in tickers:
        for t, v in mom12_1[tkr].items():
            records.append(
                (t, tkr, {f"mom_{long_w}_{short_recent}_skip": float(v)})
            )

    vol_mean = volume.rolling(window=VOLUME_MEAN_WINDOW_DAYS, min_periods=VOLUME_MEAN_WINDOW_DAYS).mean()
    vol_rel = (volume / vol_mean.replace(0, np.nan)) - 1.0
    for tkr in tickers:
        for t, v in vol_rel[tkr].items():
            records.append(
                (t, tkr, {f"vol_rel_mean{VOLUME_MEAN_WINDOW_DAYS}d": float(v)})
            )
    illiquid = (volume == 0) | (volume < ILLIQUID_VOLUME_THRESHOLD)
    illiquid_frac = illiquid.rolling(
        window=ILLIQUID_FRAC_WINDOW_DAYS, min_periods=ILLIQUID_FRAC_WINDOW_DAYS
    ).mean()
    for tkr in tickers:
        for t, v in illiquid_frac[tkr].items():
            records.append(
                (t, tkr, {f"illiquid_frac_{ILLIQUID_FRAC_WINDOW_DAYS}d": float(v)})
            )

    daily_ret = _log_returns(close, 1)
    loss = -daily_ret
    var_series = loss.rolling(window=RISK_WINDOW_DAYS, min_periods=RISK_WINDOW_DAYS).quantile(
        VAR_CONFIDENCE
    )
    for tkr in tickers:
        for t, v in var_series[tkr].items():
            records.append(
                (t, tkr, {f"var95_{RISK_WINDOW_DAYS}d": float(v)})
            )
    running_max = close.rolling(window=RISK_WINDOW_DAYS, min_periods=1).max()
    drawdown = (close / running_max) - 1.0
    mdd_window = (
        drawdown.rolling(window=RISK_WINDOW_DAYS, min_periods=RISK_WINDOW_DAYS).min()
    )
    for tkr in tickers:
        for t, v in mdd_window[tkr].items():
            records.append(
                (t, tkr, {f"mdd_{RISK_WINDOW_DAYS}d": float(v)})
            )

    records_indexed: dict[tuple[pd.Timestamp, str], dict[str, float]] = {}
    for t, tkr, feat_map in records:
        key = (t, tkr)
        if key not in records_indexed:
            records_indexed[key] = {}
        records_indexed[key].update(feat_map)

    rows = []
    for (t, tkr), fm in records_indexed.items():
        row = {"date": t, "ticker": tkr}
        row.update(fm)
        rows.append(row)
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return pd.DataFrame()

    calendar = long_df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    cal_df = pd.DataFrame({"date": calendar})
    cal_df["month"] = cal_df["date"].dt.month
    cal_df["quarter"] = cal_df["date"].dt.quarter
    cal_df["half"] = ((cal_df["quarter"] - 1) // 2 + 1).astype(int)
    month_dummies = pd.get_dummies(cal_df["month"], prefix="cal_month", dtype=float)
    quarter_dummies = pd.get_dummies(cal_df["quarter"], prefix="cal_qtr", dtype=float)
    half_dummies = pd.get_dummies(cal_df["half"], prefix="cal_half", dtype=float)
    cal_dummies = pd.concat(
        [cal_df[["date"]], month_dummies, quarter_dummies, half_dummies], axis=1
    )
    long_df = long_df.merge(cal_dummies, on="date", how="left")

    sector_s = _sector_map()
    sector_frame = (
        sector_s.reindex(long_df["ticker"].unique()).reset_index()
    )
    sector_frame.columns = ["ticker", "sector"]
    sector_dummies = pd.get_dummies(sector_frame.set_index("ticker")["sector"], prefix="sector", dtype=float).reset_index()
    long_df = long_df.merge(sector_dummies, on="ticker", how="left")

    wide_base = long_df.set_index(["date", "ticker"]).sort_index()

    rank_sources = {
        "mom_126d": "cs_rank_mom126",
        "vol_ann_21d": "cs_rank_vol21",
        f"sma_cross_{SMA_WINDOWS_DAYS[0]}d": f"cs_rank_sma{SMA_WINDOWS_DAYS[0]}",
    }
    if not wide_base.empty:
        for src_col, out_col in rank_sources.items():
            if src_col not in wide_base.columns:
                continue
            s = wide_base[src_col]
            ranked = (
                s.groupby(level="date")
                .rank(method="average", ascending=True, pct=True)
            )
            z_by_date = (
                ranked.groupby(level="date").transform(
                    lambda g: (g - g.mean()) / (g.std(ddof=0) if g.std(ddof=0) > 0 else 1.0)
                )
            )
            wide_base[out_col] = z_by_date.values

    feature_cols = [c for c in wide_base.columns if c not in {"date", "ticker"}]
    for c in feature_cols:
        wide_base[c] = pd.to_numeric(wide_base[c], errors="coerce")
    return wide_base


def make_targets(
    close: pd.DataFrame,
    horizon_days: int = FORWARD_TARGET_HORIZON_DAYS,
) -> pd.Series:
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be > 0, got {horizon_days}")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must have pd.DatetimeIndex")
    fwd = np.log(close.shift(-horizon_days) / close)
    pandas_ver = tuple(int(x) for x in pd.__version__.split(".")[:2])
    if pandas_ver >= (2, 1):
        stacked = fwd.stack(future_stack=True)
    else:
        stacked = fwd.stack(dropna=False)
    if stacked.empty:
        idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return pd.Series([], index=idx, name=f"y_fwd_{horizon_days}d")
    stacked.index.names = ["date", "ticker"]
    stacked.name = f"y_fwd_{horizon_days}d"
    return stacked


def align_X_y(
    features_df: pd.DataFrame,
    target_series: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    if features_df.index.names != ["date", "ticker"]:
        raise ValueError(
            f"features_df MultiIndex must be [date, ticker]; got {features_df.index.names}"
        )
    if target_series.index.names != ["date", "ticker"]:
        raise ValueError(
            f"target_series MultiIndex must be [date, ticker]; got {target_series.index.names}"
        )
    common = features_df.index.intersection(target_series.index)
    if len(common) == 0:
        raise ValueError(
            "features and target share zero (date, ticker) rows after alignment; "
            "check that close price inputs are identical across both calls"
        )
    X = features_df.loc[common].copy()
    y = target_series.loc[common].copy()

    drop_mask = X.isna().any(axis=1) | y.isna()
    keep = ~drop_mask
    if keep.sum() == 0:
        raise ValueError(
            "align_X_y discarded every row because every row had NaN in features or target; "
            "ensure input close data spans enough dates to fill rolling windows and forward target"
        )
    X = X.loc[keep]
    y = y.loc[keep]

    X = X.sort_index()
    y = y.sort_index()

    dates = pd.DatetimeIndex(
        X.index.get_level_values("date").unique().sort_values()
    )
    return X, y, dates
