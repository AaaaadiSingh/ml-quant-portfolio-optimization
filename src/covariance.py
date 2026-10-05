from __future__ import annotations

from collections.abc import Generator
from typing import Any, Union

import numpy as np
import pandas as pd
from pypfopt import risk_models

Timelike = Union[str, pd.Timestamp]


def log_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(close_df.index, pd.DatetimeIndex):
        raise TypeError(f"close_df must have DatetimeIndex, got {type(close_df.index).__name__}")
    close_sorted = close_df.sort_index().ffill()
    ret = np.log(close_sorted / close_sorted.shift(1))
    ret = ret.iloc[1:]
    return ret


def _condition_number(mat: np.ndarray) -> float:
    s = np.linalg.svd(mat, compute_uv=False)
    if len(s) == 0 or s[0] < 1e-18:
        return float("inf")
    denom = s[-1] if s[-1] > 0 else s[0] * 1e-18
    return float(s[0] / denom)


def ledoit_wolf_cov(
    log_ret_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(log_ret_df.index, pd.DatetimeIndex):
        raise TypeError("log_ret_df must have DatetimeIndex")
    df = log_ret_df.dropna(how="all").ffill().bfill()
    if df.shape[0] < 3:
        raise ValueError(f"not enough rows for Ledoit-Wolf: got {df.shape[0]} < 3")
    tickers = list(df.columns)
    shrunk = risk_models.CovarianceShrinkage(df, returns_data=True).ledoit_wolf()
    shrinkage_delta = float(shrunk.get("shrinkage", float("nan")))
    cov_df = shrunk["cov"] if isinstance(shrunk, dict) and "cov" in shrunk else shrunk
    if not isinstance(cov_df, pd.DataFrame):
        cov_df = pd.DataFrame(np.asarray(cov_df), index=tickers, columns=tickers)
    if list(cov_df.columns) != tickers or list(cov_df.index) != tickers:
        cov_df = cov_df.reindex(index=tickers, columns=tickers)
    cov_np = cov_df.values.astype(float)
    cond = _condition_number(cov_np)
    rank = int(np.linalg.matrix_rank(cov_np, tol=1e-10))
    meta = {
        "method": "ledoit_wolf",
        "n_obs": int(df.shape[0]),
        "n_assets": int(len(tickers)),
        "shrinkage_delta": shrinkage_delta,
        "condition_number": cond,
        "rank": rank,
    }
    return cov_df, meta


def pca_factor_cov(
    log_ret_df: pd.DataFrame,
    min_explained: float = 0.85,
    max_k: int = 15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not (0.0 < min_explained <= 1.0):
        raise ValueError(f"min_explained must be in (0, 1.0], got {min_explained}")
    if max_k < 1:
        raise ValueError(f"max_k must be >= 1, got {max_k}")
    if not isinstance(log_ret_df.index, pd.DatetimeIndex):
        raise TypeError("log_ret_df must have DatetimeIndex")
    df = log_ret_df.dropna(how="all").ffill().bfill()
    if df.shape[0] < 3:
        raise ValueError(f"not enough rows for PCA cov: got {df.shape[0]} < 3")
    tickers = list(df.columns)
    n = len(tickers)
    x = df.values.astype(float)
    mu = x.mean(axis=0, keepdims=True)
    x_centered = x - mu
    sample_cov = (x_centered.T @ x_centered) / max(float(x_centered.shape[0] - 1), 1.0)
    eigvals, eigvecs = np.linalg.eigh(sample_cov)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals_pos = np.where(eigvals > 0.0, eigvals, 0.0)
    total_trace = float(eigvals_pos.sum())
    if total_trace <= 0.0:
        raise ValueError("covariance trace is non-positive — all returns are constant")
    cum_expl = np.cumsum(eigvals_pos) / total_trace
    ks = np.arange(1, len(eigvals_pos) + 1)
    k_candidates = ks[cum_expl >= float(min_explained)]
    if len(k_candidates) == 0:
        K = min(max_k, n, int(df.shape[0]))
    else:
        K = int(min(k_candidates.min(), max_k, n, int(df.shape[0])))
    K = max(1, K)
    Qk = eigvecs[:, :K]
    Lk = np.diag(eigvals_pos[:K])
    factor_part = Qk @ Lk @ Qk.T
    common_diag = np.diag(factor_part)
    sample_diag = np.diag(sample_cov)
    idio_raw = sample_diag - common_diag
    small_pos = float(eigvals_pos[K - 1]) if K <= eigvals_pos.size and eigvals_pos[K - 1] > 0 else 1e-12
    idio_diag = np.where(idio_raw > 0.0, idio_raw, np.maximum(0.01 * np.maximum(sample_diag, small_pos), small_pos))
    psi = np.diag(idio_diag)
    cov_np = factor_part + psi
    sym = (cov_np + cov_np.T) / 2.0
    cond = _condition_number(sym)
    rank = int(np.linalg.matrix_rank(sym, tol=1e-10))
    explained_ratio_per_component = (eigvals_pos[:K] / total_trace).tolist()
    explained_total_ratio = float(cum_expl[K - 1])
    cov_df = pd.DataFrame(sym, index=tickers, columns=tickers)
    meta = {
        "method": "pca_factor",
        "n_obs": int(df.shape[0]),
        "n_assets": n,
        "K": K,
        "min_explained_target": float(min_explained),
        "explained_total_ratio": explained_total_ratio,
        "explained_ratio_per_component": explained_ratio_per_component,
        "condition_number": cond,
        "rank": rank,
        "idio_mean": float(np.mean(idio_diag)),
        "factor_trace_ratio": float(np.trace(factor_part) / total_trace),
    }
    return cov_df, meta


def walk_forward_cov(
    log_ret_df: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    method: str,
    lookback_days: int = 63,
    **method_kwargs: Any,
) -> Generator[tuple[pd.Timestamp, pd.DataFrame, dict[str, Any]], None, None]:
    if method not in {"ledoit_wolf", "pca_factor"}:
        raise ValueError(f"method must be 'ledoit_wolf' or 'pca_factor', got {method!r}")
    if lookback_days < 5:
        raise ValueError(f"lookback_days must be >= 5, got {lookback_days}")
    if not isinstance(log_ret_df.index, pd.DatetimeIndex):
        raise TypeError("log_ret_df must have DatetimeIndex")
    ret_sorted = log_ret_df.sort_index()
    rebal_sorted = pd.DatetimeIndex(sorted(set(rebal_dates.tolist())))
    dates_available = ret_sorted.index
    for rd in rebal_sorted:
        cutoff = rd - pd.Timedelta(days=1)
        eligible = dates_available[dates_available <= cutoff]
        if len(eligible) == 0:
            continue
        end_pos = dates_available.get_loc(eligible[-1])
        start_idx = max(0, int(end_pos) - int(lookback_days) + 1)
        slice_ = ret_sorted.iloc[start_idx : int(end_pos) + 1]
        if slice_.shape[0] < 5:
            continue
        if method == "ledoit_wolf":
            cov_df, meta = ledoit_wolf_cov(slice_, **method_kwargs)
        else:
            cov_df, meta = pca_factor_cov(slice_, **method_kwargs)
        meta["lookback_rows_used"] = int(slice_.shape[0])
        meta["rebal_date"] = rd
        meta["lookback_end_exclusive"] = cutoff
        yield rd, cov_df, meta
