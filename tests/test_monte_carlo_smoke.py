from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import resample_weights, simulate_scenarios


def _make_synthetic_panel(
    n_dates: int,
    tickers: list[str],
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    daily = rng.normal(loc=0.0005, scale=0.015, size=(n_dates, len(tickers)))
    log_ret = pd.DataFrame(daily, index=dates, columns=tickers, dtype=float)
    close_prices = 100.0 * np.exp(np.cumsum(daily, axis=0))
    prices = pd.DataFrame(close_prices, index=dates, columns=tickers, dtype=float)

    mu_frozen_rows = []
    for k in range(3):
        rd = dates[-60 + k * 20]
        for t_i, t in enumerate(tickers):
            mu_frozen_rows.append(
                {
                    "rebal_date": rd,
                    "ticker": t,
                    "model_family": "synthetic",
                    "mu_hat_logret_63d": float(rng.normal(0.0, 0.04)),
                    "yhat_21d_point": np.nan,
                    "train_start_date": np.nan,
                    "train_end_date": np.nan,
                    "n_train_rows": 0,
                }
            )
    mu = pd.DataFrame(mu_frozen_rows)

    rdates = mu["rebal_date"].drop_duplicates().sort_values().tolist()
    sector_names = ["FIN", "IT", "MAT", "CGS", "ENE"]
    ticker_to_sector = {t: sector_names[i % len(sector_names)] for i, t in enumerate(tickers)}

    sec_targets_per_rd = {}
    for rd in rdates:
        from collections import Counter

        cnt = Counter(ticker_to_sector[t] for t in tickers)
        total = sum(cnt.values())
        sec_targets_per_rd[pd.to_datetime(rd)] = {s: float(c) / total for s, c in cnt.items()}

    return log_ret, prices, mu, ticker_to_sector, sec_targets_per_rd, rdates


def test_monte_carlo_shape_finite_multivariate_centroid_constrained() -> None:
    tickers = [
        "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ",
        "KKK", "LLL", "MMM", "NNN", "OOO", "PPP", "QQQ", "RRR", "SSS", "TTT",
    ]
    n_tickers = len(tickers)
    n_dates = 120
    seed = 7
    assert n_tickers >= int(1.0 / 0.10 + 1e-9) + 2, "ticker count must support 10% cap"
    log_ret, prices, mu_df, ticker_to_sector, sec_targets_per_rd, rds = _make_synthetic_panel(
        n_dates=n_dates, tickers=tickers, seed=seed
    )

    resid_rows = []
    for d in log_ret.index[:60]:
        rng = np.random.default_rng(seed + d.dayofyear)
        for t in tickers:
            resid_rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "y_true": float(rng.normal(0.0, 0.05)),
                    "yhat_oos": float(rng.normal(0.0, 0.04)),
                    "residual": float(rng.normal(0.0, 0.02)),
                }
            )
    resid_df = pd.DataFrame(resid_rows)

    n_scenarios = 10
    tensor = simulate_scenarios(
        mu_frozen_df=mu_df,
        ridge_oos_resid_long_df=resid_df,
        mode="multivariate_row",
        n_scenarios=n_scenarios,
        block_size=21,
        seed=seed,
        rebal_dates_order=rds,
        tickers_order=tickers,
        scale_resid_sqrt3=True,
    )
    assert tensor.shape == (len(rds), len(tickers), n_scenarios), (
        f"shape {tensor.shape} != ({len(rds)},{len(tickers)},{n_scenarios})"
    )
    assert np.isfinite(tensor).all(), "non-finite in mu tensor"

    for k in range(min(3, n_scenarios)):
        same_draw_slice = tensor[:, :, k]
        assert same_draw_slice.shape[1] == len(tickers)

    centroid_df, raw_df, per_draw_df = resample_weights(
        rd_list=rds,
        tickers_list=tickers,
        mu_scenario_tensor=tensor,
        log_ret_panel=log_ret,
        sector_targets_per_rd=sec_targets_per_rd,
        ticker_to_sector=ticker_to_sector,
        aggregation="median",
        risk_aversion_lambda=2.0,
        n_jobs=1,
        warm_start=False,
        seed=seed,
    )

    assert len(raw_df) == len(rds) * n_scenarios * len(tickers)
    assert np.isfinite(raw_df["weight"]).all(), "raw weights contain NaN/Inf"

    grouped = raw_df.groupby(["rebal_date", "draw_id"])["weight"].sum()
    tol = 1e-8
    assert (grouped - 1.0).abs().max() < tol, f"sum to 1 fail, max diff = {(grouped - 1.0).abs().max():.3e}"

    w_max = float(raw_df["weight"].max())
    w_min = float(raw_df["weight"].min())
    assert w_max <= 0.10001, f"w_max = {w_max} > 0.10 cap"
    assert w_min >= -5e-9, f"w_min = {w_min} < 0"
