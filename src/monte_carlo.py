from __future__ import annotations

from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.covariance import ledoit_wolf_cov
from src.optimizer import (
    COV_ESTIMATOR_WINNER,
    TRADING_DAYS_PER_YEAR,
    _post_verify_weights,
    classic_max_sharpe_weights,
)
from src.optimizer import _sector_group_matrix


SEED_LOCK: int = 7
SCALE_HORIZON_63_OVER_21: float = 3.0
SCALE_RESID_SQRT: float = float(np.sqrt(SCALE_HORIZON_63_OVER_21))
WEIGHT_UPPER_LOCKED: float = 0.10
SECTOR_TOLERANCE_LOCKED: float = 0.03
RISK_FREE_ANNUAL_LOCKED: float = 0.04
RF_DAILY: float = (1.0 + RISK_FREE_ANNUAL_LOCKED) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


if COV_ESTIMATOR_WINNER != "LW":
    raise RuntimeError(
        f"[Phase5 frozen seed violation] COV_ESTIMATOR_WINNER must be LW. Got {COV_ESTIMATOR_WINNER!r}. "
        f"Refuse to run monte_carlo module."
    )


def _one_qp_solve_one_draw(
    mu_perturbed: np.ndarray,
    sigma_lw: pd.DataFrame | np.ndarray,
    tickers_order: list[str],
    sector_info: dict[str, Any],
    w_warmstart: Optional[np.ndarray] = None,
    risk_aversion_lambda: float = 2.0,
    label: str = "mc_draw",
) -> np.ndarray:
    mu_arr = np.asarray(mu_perturbed, dtype=float).flatten()
    n = len(tickers_order)
    if mu_arr.size != n:
        raise ValueError(f"{label}: mu size {mu_arr.size} != tickers_order {n}")
    if not np.isfinite(mu_arr).all():
        raise ValueError(f"{label}: mu contains non-finite")
    if isinstance(sigma_lw, pd.DataFrame):
        sig_arr = sigma_lw.loc[tickers_order, tickers_order].to_numpy(dtype=float)
    else:
        sig_arr = np.asarray(sigma_lw, dtype=float)
    if sig_arr.shape != (n, n):
        raise ValueError(f"{label}: sigma shape {sig_arr.shape} != ({n},{n})")

    solver_kws: dict[str, Any] = {}
    if w_warmstart is not None:
        ws = np.asarray(w_warmstart, dtype=float).flatten()
        if ws.size == n and np.isfinite(ws).all():
            try:
                solver_kws["solver_kws"] = {"initvals": ws}
            except Exception:
                pass

    sector_constraints = None
    if sector_info is not None:
        sector_targets = sector_info.get("sector_targets")
        ticker_to_sector = sector_info.get("ticker_to_sector")
        if sector_targets is not None and ticker_to_sector is not None:
            tol = float(sector_info.get("sector_tolerance", SECTOR_TOLERANCE_LOCKED))
            sector_constraints = {
                "sector_targets": sector_targets,
                "ticker_to_sector": ticker_to_sector,
                "sector_tolerance": tol,
            }
    w_sol = classic_max_sharpe_weights(
        mu=mu_arr,
        cov=sig_arr,
        risk_free_rate=RISK_FREE_ANNUAL_LOCKED,
        w_upper=WEIGHT_UPPER_LOCKED,
        sector_constraints=sector_constraints,
        tickers_order=tickers_order,
        **solver_kws,
    )

    post_sector_info = None
    if sector_constraints is not None:
        post_sector_info = {
            "tickers_order": list(tickers_order),
            "sector_targets": sector_constraints["sector_targets"],
            "ticker_to_sector": sector_constraints["ticker_to_sector"],
            "sector_tolerance": sector_constraints["sector_tolerance"],
        }
    w_final = _post_verify_weights(
        w_sol,
        w_upper=WEIGHT_UPPER_LOCKED,
        label=label,
        sector_info=post_sector_info,
    )
    assert np.isfinite(w_final).all(), f"{label}: non-finite after post_verify"
    assert (
        abs(float(w_final.sum()) - 1.0) < 1e-8
    ), f"{label}: sum w = {w_final.sum()}, expected 1.0"
    assert w_final.min() >= -5e-9, f"{label}: min w = {w_final.min()}"
    assert (
        w_final.max() <= WEIGHT_UPPER_LOCKED + 1e-4
    ), f"{label}: max w = {w_final.max()} > cap {WEIGHT_UPPER_LOCKED}"

    if post_sector_info is not None:
        to_sec = post_sector_info["ticker_to_sector"]
        secs, G = _sector_group_matrix(n, pd.Series(to_sec), list(tickers_order))
        targets_vec = np.array(
            [float(post_sector_info["sector_targets"][s]) for s in secs], dtype=float
        )
        s_sum = float(targets_vec.sum())
        if s_sum > 0 and abs(s_sum - 1.0) > 1e-6:
            targets_vec = targets_vec / s_sum
        drift = (G @ w_final - targets_vec) * 100.0
        tol_pp = SECTOR_TOLERANCE_LOCKED * 100.0 + 0.001
        assert (
            abs(drift).max() <= tol_pp
        ), f"{label}: sector drift pp max = {abs(drift).max():.5f} > ±{tol_pp:.3f}"

    return w_final


def _multivariate_row_draw(
    ridge_oos_resid_long_df: pd.DataFrame,
    tickers_order: list[str],
    n_scenarios: int,
    seed: int,
    n_resid_draws_per_scenario: int = 1,
) -> np.ndarray:
    required = {"date", "ticker", "residual"}
    missing = required - set(ridge_oos_resid_long_df.columns)
    if missing:
        raise ValueError(f"resid_long_df missing cols: {sorted(missing)}")
    df = ridge_oos_resid_long_df[["date", "ticker", "residual"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"].isin(set(tickers_order))]
    piv = df.pivot_table(index="date", columns="ticker", values="residual", aggfunc="first")
    piv = piv.reindex(columns=tickers_order)
    valid_rows_mask = piv.notna().all(axis=1)
    if valid_rows_mask.sum() < 20:
        raise ValueError(
            f"multivariate_row_draw: only {valid_rows_mask.sum()} dates with all {len(tickers_order)} tickers"
        )
    valid_dates = piv.index[valid_rows_mask]
    valid_panel = piv.loc[valid_dates, tickers_order].to_numpy(dtype=float)
    n_dates = valid_panel.shape[0]

    rng = np.random.default_rng(seed)
    draws_out = np.empty((n_scenarios, len(tickers_order)), dtype=float)
    for k in range(n_scenarios):
        idxs = rng.integers(0, n_dates, size=(n_resid_draws_per_scenario,))
        slice_mat = valid_panel[idxs, :]
        if n_resid_draws_per_scenario == 1:
            draws_out[k, :] = slice_mat[0, :]
        else:
            draws_out[k, :] = slice_mat.mean(axis=0)
    same_joint_checks = 10
    if same_joint_checks > n_scenarios:
        same_joint_checks = n_scenarios
    for t in range(same_joint_checks):
        assert draws_out[t, :].shape == (len(tickers_order),)
    return draws_out


def simulate_scenarios(
    mu_frozen_df: pd.DataFrame,
    ridge_oos_resid_long_df: pd.DataFrame,
    mode: Literal["iid", "block_21", "multivariate_row"],
    n_scenarios: int,
    block_size: int = 21,
    seed: int = SEED_LOCK,
    rebal_dates_order: Optional[Iterable[Any]] = None,
    tickers_order: Optional[Iterable[str]] = None,
    scale_resid_sqrt3: bool = True,
) -> np.ndarray:
    mu_piv = mu_frozen_df.pivot_table(
        index="rebal_date", columns="ticker", values="mu_hat_logret_63d", aggfunc="first"
    )
    if rebal_dates_order is not None:
        rds = [pd.to_datetime(x) if not isinstance(x, pd.Timestamp) else x for x in rebal_dates_order]
        mu_piv.index = pd.to_datetime(mu_piv.index)
        mu_piv = mu_piv.reindex(index=rds)
    else:
        rds = list(pd.to_datetime(mu_piv.index))
    if tickers_order is not None:
        tkrs = list(tickers_order)
        mu_piv = mu_piv.reindex(columns=tkrs)
    else:
        tkrs = list(mu_piv.columns)
    mu_mat = mu_piv.to_numpy(dtype=float)
    n_rd, n_tkr = mu_mat.shape

    rdf = ridge_oos_resid_long_df.copy()
    rdf["date"] = pd.to_datetime(rdf["date"])
    resid_series_flat_21d = rdf["residual"].dropna().to_numpy(dtype=float)
    resid_std_21d = float(np.nanstd(resid_series_flat_21d))
    target_63d_std = resid_std_21d * (np.sqrt(SCALE_HORIZON_63_OVER_21) if scale_resid_sqrt3 else 1.0)

    rng = np.random.default_rng(seed)
    perturb = np.zeros((n_rd, n_tkr, n_scenarios), dtype=float)

    if mode == "iid":
        for i in range(n_rd):
            samples = rng.choice(resid_series_flat_21d, size=(n_tkr, n_scenarios), replace=True)
            if scale_resid_sqrt3:
                samples = samples * SCALE_RESID_SQRT
            perturb[i, :, :] = samples
    elif mode == "block_21":
        try:
            from arch.bootstrap import CircularBlockBootstrap
        except Exception as exc:  # pragma: no cover
            raise ImportError("arch.bootstrap required for mode='block_21'") from exc
        resid_mat_tkr = rdf.pivot_table(
            index="date", columns="ticker", values="residual", aggfunc="first"
        ).reindex(columns=tkrs)
        resid_mat_tkr = resid_mat_tkr.fillna(0.0).to_numpy(dtype=float)
        n_blocks_needed = int(np.ceil(max(1, n_scenarios / block_size))) + 2
        for i in range(n_rd):
            for t in range(n_tkr):
                series = resid_mat_tkr[:, t]
                bs = CircularBlockBootstrap(block_size, series)
                acc_list: list[np.ndarray] = []
                total_len = 0
                for data_gen in bs.bootstrap(n_blocks_needed):
                    sample_block = np.asarray(data_gen[0][0], dtype=float).flatten()
                    if sample_block.size == 0:
                        continue
                    acc_list.append(sample_block)
                    total_len += sample_block.size
                    if total_len >= n_scenarios:
                        break
                if acc_list:
                    col_long = np.concatenate(acc_list) if len(acc_list) > 1 else acc_list[0]
                else:
                    col_long = np.array([], dtype=float)
                if col_long.size < n_scenarios:
                    rng_fill = np.random.default_rng(seed + t + i * n_tkr + 7)
                    pads = rng_fill.choice(series, size=(n_scenarios - col_long.size), replace=True)
                    col_long = (
                        np.concatenate([col_long, pads]) if col_long.size > 0 else pads
                    )
                col_fill = col_long[:n_scenarios].astype(float, copy=False)
                if scale_resid_sqrt3:
                    col_fill = col_fill * SCALE_RESID_SQRT
                perturb[i, t, :] = col_fill
    elif mode == "multivariate_row":
        joint_draws = _multivariate_row_draw(
            ridge_oos_resid_long_df=rdf,
            tickers_order=tkrs,
            n_scenarios=n_scenarios,
            seed=seed,
        )
        if scale_resid_sqrt3:
            joint_draws = joint_draws * SCALE_RESID_SQRT
        for i in range(n_rd):
            perturb[i, :, :] = joint_draws.T
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    scale_actual_std = float(np.nanstd(perturb))
    if scale_resid_sqrt3:
        expected_ratio = float(np.sqrt(SCALE_HORIZON_63_OVER_21))
        perturb_no_scale: np.ndarray
        if mode == "iid":
            rng2 = np.random.default_rng(seed)
            samples_iid = rng2.choice(resid_series_flat_21d, size=(n_rd, n_tkr, n_scenarios), replace=True)
            perturb_no_scale = samples_iid.astype(float)
        elif mode == "block_21":
            from arch.bootstrap import CircularBlockBootstrap

            resid_mat_tkr2 = (
                rdf.pivot_table(index="date", columns="ticker", values="residual", aggfunc="first")
                .reindex(columns=tkrs)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            perturb_no_scale = np.zeros_like(perturb)
            n_blocks2 = int(np.ceil(max(1, n_scenarios / block_size))) + 2
            for i in range(n_rd):
                for t in range(n_tkr):
                    series2 = resid_mat_tkr2[:, t]
                    bs2 = CircularBlockBootstrap(block_size, series2)
                    acc2: list[np.ndarray] = []
                    total2 = 0
                    for dg2 in bs2.bootstrap(n_blocks2):
                        sb2 = np.asarray(dg2[0][0], dtype=float).flatten()
                        if sb2.size == 0:
                            continue
                        acc2.append(sb2)
                        total2 += sb2.size
                        if total2 >= n_scenarios:
                            break
                    if acc2:
                        col_long2 = np.concatenate(acc2) if len(acc2) > 1 else acc2[0]
                    else:
                        col_long2 = np.array([], dtype=float)
                    if col_long2.size < n_scenarios:
                        rng2 = np.random.default_rng(seed + t + 17 + i * n_tkr)
                        pads2 = rng2.choice(series2, size=(n_scenarios - col_long2.size), replace=True)
                        col_long2 = (
                            np.concatenate([col_long2, pads2]) if col_long2.size > 0 else pads2
                        )
                    perturb_no_scale[i, t, :] = col_long2[:n_scenarios].astype(float, copy=False)
        elif mode == "multivariate_row":
            jd_noscale = _multivariate_row_draw(
                ridge_oos_resid_long_df=rdf,
                tickers_order=tkrs,
                n_scenarios=n_scenarios,
                seed=seed,
            )
            perturb_no_scale = np.zeros_like(perturb)
            for i in range(n_rd):
                perturb_no_scale[i, :, :] = jd_noscale.T
        else:  # pragma: no cover
            raise ValueError(mode)
        std_with = np.nanstd(perturb.astype(float))
        std_without = np.nanstd(perturb_no_scale.astype(float))
        actual_ratio = (std_with / std_without) if std_without > 0 else 0.0
        tol_ratio = 5e-2
        assert abs(actual_ratio - expected_ratio) < tol_ratio, (
            f"[SCALE√3 ASSERT FAIL] actual (with_scale / no_scale) std ratio = {actual_ratio:.9f}, "
            f"expected sqrt(3) = {expected_ratio:.9f}, diff = {abs(actual_ratio - expected_ratio):.3e}, "
            f"tol={tol_ratio}"
        )
    tensor_out = mu_mat[:, :, None] + perturb
    assert tensor_out.shape == (n_rd, n_tkr, n_scenarios)
    if not np.isfinite(tensor_out).all():
        bad = int((~np.isfinite(tensor_out)).sum())
        raise ValueError(f"simulate_scenarios: {bad} non-finite cells in output tensor")
    return tensor_out


def resample_weights(
    rd_list: Iterable[Any],
    tickers_list: Iterable[str],
    mu_scenario_tensor: np.ndarray,
    log_ret_panel: pd.DataFrame,
    sector_targets_per_rd: dict[Any, dict[str, float]],
    ticker_to_sector: dict[str, str],
    *,
    aggregation: Literal["mean", "median", "medoid_draw"],
    risk_aversion_lambda: float = 2.0,
    cov_lookback_days: int = 63,
    n_jobs: int = -1,
    joblib_backend: Optional[Literal["processes", "threads"]] = None,
    warm_start: bool = True,
    seed: int = SEED_LOCK,
    lookback_end_exclusive_shifted: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rds = [pd.to_datetime(x) if not isinstance(x, pd.Timestamp) else x for x in rd_list]
    tkrs = list(tickers_list)
    n_rd = len(rds)
    n_tkr = len(tkrs)
    n_scen = int(mu_scenario_tensor.shape[2])
    tensor_check = (n_rd, n_tkr, n_scen)
    if mu_scenario_tensor.shape != tensor_check:
        raise ValueError(
            f"mu_scenario_tensor shape {mu_scenario_tensor.shape} != expected {tensor_check}"
        )

    log_ret_panel = log_ret_panel.copy()
    log_ret_panel.index = pd.to_datetime(log_ret_panel.index)

    raw_rows: list[tuple[pd.Timestamp, int, str, float]] = []
    per_draw_sharpe_rows: list[dict[str, Any]] = []
    centroid_rows: list[tuple[pd.Timestamp, str, float]] = []

    for i, rd in enumerate(rds):
        rd_ts = pd.Timestamp(rd)
        if lookback_end_exclusive_shifted:
            end = rd_ts - pd.Timedelta(days=1)
        else:
            end = rd_ts
        start_default = end - pd.Timedelta(days=int(cov_lookback_days * 2))
        sub = log_ret_panel.loc[start_default:end, tkrs].dropna(how="any")
        min_rows_required = max(5, min(cov_lookback_days // 2, 10))
        if sub.shape[0] < min_rows_required:
            sub_before = log_ret_panel.loc[:end, tkrs].dropna(how="any")
            if sub_before.shape[0] >= min_rows_required:
                sub = sub_before
            else:
                sub_after = log_ret_panel.loc[rd_ts:, tkrs].dropna(how="any")
                if sub_before.shape[0] + sub_after.shape[0] >= min_rows_required:
                    sub = pd.concat([sub_before, sub_after.head(max(0, min_rows_required - sub_before.shape[0]))])
        if sub.shape[0] < min_rows_required:
            raise ValueError(
                f"[RD={rd_ts.date()}] only {sub.shape[0]} lookback rows for Sigma, need ≥ {min_rows_required}"
            )
        actual_lb = min(cov_lookback_days, sub.shape[0])
        sub_tail = sub.tail(actual_lb)
        cov_lw_df, _cov_meta = ledoit_wolf_cov(sub_tail)
        cov_df = cov_lw_df.loc[tkrs, tkrs].astype(float)
        sector_targets = sector_targets_per_rd.get(rd_ts, sector_targets_per_rd.get(str(rd_ts.date()), {}))
        sector_info_rd = {
            "tickers_order": tkrs,
            "sector_targets": sector_targets,
            "ticker_to_sector": ticker_to_sector,
            "sector_tolerance": SECTOR_TOLERANCE_LOCKED,
        }

        mu_rd_tensor = mu_scenario_tensor[i, :, :]

        def solve_k(k: int, prev_w: Optional[np.ndarray]) -> np.ndarray:
            mu_k = mu_rd_tensor[:, k]
            lab = f"RD={rd_ts.date()}_drw_{k:04d}"
            ww = prev_w if warm_start and prev_w is not None else None
            return _one_qp_solve_one_draw(
                mu_perturbed=mu_k,
                sigma_lw=cov_df,
                tickers_order=tkrs,
                sector_info=sector_info_rd,
                w_warmstart=ww,
                risk_aversion_lambda=risk_aversion_lambda,
                label=lab,
            )

        rng_local = np.random.default_rng(seed + i)
        _ = rng_local.random()
        prefer_kw: dict[str, str] = {}
        if joblib_backend is not None:
            prefer_kw["prefer"] = str(joblib_backend)
        all_w = Parallel(n_jobs=n_jobs, verbose=0, **prefer_kw)(
            delayed(solve_k)(k, None) for k in range(n_scen)
        )
        w_stack = np.stack(all_w, axis=0)

        mu_daily = mu_rd_tensor / TRADING_DAYS_PER_YEAR
        sig_daily = cov_df.to_numpy(dtype=float) / TRADING_DAYS_PER_YEAR
        rf_daily_col = np.full(n_tkr, RF_DAILY, dtype=float)
        approx_sharpes = np.full(n_scen, np.nan, dtype=float)
        for k in range(n_scen):
            w = w_stack[k]
            exret_daily = float(w @ (mu_daily[:, k] - rf_daily_col))
            vol_daily_sq = float(w @ sig_daily @ w)
            if vol_daily_sq > 1e-18:
                approx_sharpes[k] = exret_daily / np.sqrt(vol_daily_sq) * np.sqrt(TRADING_DAYS_PER_YEAR)

        for k in range(n_scen):
            for j, t in enumerate(tkrs):
                raw_rows.append((rd_ts, k, t, float(w_stack[k, j])))
            per_draw_sharpe_rows.append(
                {"rebal_date": rd_ts, "draw_id": k, "approx_sharpe": float(approx_sharpes[k])}
            )

        if aggregation == "mean":
            cent_vec = np.mean(w_stack, axis=0)
        elif aggregation == "median":
            cent_vec = np.median(w_stack, axis=0)
        elif aggregation == "medoid_draw":
            med_sharpe = float(np.nanmedian(approx_sharpes))
            order = sorted(range(n_scen), key=lambda kk: abs(float(approx_sharpes[kk]) - med_sharpe))
            kstar = int(order[0])
            cent_vec = w_stack[kstar].copy()
        else:
            raise ValueError(f"Unknown aggregation {aggregation!r}")
        s_raw = float(np.nansum(cent_vec))
        if abs(s_raw) < 1e-15:
            cent_vec = np.full_like(cent_vec, 1.0 / n_tkr, dtype=float)
        else:
            cent_vec = cent_vec / s_raw
        cent_vec = np.where(np.isfinite(cent_vec), cent_vec, 0.0)
        s_post = float(cent_vec.sum())
        if abs(s_post - 1.0) > 1e-9 and s_post > 0:
            cent_vec = cent_vec / s_post
        post_sec = {
            "tickers_order": tkrs,
            "sector_targets": sector_info_rd["sector_targets"],
            "ticker_to_sector": ticker_to_sector,
            "sector_tolerance": SECTOR_TOLERANCE_LOCKED,
        }
        cent_final = _post_verify_weights(
            cent_vec,
            w_upper=WEIGHT_UPPER_LOCKED,
            label=f"RD={rd_ts.date()}_centroid_{aggregation}",
            sector_info=post_sec,
        )
        if post_sec is not None and post_sec.get("tickers_order") is not None:
            secs_r, G_r = _sector_group_matrix(
                n_tkr,
                pd.Series(post_sec["ticker_to_sector"]),
                list(post_sec["tickers_order"]),
            )
            tgt_r = np.array(
                [float(post_sec["sector_targets"].get(s, 0.0)) for s in secs_r], dtype=float
            )
            if abs(float(tgt_r.sum()) - 1.0) > 1e-9 and tgt_r.sum() > 0:
                tgt_r = tgt_r / tgt_r.sum()
            tol_r = float(post_sec.get("sector_tolerance", SECTOR_TOLERANCE_LOCKED))
            lower_r = tgt_r - tol_r
            upper_r = tgt_r + tol_r
            drift_before = (G_r @ cent_final - tgt_r) * 100.0
            if abs(drift_before).max() > tol_r * 100.0 + 1e-4:
                try:
                    import cvxpy as cp

                    wv = cp.Variable(n_tkr)
                    w0 = cent_final.copy()
                    repair_prob = cp.Problem(
                        cp.Minimize(cp.sum_squares(wv - w0)),
                        constraints=[
                            cp.sum(wv) == 1.0,
                            wv >= 0.0,
                            wv <= WEIGHT_UPPER_LOCKED,
                            G_r @ wv >= lower_r - 1e-10,
                            G_r @ wv <= upper_r + 1e-10,
                        ],
                    )
                    try:
                        repair_prob.solve(solver=cp.CLARABEL, max_iter=500_000)
                    except Exception:
                        try:
                            repair_prob.solve(solver=cp.SCS, max_iters=500_000, eps=1e-12)
                        except Exception:
                            repair_prob.status = "fail"
                    if repair_prob.status in {"optimal", "optimal_inaccurate"} and wv.value is not None:
                        w_rep = np.asarray(wv.value, dtype=float).flatten()
                        s_r = float(w_rep.sum())
                        if abs(s_r) < 1e-12:
                            w_rep = np.full_like(w_rep, 1.0 / n_tkr)
                        else:
                            w_rep = w_rep / s_r
                        w_rep = np.where(w_rep < 0.0, 0.0, w_rep)
                        w_rep = np.where(w_rep > WEIGHT_UPPER_LOCKED, WEIGHT_UPPER_LOCKED, w_rep)
                        s_r2 = float(w_rep.sum())
                        if s_r2 > 0:
                            w_rep = w_rep / s_r2
                        drift_after = (G_r @ w_rep - tgt_r) * 100.0
                        if abs(drift_after).max() <= tol_r * 100.0 + 0.001:
                            cent_final = w_rep.astype(float, copy=False)
                except Exception:
                    pass
        assert abs(float(cent_final.sum()) - 1.0) < 1e-6, (
            f"RD={rd_ts.date()} centroid Σw = {cent_final.sum()}"
        )
        assert float(cent_final.max()) <= WEIGHT_UPPER_LOCKED + 1e-3, (
            f"RD={rd_ts.date()} cent max w = {cent_final.max()}"
        )
        assert float(cent_final.min()) >= -1e-6
        if post_sec is not None and post_sec.get("tickers_order") is not None:
            drift_final = (G_r @ cent_final - tgt_r) * 100.0
            assert abs(drift_final).max() <= tol_r * 100.0 + 0.001, (
                f"RD={rd_ts.date()} drift pp max = {abs(drift_final).max():.5f} > ±{tol_r * 100:.3f}"
            )
        for j, t in enumerate(tkrs):
            centroid_rows.append((rd_ts, t, float(cent_final[j])))

    columns = ["rebal_date", "draw_id", "ticker", "weight"]
    raw_df = pd.DataFrame.from_records(raw_rows, columns=columns)
    centroid_df = pd.DataFrame.from_records(
        centroid_rows, columns=["rebal_date", "ticker", "weight"]
    )
    per_draw_df = pd.DataFrame.from_records(per_draw_sharpe_rows)
    return centroid_df, raw_df, per_draw_df
