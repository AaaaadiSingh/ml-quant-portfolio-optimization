from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, objective_functions
from scipy.optimize import minimize

WEIGHT_SUM_TOL = 1e-6
WEIGHT_SUM_AUTOFIX_MAX = 0.05
WEIGHT_NONNEG_TOL = 1e-8
WEIGHT_UPPER_TOL = 1e-6
RISK_PARITY_MIN_WEIGHT = 1e-6

TRADING_DAYS_PER_YEAR = 252

COV_ESTIMATOR_WINNER: str = "LW"
COV_ESTIMATOR_WINNER_DETAIL: str = (
    "Ledoit-Wolf shrinkage (ledoit_wolf_cov). Selected 2026-10-04 Stage 1B vs PCA-factor on "
    "3-cov-sensitive-strategy mean tx-Sharpe (10 bps/turn). Mean: LW +0.0526 vs PCA +0.0518 → "
    "margin 8.3 bps-Sharpe (< 0.02 Sharpe Δ → tiebreaker 1 lower turnover: LW mean 18468 bps "
    "vs PCA 19099 bps → confirms LW). Equal Weight excluded from vote per frozen D10 (cov-agnostic). "
    "RW PCA-factor meta: min explained 85%, max K=15 per rebal. Source: stage1b_decision.csv §18."
)


def _post_verify_weights(
    w: np.ndarray,
    w_upper: float,
    label: str,
    *,
    sector_info: Optional[dict] = None,
) -> np.ndarray:
    w = np.asarray(w, dtype=float).flatten()
    if not np.isfinite(w).all():
        raise ValueError(f"{label}: weight vector contains non-finite values")
    n = len(w)

    if sector_info is not None:
        tickers_order = sector_info.get("tickers_order")
        sector_targets = sector_info.get("sector_targets")
        ticker_to_sector = sector_info.get("ticker_to_sector")
        tolerance = float(sector_info.get("sector_tolerance", 0.03))
        if (
            tickers_order is not None
            and len(tickers_order) == n
            and sector_targets is not None
            and ticker_to_sector is not None
        ):
            sectors = sorted({ticker_to_sector.get(t, f"?{i}") for i, t in enumerate(tickers_order)})
            smap = {s: i for i, s in enumerate(sectors)}
            n_sec = len(sectors)
            G = np.zeros((n_sec, n), dtype=float)
            for j, t in enumerate(tickers_order):
                G[smap[ticker_to_sector.get(t, f"?{j}")], j] = 1.0
            target_vec = np.array([float(sector_targets.get(s, 0.0)) for s in sectors], dtype=float)
            if abs(target_vec.sum() - 1.0) > 1e-6 and target_vec.sum() > 0:
                target_vec = target_vec / target_vec.sum()
            lower_b = target_vec - tolerance
            upper_b = target_vec + tolerance
            sw = G @ w
            if np.any(sw < lower_b - 1e-8) or np.any(sw > upper_b + 1e-8):
                w_var = cp.Variable(n)
                w0 = w.copy()
                prob = cp.Problem(
                    cp.Minimize(cp.sum_squares(w_var - w0)),
                    constraints=[
                        cp.sum(w_var) == 1.0,
                        w_var >= 0.0,
                        w_var <= w_upper,
                        G @ w_var >= lower_b - 1e-9,
                        G @ w_var <= upper_b + 1e-9,
                    ],
                )
                try:
                    prob.solve(
                        solver=cp.CLARABEL,
                        max_iter=400_000,
                        tol_gap_abs=1e-11,
                        tol_gap_rel=1e-11,
                        tol_feas=1e-11,
                    )
                except Exception:
                    try:
                        prob.solve(solver=cp.SCS, max_iters=400_000, eps=1e-11)
                    except Exception:
                        prob.status = "solver_fail"
                if prob.status in {"optimal", "optimal_inaccurate"} and w_var.value is not None:
                    w_new = np.asarray(w_var.value, dtype=float).flatten()
                    s = float(w_new.sum())
                    if abs(s) < 1e-12:
                        w_new[:] = 1.0 / n
                    else:
                        w_new = w_new / s
                    w_new = np.where(w_new < 0.0, 0.0, w_new)
                    w_new = np.where(w_new > w_upper, w_upper, w_new)
                    s = float(w_new.sum())
                    if s > 0:
                        w_new = w_new / s
                    sw_ok = G @ w_new
                    overshoot = float(
                        np.maximum(sw_ok - upper_b, 0.0).sum()
                        + np.maximum(lower_b - sw_ok, 0.0).sum()
                    )
                    if overshoot <= 1e-8:
                        w = w_new

    sum_err = abs(float(w.sum()) - 1.0)
    if sum_err >= WEIGHT_SUM_TOL:
        if sum_err >= WEIGHT_SUM_AUTOFIX_MAX:
            raise ValueError(
                f"{label}: sum-to-1 violated by |Σw - 1| = {sum_err:.6f} >= autofix ceiling {WEIGHT_SUM_AUTOFIX_MAX:.2f}. Σw = {float(w.sum()):.10f}. Refusing to blind-rescale a vector this far from feasible."
            )
        w = w / max(float(w.sum()), 1e-12)
        sum_err = abs(float(w.sum()) - 1.0)
        if sum_err >= WEIGHT_SUM_TOL:
            raise ValueError(
                f"{label}: sum-to-1 violated after renormalize: Σw = {float(w.sum()):.10f} (|err| = {sum_err:.2e} >= tol {WEIGHT_SUM_TOL:.0e})"
            )
    min_w = float(w.min())
    if min_w < -WEIGHT_NONNEG_TOL:
        clamped = np.where(w < 0.0, 0.0, w)
        s = float(clamped.sum())
        if abs(s) < 1e-12:
            clamped = np.full(len(clamped), 1.0 / len(clamped))
        else:
            clamped = clamped / s
        return _post_verify_weights(clamped, w_upper, label + "_clamped")
    max_w = float(w.max())
    if max_w > w_upper + WEIGHT_UPPER_TOL:
        w_c = w.copy()
        n_w = len(w_c)
        for _ in range(400):
            w_c = np.where(w_c > w_upper, w_upper, w_c)
            w_c = np.where(w_c < 0.0, 0.0, w_c)
            s = float(w_c.sum())
            if abs(s) < 1e-12:
                w_c = np.full(n_w, 1.0 / n_w)
                s = 1.0
            w_c = w_c / s
            if float(w_c.max()) <= w_upper + WEIGHT_UPPER_TOL:
                break
        if float(w_c.max()) > w_upper + WEIGHT_UPPER_TOL:
            raise ValueError(
                f"{label}: upper cap violated after 400 iter rescale: max(w) = {float(w_c.max()):.8f} > w_upper={w_upper} + tol {WEIGHT_UPPER_TOL:.0e}"
            )
        w = w_c
    return w


def equal_weight(n_assets: int) -> np.ndarray:
    if n_assets <= 0:
        raise ValueError(f"n_assets must be > 0, got {n_assets}")
    w = np.full(int(n_assets), 1.0 / float(n_assets), dtype=float)
    return _post_verify_weights(w, w_upper=1.0, label="equal_weight")


def freefloat_proxy_weight(free_float_ranks: np.ndarray) -> np.ndarray:
    ranks = np.asarray(free_float_ranks, dtype=float).flatten()
    if len(ranks) == 0:
        raise ValueError("free_float_ranks is empty")
    if (ranks <= 0).any() or not np.isfinite(ranks).all():
        raise ValueError("free_float_ranks must be strictly positive finite numbers")
    inv = 1.0 / ranks
    denom = float(inv.sum())
    n = len(ranks)
    if denom < 1e-12:
        return equal_weight(n)
    w = inv / denom
    w_upper = 0.10 if n >= 10 else 1.0
    if float(w.max()) > w_upper:
        cur = w.copy()
        for _ in range(50):
            clipped = np.clip(cur, 0.0, w_upper)
            s = float(clipped.sum())
            if s < 1e-12:
                cur = np.full(n, 1.0 / n)
                break
            cur = clipped / s
            if float(cur.max()) <= w_upper + WEIGHT_UPPER_TOL:
                break
        w = cur
    return _post_verify_weights(w, w_upper=w_upper, label="freefloat_proxy_weight")


def _min_variance_pypfopt(
    cov: np.ndarray,
    w_upper: float,
    tickers: list[str] | None = None,
) -> np.ndarray:
    n = int(cov.shape[0])
    labs = tickers if (tickers is not None and len(tickers) == n) else [f"A{i}" for i in range(n)]
    cov_df = pd.DataFrame(cov, index=labs, columns=labs)
    mu_zero = pd.Series(np.zeros(n), index=labs)
    ef = EfficientFrontier(mu_zero, cov_df, weight_bounds=(0.0, w_upper))
    ef.add_objective(objective_functions.L2_reg, gamma=1e-6)
    try:
        raw = ef.min_volatility()
    except Exception as exc:
        raise RuntimeError(f"pypfopt min_volatility failed: {exc!s}") from exc
    w_arr = np.array([float(raw[t]) for t in labs], dtype=float)
    if abs(float(w_arr.sum()) - 1.0) >= WEIGHT_SUM_TOL:
        s = float(w_arr.sum())
        if abs(s) < 1e-12:
            return equal_weight(n)
        w_arr = w_arr / s
    return w_arr


def _min_variance_cvxpy(
    cov: np.ndarray,
    w_upper: float,
) -> np.ndarray:
    n = int(cov.shape[0])
    cov_safe = np.asarray(cov, dtype=float)
    if not np.isfinite(cov_safe).all():
        raise ValueError("cov matrix contains non-finite entries")
    w_var = cp.Variable(n)
    ones = np.ones(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w_var, cp.psd_wrap(cov_safe)) + 1e-6 * cp.sum_squares(w_var)),
        [
            cp.sum(w_var) == 1.0,
            w_var >= 0.0,
            w_var <= w_upper,
        ],
    )
    try:
        prob.solve(solver=cp.CLARABEL, qcp=False, max_iter=200_000)
    except Exception:
        prob.solve(solver=cp.SCS, max_iters=100_000)
    if prob.status not in {"optimal", "optimal_inaccurate"} or w_var.value is None:
        return equal_weight(n)
    w_val = np.asarray(w_var.value, dtype=float).flatten()
    s = float(w_val.sum())
    if abs(s) < 1e-12:
        return equal_weight(n)
    w_val = w_val / s
    return w_val


def min_variance_weights(
    cov: np.ndarray,
    w_upper: float = 0.10,
    solver_backend: str = "pypfopt",
    cross_check: bool = False,
    cross_check_rms_tol: float = 5e-4,
) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square 2-D, got shape {cov.shape}")
    n = cov.shape[0]
    if n == 0:
        raise ValueError("cov is empty")
    if w_upper * n < 1.0 - 1e-9:
        raise ValueError(
            f"infeasible: n*w_upper = {n*w_upper:.4f} < 1 (sum-to-1 impossible with cap {w_upper})"
        )
    if solver_backend == "pypfopt":
        w_main = _min_variance_pypfopt(cov, w_upper)
    elif solver_backend == "cvxpy":
        w_main = _min_variance_cvxpy(cov, w_upper)
    else:
        raise ValueError(f"unknown solver_backend: {solver_backend!r}")
    if cross_check:
        alt_backend = "cvxpy" if solver_backend == "pypfopt" else "pypfopt"
        w_alt = (
            _min_variance_cvxpy(cov, w_upper)
            if alt_backend == "cvxpy"
            else _min_variance_pypfopt(cov, w_upper)
        )
        rms = float(np.sqrt(np.mean((w_main - w_alt) ** 2)))
        if rms > cross_check_rms_tol:
            raise RuntimeError(
                f"solver cross-check FAILED: RMS(Δw) = {rms:.2e} > tol {cross_check_rms_tol:.0e} "
                f"between backends {solver_backend!r} and {alt_backend!r}"
            )
    return _post_verify_weights(w_main, w_upper=w_upper, label=f"min_variance[{solver_backend}]")


def risk_parity_weights(
    cov: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 5000,
) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square 2-D, got shape {cov.shape}")
    n = cov.shape[0]
    if n == 0:
        raise ValueError("cov is empty")
    if n == 1:
        return np.array([1.0])

    def rc_obj(log_w: np.ndarray) -> float:
        w = np.exp(log_w)
        w = w / w.sum()
        sigma_w = cov @ w
        rc = w * sigma_w
        target = float(rc.mean()) if rc.size > 0 else 0.0
        return float(np.sum((rc - target) ** 2))

    diag = np.diag(cov).copy()
    diag = np.where(diag > 0.0, diag, np.nanmean(np.where(diag > 0.0, diag, np.nan)) if (diag > 0).any() else 1.0)
    inv_vol = 1.0 / np.sqrt(np.clip(diag, 1e-16, None))
    w0 = inv_vol / inv_vol.sum()
    w0 = np.clip(w0, RISK_PARITY_MIN_WEIGHT, None)
    w0 = w0 / w0.sum()
    lw0 = np.log(w0)

    try:
        res = minimize(
            rc_obj,
            lw0,
            method="L-BFGS-B",
            jac="2-point",
            options={"maxiter": max_iter, "ftol": tol, "gtol": max(1e-12, tol / 100.0)},
        )
        w_sol = np.exp(np.asarray(res.x, dtype=float))
    except Exception:
        w_sol = w0.copy()
    if not np.isfinite(w_sol).all():
        w_sol = w0.copy()
    if w_sol.sum() < 1e-12:
        w_sol = w0.copy()
    else:
        w_sol = w_sol / w_sol.sum()
    w_sol = np.clip(w_sol, RISK_PARITY_MIN_WEIGHT, None)
    if w_sol.sum() < 1e-12:
        return equal_weight(n)
    w_sol = w_sol / w_sol.sum()
    w_upper = 0.10 if n >= 10 else 1.0
    if w_sol.max() > w_upper:
        capped = np.clip(w_sol, 0.0, w_upper)
        s = capped.sum()
        if s < 1e-12:
            return equal_weight(n)
        capped = capped / s
        return _post_verify_weights(capped, w_upper=w_upper, label="risk_parity_capped")
    return _post_verify_weights(w_sol, w_upper=w_upper, label="risk_parity")


def _sector_group_matrix(
    n: int,
    ticker_to_sector: pd.Series,
    tickers_order: list[str],
) -> tuple[list[str], np.ndarray]:
    sectors = sorted({ticker_to_sector[t] for t in tickers_order})
    n_sectors = len(sectors)
    G = np.zeros((n_sectors, n), dtype=float)
    for j, t in enumerate(tickers_order):
        s = ticker_to_sector[t]
        i = sectors.index(s)
        G[i, j] = 1.0
    return sectors, G


def classic_max_sharpe_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = 0.04,
    w_upper: float = 0.10,
    sector_constraints: Optional[dict] = None,
    tickers_order: list[str] | None = None,
) -> np.ndarray:
    mu_arr = np.asarray(mu, dtype=float).flatten()
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError(f"cov must be 2D square, got {cov_arr.shape}")
    n = int(cov_arr.shape[0])
    if mu_arr.size != n:
        raise ValueError(f"mu size {mu_arr.size} != cov dim {n}")
    if tickers_order is not None and len(tickers_order) != n:
        raise ValueError(f"len(tickers_order) {len(tickers_order)} != n {n}")
    if w_upper * n < 1.0 - 1e-9:
        raise ValueError(f"n*w_upper = {n*w_upper:.4f} < 1 infeasible with cap {w_upper}")

    use_sector = bool(sector_constraints)
    if use_sector:
        if tickers_order is None:
            raise ValueError("sector_constraints provided but tickers_order is None")
        required = {"sector_targets", "ticker_to_sector"}
        missing = required - set(sector_constraints.keys())
        if missing:
            raise ValueError(f"sector_constraints missing keys: {sorted(missing)}")
        sector_targets = sector_constraints["sector_targets"]
        ticker_to_sector = sector_constraints["ticker_to_sector"]
        tolerance = float(sector_constraints.get("sector_tolerance", 0.03))
        if not hasattr(sector_targets, "__getitem__"):
            raise TypeError("sector_targets must be dict-like with one entry per sector")
        if not hasattr(ticker_to_sector, "__getitem__"):
            raise TypeError("ticker_to_sector must be dict-like mapping ticker → sector")
        sectors, G = _sector_group_matrix(n, ticker_to_sector, tickers_order)
        target_vec = np.array([float(sector_targets[s]) for s in sectors], dtype=float)
        target_sum = float(target_vec.sum())
        if target_sum > 0.0 and abs(target_sum - 1.0) > 1e-6:
            target_vec = target_vec / target_sum
        lower_b = target_vec - tolerance
        upper_b = target_vec + tolerance
        w_var = cp.Variable(n)
        rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
        mu_daily = mu_arr / TRADING_DAYS_PER_YEAR
        cov_daily = cov_arr / TRADING_DAYS_PER_YEAR
        ones = np.ones(n)
        prob = cp.Problem(
            cp.Maximize(
                w_var @ (mu_daily - rf_daily * ones)
                - 1e-6 * cp.sum_squares(w_var)
            ),
            constraints=[
                cp.sum(w_var) == 1.0,
                w_var >= 0.0,
                w_var <= w_upper,
                G @ w_var >= lower_b,
                G @ w_var <= upper_b,
                (w_var @ (mu_daily - rf_daily * ones)) >= 1e-8,
            ],
        )
        try:
            prob.solve(solver=cp.CLARABEL, max_iter=200_000)
            status = prob.status
        except Exception:
            prob.solve(solver=cp.SCS, max_iters=200_000)
            status = prob.status if hasattr(prob, "status") else "solver_exception"
        if status not in {"optimal", "optimal_inaccurate"} or w_var.value is None:
            prob = cp.Problem(
                cp.Maximize(
                    w_var @ (mu_daily - rf_daily * ones)
                    - 1e-6 * cp.sum_squares(w_var)
                ),
                constraints=[
                    cp.sum(w_var) == 1.0,
                    w_var >= 0.0,
                    w_var <= w_upper,
                    G @ w_var >= lower_b,
                    G @ w_var <= upper_b,
                ],
            )
            try:
                prob.solve(solver=cp.CLARABEL, max_iter=200_000)
                status = prob.status
            except Exception:
                try:
                    prob.solve(solver=cp.SCS, max_iters=200_000)
                    status = prob.status if hasattr(prob, "status") else "solver_exception"
                except Exception:
                    status = "all_solver_paths_failed"
            if status not in {"optimal", "optimal_inaccurate"} or w_var.value is None:
                w_var2 = cp.Variable(n)
                prob2 = cp.Problem(
                    cp.Minimize(cp.quad_form(w_var2, cp.psd_wrap(cov_arr)) + 1e-6 * cp.sum_squares(w_var2)),
                    [
                        cp.sum(w_var2) == 1.0,
                        w_var2 >= 0.0,
                        w_var2 <= w_upper,
                    ],
                )
                try:
                    prob2.solve(solver=cp.CLARABEL, max_iter=200_000)
                    if w_var2.value is None or prob2.status not in {"optimal", "optimal_inaccurate"}:
                        return equal_weight(n)
                    w_sol = np.asarray(w_var2.value, dtype=float).flatten()
                except Exception:
                    return equal_weight(n)
                s = float(w_sol.sum())
                if abs(s) < 1e-12:
                    return equal_weight(n)
                w_sol = w_sol / s
                return _post_verify_weights(w_sol, w_upper=w_upper, label="classic_max_sharpe_sector_infeasible_fell_back_minvar")
        w_sol = np.asarray(w_var.value, dtype=float).flatten()
        s = float(w_sol.sum())
        if abs(s) < 1e-12:
            return equal_weight(n)
        w_sol = w_sol / s
        return _post_verify_weights(w_sol, w_upper=w_upper, label="classic_max_sharpe[sector_cvxpy]")

    labs = tickers_order if (tickers_order is not None and len(tickers_order) == n) else [f"A{i}" for i in range(n)]
    cov_df = pd.DataFrame(cov_arr, index=labs, columns=labs)
    mu_series = pd.Series(mu_arr.astype(float), index=labs)
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    ef = EfficientFrontier(mu_series - rf_daily, cov_df, weight_bounds=(0.0, w_upper))
    ef.add_objective(objective_functions.L2_reg, gamma=1e-6)
    try:
        raw = ef.max_sharpe(risk_free_rate=0.0)
    except Exception:
        return equal_weight(n)
    w_arr = np.array([float(raw[t]) for t in labs], dtype=float)
    s = float(w_arr.sum())
    if abs(s) < 1e-12:
        return equal_weight(n)
    w_arr = w_arr / s
    return _post_verify_weights(w_arr, w_upper=w_upper, label="classic_max_sharpe[pypfopt]")
