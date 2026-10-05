from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    COV_ESTIMATOR_WINNER,
    classic_max_sharpe_weights,
    equal_weight,
    freefloat_proxy_weight,
    min_variance_weights,
    risk_parity_weights,
    _post_verify_weights,
    WEIGHT_SUM_TOL,
    WEIGHT_NONNEG_TOL,
    WEIGHT_UPPER_TOL,
)


N_ASSETS = 46
SEED = 20250104
UPPER_CAP = 0.10


def _symmetrize(x: np.ndarray) -> np.ndarray:
    return (x + x.T) / 2.0


def _random_corr_matrix(n: int, seed: int, rho_range: tuple[float, float]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.uniform(*rho_range, size=(n, n))
    a = _symmetrize(a)
    np.fill_diagonal(a, 1.0)
    try:
        L = np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(a)
        w = np.where(w > 1e-8, w, 1e-8)
        a = v @ np.diag(w) @ v.T
        d = 1.0 / np.sqrt(np.diag(a))
        a = (a * d[None, :]) * d[:, None]
        a = _symmetrize(a)
        np.fill_diagonal(a, 1.0)
    return a


def _corr_to_cov(corr: np.ndarray, vols: np.ndarray) -> np.ndarray:
    d = np.diag(vols)
    return d @ corr @ d


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture(params=["A_well_conditioned", "B_ill_conditioned", "C_near_singular"])
def synthetic_cov(request, rng) -> tuple[str, np.ndarray]:
    label = request.param
    if label == "A_well_conditioned":
        corr = _random_corr_matrix(N_ASSETS, seed=SEED, rho_range=(0.05, 0.35))
        vols = rng.uniform(0.008, 0.022, size=N_ASSETS)
        cov = _corr_to_cov(corr, vols)
        return label, cov
    if label == "B_ill_conditioned":
        k_rank = 5
        factor_loadings = rng.normal(0.0, 0.014, size=(N_ASSETS, k_rank))
        common = factor_loadings @ factor_loadings.T
        idio = np.diag(rng.uniform(1e-6, 5e-6, size=N_ASSETS))
        cov = _symmetrize(common + idio)
        return label, cov
    if label == "C_near_singular":
        corr = _random_corr_matrix(N_ASSETS, seed=SEED + 9, rho_range=(0.10, 0.40))
        vols = rng.uniform(0.010, 0.020, size=N_ASSETS)
        cov = _corr_to_cov(corr, vols)
        copy_idx = 19
        source_idx = 3
        cov[:, copy_idx] = cov[:, source_idx] * (1.0 - 1e-10)
        cov[copy_idx, :] = cov[source_idx, :] * (1.0 - 1e-10)
        cov[copy_idx, copy_idx] = cov[source_idx, source_idx]
        cov = _symmetrize(cov)
        return label, cov
    raise ValueError(f"unknown label {label}")


def _assert_integrity(w: np.ndarray, w_upper: float):
    assert np.isfinite(w).all(), "non-finite weights"
    assert abs(float(w.sum()) - 1.0) < WEIGHT_SUM_TOL, (
        f"sum(w) = {float(w.sum()):.10f}, |err| = {abs(float(w.sum())-1.0):.2e} >= tol {WEIGHT_SUM_TOL:.0e}"
    )
    assert float(w.min()) >= -WEIGHT_NONNEG_TOL, (
        f"min(w) = {float(w.min()):.2e} < -tol {WEIGHT_NONNEG_TOL:.0e}"
    )
    assert float(w.max()) <= float(w_upper) + WEIGHT_UPPER_TOL, (
        f"max(w) = {float(w.max()):.6f} > cap {w_upper} + tol {WEIGHT_UPPER_TOL:.0e}"
    )


class TestPostVerifyWeightsSanity:
    def test_sum_to_1_tiny_error_passes(self):
        n = 10
        w = np.full(n, 1.0 / n)
        w[0] += 1e-9
        out = _post_verify_weights(w, 1.0, "label")
        assert abs(float(out.sum()) - 1.0) < WEIGHT_SUM_TOL

    def test_sum_to_1_large_error_raises(self):
        w = np.array([0.8, 0.1])
        with pytest.raises(ValueError, match="sum-to-1 violated"):
            _post_verify_weights(w, 1.0, "bad")

    def test_non_finite_raises(self):
        w = np.array([np.nan, 1.0])
        with pytest.raises(ValueError, match="non-finite"):
            _post_verify_weights(w, 1.0, "bad")


class TestEqualWeight:
    @pytest.mark.parametrize("n", [1, 2, 10, 46, 100])
    def test_sum_min_max(self, n):
        w = equal_weight(n)
        assert w.shape == (n,)
        _assert_integrity(w, w_upper=1.0)
        assert abs(float(w[0]) - 1.0 / n) < 1e-12

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="n_assets must be > 0"):
            equal_weight(0)


class TestFreeFloatProxy:
    def test_increasing_rank_decreasing_weight(self, rng):
        ranks = np.arange(1, N_ASSETS + 1, dtype=float)
        w = freefloat_proxy_weight(ranks)
        _assert_integrity(w, w_upper=0.10)
        assert np.all(np.diff(w) <= 1e-12), "weights should be non-increasing with rank"

    def test_uniform_ranks_equal_weight(self):
        ranks = np.full(10, 5.0)
        w = freefloat_proxy_weight(ranks)
        _assert_integrity(w, w_upper=0.10)
        assert np.allclose(w, 1.0 / 10.0, atol=1e-10)


class TestMinVariance:
    def test_all_synth_cases_pypfopt(self, synthetic_cov):
        label, cov = synthetic_cov
        w = min_variance_weights(cov, w_upper=UPPER_CAP, solver_backend="pypfopt")
        assert w.shape == (N_ASSETS,)
        _assert_integrity(w, w_upper=UPPER_CAP)

    def test_all_synth_cases_cvxpy(self, synthetic_cov):
        label, cov = synthetic_cov
        w = min_variance_weights(cov, w_upper=UPPER_CAP, solver_backend="cvxpy")
        assert w.shape == (N_ASSETS,)
        _assert_integrity(w, w_upper=UPPER_CAP)

    def test_infeasible_cap_raises(self):
        cov = np.eye(5) * 0.01
        with pytest.raises(ValueError, match="infeasible"):
            min_variance_weights(cov, w_upper=0.15, solver_backend="pypfopt")


class TestRiskParity:
    def test_all_synth_cases(self, synthetic_cov):
        label, cov = synthetic_cov
        w = risk_parity_weights(cov)
        assert w.shape == (N_ASSETS,)
        _assert_integrity(w, w_upper=UPPER_CAP)

    def test_id_cov_equal_weights(self):
        n = 5
        cov = np.eye(n) * 0.01
        w = risk_parity_weights(cov)
        _assert_integrity(w, w_upper=1.0)
        assert np.allclose(w, 1.0 / n, atol=1e-4)


def test_cov_estimator_winner_locked_lw_or_pca():
    assert COV_ESTIMATOR_WINNER in {"LW", "PCA", "PCA(K=?)"}
    assert COV_ESTIMATOR_WINNER == "LW", (
        "Stage 1B decided Ledoit-Wolf 2026-10-04; if this fails, re-verify stage1b_decision.csv "
        "vs tests/test_optimizer_constraints.py test expectation (or updated to 'PCA')."
    )


class TestClassicMaxSharpe:
    def test_basic_no_sector_pypfopt_solves(self, rng):
        n = 12
        corr = _random_corr_matrix(n, seed=SEED + 7, rho_range=(0.05, 0.40))
        vols = rng.uniform(0.008, 0.022, size=n)
        cov = _corr_to_cov(corr, vols)
        mu_annual = rng.normal(0.08, 0.15, size=n)
        w = classic_max_sharpe_weights(
            mu=mu_annual, cov=cov,
            risk_free_rate=0.04, w_upper=0.20,
        )
        _assert_integrity(w, w_upper=0.20)

    def test_sector_constrained_within_tolerance(self, rng):
        n = 12
        corr = _random_corr_matrix(n, seed=SEED + 8, rho_range=(0.03, 0.30))
        vols = rng.uniform(0.009, 0.020, size=n)
        cov = _corr_to_cov(corr, vols)
        mu_annual = rng.normal(0.07, 0.12, size=n)
        tickers = [f"STOCK{i}" for i in range(n)]
        ticker_to_sector = pd.Series(
            ["S1"] * 4 + ["S2"] * 5 + ["S3"] * 3,
            index=tickers,
        )
        sector_targets = {"S1": 0.34, "S2": 0.41, "S3": 0.25}
        w = classic_max_sharpe_weights(
            mu=mu_annual,
            cov=cov,
            risk_free_rate=0.04,
            w_upper=0.22,
            sector_constraints={
                "sector_targets": sector_targets,
                "ticker_to_sector": ticker_to_sector,
                "sector_tolerance": 0.05,
            },
            tickers_order=tickers,
        )
        _assert_integrity(w, w_upper=0.22)
        sectors_sorted = ["S1", "S2", "S3"]
        w_series = pd.Series(w, index=tickers)
        actual = w_series.groupby(ticker_to_sector).sum()
        tol = 0.05 + 1e-5
        for s in sectors_sorted:
            target = sector_targets[s]
            assert float(actual.get(s, 0.0)) <= target + tol, (
                f"sector {s} weight {float(actual[s]):.4f} > target {target:.2f} + tol {tol:.2f}"
            )
            assert float(actual.get(s, 0.0)) >= max(0.0, target - tol), (
                f"sector {s} weight {float(actual[s]):.4f} < target {target:.2f} − tol {tol:.2f}"
            )

    def test_sector_missing_required_keys_raises(self):
        n = 10
        cov = np.eye(n) * 0.01
        mu = np.full(n, 0.10)
        with pytest.raises(ValueError, match="missing keys"):
            classic_max_sharpe_weights(
                mu=mu, cov=cov,
                sector_constraints={"sector_targets": {"S1": 1.0}},
                tickers_order=[f"T{i}" for i in range(n)],
            )

    def test_sector_no_tickers_order_raises(self):
        n = 10
        cov = np.eye(n) * 0.01
        mu = np.zeros(n)
        with pytest.raises(ValueError, match="sector_constraints provided but tickers_order"):
            classic_max_sharpe_weights(
                mu=mu, cov=cov,
                sector_constraints={
                    "sector_targets": {"S1": 0.5, "S2": 0.5},
                    "ticker_to_sector": pd.Series(
                        ["S1"] * 5 + ["S2"] * 5,
                        index=[f"T{i}" for i in range(n)],
                    ),
                },
            )

    def test_infeasible_cap_raises(self):
        n = 5
        cov = np.eye(n) * 0.01
        mu = np.ones(n)
        with pytest.raises(ValueError, match="infeasible"):
            classic_max_sharpe_weights(mu=mu, cov=cov, w_upper=0.10)


class TestSolverCrossCheckPypfoptVsCvxpy:
    @pytest.mark.parametrize("n_random", [15, 46, 80])
    def test_solver_crosscheck_rms_within_1e5(self, n_random, rng):
        corr = _random_corr_matrix(n_random, seed=SEED + n_random, rho_range=(0.02, 0.45))
        vols = rng.uniform(0.005, 0.030, size=n_random)
        cov = _corr_to_cov(corr, vols)
        cap = 0.10 if n_random * 0.10 >= 1.0 else 1.0 / n_random + 0.05
        w_pypfopt = min_variance_weights(cov, w_upper=cap, solver_backend="pypfopt")
        w_cvxpy = min_variance_weights(cov, w_upper=cap, solver_backend="cvxpy")
        rms = float(np.sqrt(np.mean((w_pypfopt - w_cvxpy) ** 2)))
        assert rms <= 5e-4, f"RMS(Δw) = {rms:.2e} > 5e-4 between pypfopt and cvxpy (n={n_random})"

    def test_crosscheck_flag_on_synthetic_passes(self):
        n = 30
        rng = np.random.default_rng(SEED + 100)
        corr = _random_corr_matrix(n, seed=SEED + 200, rho_range=(0.05, 0.40))
        vols = rng.uniform(0.008, 0.025, size=n)
        cov = _corr_to_cov(corr, vols)
        w = min_variance_weights(
            cov,
            w_upper=0.20,
            solver_backend="pypfopt",
            cross_check=True,
            cross_check_rms_tol=5e-4,
        )
        _assert_integrity(w, w_upper=0.20)
