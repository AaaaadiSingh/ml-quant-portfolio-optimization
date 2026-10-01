# CONTEXT.md
## Machine Learning-Based Portfolio Optimization for Risk-Aware Investment Decision Making
**Author:** Adi Aditya Singh (12419051723, IIOT-B2) | **Supervisor:** Dr. Ruchika Sehgal
**Institution:** University School of Automation and Robotics, GGSIPU
**Type:** Minor Academic Project / Quantitative Finance Research Pipeline
**Duration:** 16 Weeks

---

## 0. PURPOSE OF THIS DOCUMENT

This file is the single source of truth for the project — architecture, math, code structure,
pitfalls, and a week-by-week execution roadmap. It is written so that a quant developer
picking this up cold could execute the entire project without needing any other reference
material except the actual data and codebase.

---

## 1. CORE PHILOSOPHY (READ THIS BEFORE WRITING ANY CODE)

Classical Markowitz Mean-Variance Optimization (MVO) is what practitioners call an
**"error maximizer."** The optimizer doesn't just use your expected-return estimates — it
actively *exploits* the noisiest, most overconfident numbers in your input vector, because
those are mathematically the most "attractive" to an unconstrained maximizer. A tiny
positive bias in one asset's estimated return can cause the optimizer to dump the entire
portfolio into that one asset. This is not a coding bug — it's a structural property of
quadratic optimization over noisy inputs, and it is the single most important fact about
this project.

Most academic financial-ML papers stop at "did my model predict returns accurately"
(RMSE, R², directional accuracy). This project explicitly rejects that as the success
criterion. **A model is only successful here if it improves realized, walk-forward,
risk-adjusted portfolio performance — Sharpe, Sortino, Calmar, drawdown, CAGR — against
honest quantitative benchmarks.** Prediction accuracy is treated as an intermediate
diagnostic, never as the final scoreboard.

The three-layer defense against the "error maximizer" problem, in order of where they sit
in the pipeline:
1. **ML layer** — replaces naive historical-average return/volatility estimates with
   conditional, feature-driven estimates (still noisy, but hopefully *less* noisy and more
   responsive to current regime).
2. **Monte Carlo resampling layer** — treats the ML point estimate as the *center of a
   distribution*, not ground truth, and forces the optimizer to solve across hundreds of
   plausible perturbations rather than committing fully to one number.
3. **Walk-forward validation layer** — treats every historical claim about performance as
   provisional until it has survived being tested strictly out-of-sample, sequentially,
   with retraining at every step, exactly as a live deployment would experience time.

---

## 2. FULL PIPELINE ARCHITECTURE

```
[Raw Market Data: OHLCV via yfinance / NSE-BSE APIs]
              │
              ▼
[Feature Engineering: log returns, rolling volatility, momentum, technical indicators]
              │
              ▼
[ML Models: Linear Regression / Random Forest / XGBoost]
   → estimate μ̂ (expected return) and/or σ̂ (expected volatility) per asset, per rebalance date
              │
              ▼
[Monte Carlo Resampling Layer]
   → simulate N=500–1000 scenarios: μ_i = μ̂ + ε_i,  ε_i ~ residual distribution
              │
              ▼
[Portfolio Optimizer — re-solved once per simulated scenario]
   → max_w (wᵀμ̂ - r_f) / sqrt(wᵀΣ̂w)   s.t.  Σw_i = 1,  0 ≤ w_i ≤ w_max
              │
              ▼
[Aggregate: w_final = mean_i(w_i)  or  median_i(w_i)]
              │
              ▼
[Walk-Forward Backtest — repeat everything above at every rebalance date, sequentially]
              │
              ▼
[Performance Evaluation: Sharpe, Sortino, CAGR, MDD, Calmar, Monte Carlo VaR/CVaR]
   → benchmarked against Equal Weight, Min Variance, Max Sharpe (classic), Risk Parity
```

---

## 3. CORE MATHEMATICAL REFERENCE (ALL FORMULAS IN ONE PLACE)

### 3.1 Return and Risk Primitives
- Log return: `r_t = ln(P_t / P_{t-1})`
- Sample mean return (naive estimator MVO tries to improve on): `μ̂_naive = (1/T)Σr_t`
- Sample covariance matrix: `Σ̂_{ij} = Cov(r_i, r_j)` over the trailing estimation window
- Correlation: `ρ_{ij} = Σ_{ij} / (σ_i · σ_j)`
- Two-asset portfolio variance identity (the mathematical basis of diversification):
  `Var(portfolio) = w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁σ₂ρ₁₂`

### 3.2 ML Estimators

**Linear Regression**
```
r̂_{t+1} = β₀ + β₁x_{1,t} + β₂x_{2,t} + ... + β_kx_{k,t} + ε_t
minimize_β  Σ_t (r_{t+1} - r̂_{t+1})²
```

**Random Forest Regression**
```
r̂ = (1/T) Σ_{t=1}^{T} f_t(x)
```
where each `f_t` is a regression tree trained on a bootstrap sample with random feature
subsampling at each split.

**XGBoost (Gradient Boosted Trees)**
```
ŷ_i^{(t)} = Σ_{k=1}^{t} f_k(x_i)
L^{(t)} = Σ_i l(y_i, ŷ_i^{(t-1)} + f_t(x_i)) + Σ_k Ω(f_k)
Ω(f) = γT + (1/2)λ‖w‖²
```
`l` = differentiable loss (squared error), `T` = number of leaves, `w` = leaf weights,
`γ, λ` = regularization strength controlling tree complexity (directly relevant to
preventing overfitting on noisy financial targets).

### 3.3 Portfolio Optimizers

**Maximum Sharpe (Mean-Variance)**
```
max_w  (wᵀμ̂ - r_f) / sqrt(wᵀΣ̂w)
s.t.   Σ_i w_i = 1,   0 ≤ w_i ≤ w_max
```

**Global Minimum Variance**
```
min_w  wᵀΣ̂w   s.t.  Σ_i w_i = 1
```

**Risk Parity** (equal risk contribution)
```
w_i · (Σ̂w)_i = w_j · (Σ̂w)_j   for all i, j
```

### 3.4 Monte Carlo Resampling of Portfolio Weights
```
μ_i = μ̂ + ε_i,   ε_i ~ (empirical or parametric) residual distribution,  i = 1,...,N
w_i = Optimizer(μ_i, Σ̂)
w_final = (1/N) Σ_{i=1}^{N} w_i     (or median_i w_i)
```
N is typically 500–1000. The residual distribution should be drawn from the **ML model's
own out-of-sample residuals** (empirical bootstrap) rather than assumed Gaussian, unless
a normality test on residuals genuinely supports it.

### 3.5 Monte Carlo VaR and CVaR
Given N simulated portfolio return paths `{R_1, ..., R_N}`:
```
VaR_α  = -Quantile_α({R_1, ..., R_N})
CVaR_α = -E[R | R ≤ -VaR_α]
```

### 3.6 Performance Metrics
```
Sharpe   = (R_p - R_f) / σ_p
Sortino  = (R_p - R_f) / σ_d              (σ_d = downside deviation)
CAGR     = (V_end / V_start)^(1/n) - 1
MDD      = min_t [ (V_t - max_{s≤t} V_s) / max_{s≤t} V_s ]
Calmar   = CAGR / |MDD|
```

---

## 4. TECH STACK — EXACT MODULES

| Purpose | Library | Key sub-modules / functions |
|---|---|---|
| Data wrangling | `pandas`, `numpy` | `pd.DataFrame`, `rolling()`, `pct_change()`, `np.log` |
| Data acquisition | `yfinance` | `yf.download()`, `yf.Ticker().history()` |
| ML models | `scikit-learn` | `linear_model.LinearRegression`, `ensemble.RandomForestRegressor`, `model_selection.TimeSeriesSplit` |
| Gradient boosting | `xgboost` | `XGBRegressor`, `early_stopping_rounds`, `eval_set` |
| Portfolio optimization | `PyPortfolioOpt` | `EfficientFrontier`, `risk_models.CovarianceShrinkage`, `expected_returns` |
| Custom/constrained optimization | `cvxpy` | `cp.Variable`, `cp.Maximize`, `cp.quad_form`, `cp.Problem` |
| Numerical optimization fallback | `scipy.optimize` | `minimize`, `SLSQP`, `NonlinearConstraint` |
| Statistics / distributions | `scipy.stats` | `norm`, `t`, `skew`, `kurtosis`, `jarque_bera` |
| Bootstrap / resampling | `arch.bootstrap` | `StationaryBootstrap`, `CircularBlockBootstrap` |
| Parallelizing MC simulations | `joblib` | `Parallel`, `delayed` |
| Backtesting engine | `vectorbt` or `backtrader` | portfolio simulation, rebalancing logic |
| Visualization | `matplotlib`, `seaborn` | equity curves, drawdown charts, heatmaps |
| Reproducibility | `numpy.random.default_rng` | seeded RNG for every MC run |

---

## 5. REPO STRUCTURE

```
portfolio-optimization-ml/
├── README.md
├── CONTEXT.md                    ← this file
├── CHANGELOG.md
├── requirements.txt
├── docs/
│   ├── synopsis.pdf
│   └── literature_matrix.md
├── data/
│   ├── raw/                      ← untouched OHLCV pulls
│   └── processed/                ← engineered feature tables
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_portfolios.ipynb
│   ├── 04_ml_models.ipynb
│   ├── 05_monte_carlo_resampling.ipynb
│   ├── 06_backtesting.ipynb
│   └── 07_robustness_checks.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── monte_carlo.py
│   ├── optimizer.py
│   ├── backtest_engine.py
│   └── metrics.py
├── tests/
│   ├── test_no_lookahead.py
│   ├── test_optimizer_constraints.py
│   └── test_metrics.py
├── results/
│   ├── figures/
│   └── metrics_summary.csv
```

---

## 6. THE 16-WEEK ROADMAP

---

### PHASE 1 — FOUNDATIONS (Weeks 1–2)
**Goal:** Literature grounding, dataset finalized, environment fully reproducible.

**Math/Statistical concepts**
- Distributional properties of financial returns: fat tails, skewness, kurtosis, volatility
  clustering, non-stationarity.
- Formal statement of the estimation-error problem in MVO (Michaud, 1989) — understand
  *why* small μ̂ perturbations cause large w swings (sensitivity is proportional to Σ̂⁻¹).

**Libraries/setup**
- `conda`/`venv` environment; pin versions in `requirements.txt`.
- `yfinance` smoke test — pull 1 ticker, confirm OHLCV integrity.
- Git repo initialized with the structure in Section 5, first commit.

**Pitfalls to avoid**
- Don't finalize a stock universe using *today's* index constituents only — flag
  survivorship bias explicitly in your literature matrix and in the final write-up even if
  you don't fully correct for it.
- Don't skip building the literature matrix — without it, your "gap" in the Problem
  Statement is just an assumption, not a finding.

**Deliverables / checkpoints**
- [ ] `docs/literature_matrix.md` populated with 15–30 papers, each tagged by theme
      (MVO limitations / ML forecasting / resampling / backtesting rigor / risk metrics).
- [ ] Final stock universe list frozen (e.g., 30–50 tickers) with rationale documented.
- [ ] Full date range decided (recommend 10+ years) with an explicit **embargo period** —
      the final 12–18 months of data marked untouched until Phase 6.
- [ ] `environment.yml` / `requirements.txt` committed; repo runs cleanly on a fresh clone.

---

### PHASE 2 — DATA & FEATURE ENGINEERING (Weeks 3–4)
**Goal:** Clean, leak-free feature tables for every stock, every day.

**Math/Statistical concepts**
- Log returns as the additive-over-time return measure.
- Rolling estimators: rolling mean, rolling std (volatility), rolling covariance — and the
  **window-length tradeoff** (short window = responsive but noisy; long window = stable
  but stale).
- Momentum formalized as trailing N-period cumulative log return.
- Technical indicators: RSI (Wilder's smoothing), MACD (EMA differences), Bollinger Bands
  (rolling mean ± k·rolling std).

**Libraries/modules**
- `pandas.DataFrame.rolling()`, `.ewm()` for exponential weighting.
- `numpy.log`, `numpy.diff` for log returns.
- Consider `ta` or `pandas_ta` for technical indicators instead of hand-rolling all of
  them (but hand-roll at least one, e.g. RSI, to actually understand the mechanics).

**Pitfalls to avoid — this is the single highest-risk phase for silent bugs**
- **Look-ahead bias via rolling windows**: `rolling(window=20).mean()` on a column
  *without* shifting will include the current day's own return in a feature meant to
  predict that same day. Always verify: feature at time `t` must only use data through
  `t-1` (i.e., `.shift(1)` after rolling, or compute the rolling window ending at `t-1`).
- **Corporate actions**: unadjusted prices around stock splits/dividends look like crashes
  or spikes. Use adjusted close, verify against a known split event.
- **Scale invariance traps**: features on wildly different scales (raw volume vs. RSI
  0–100 vs. returns ~0.01) will distort tree-based *feature importance* interpretation and
  can hurt linear regression conditioning — standardize/normalize explicitly and document
  the choice.
- **NaNs from rolling windows**: the first `window-1` rows of every rolling feature are
  NaN — decide explicitly (drop rows / forward-fill / mask) rather than let a library
  silently do it.

**Deliverables / checkpoints**
- [ ] `src/features.py` with unit-testable functions, each taking a price series and
      returning a leak-free feature series.
- [ ] `tests/test_no_lookahead.py` — a synthetic test: inject a known future value into a
      series and assert the feature at time `t` does *not* change when that future value
      changes. This test must pass before any modeling begins.
- [ ] Final processed feature table saved to `data/processed/`, one row per (date, ticker).
- [ ] Exploratory notebook showing feature distributions, correlation heatmap between
      features, and a check for fat tails/skew in the raw return series (via
      `scipy.stats.skew`, `kurtosis`, `jarque_bera`).

---

### PHASE 3 — QUANTITATIVE BASELINES (Weeks 5–6)
**Goal:** All four benchmark strategies implemented and validated *before* any ML enters
the picture — you need a trustworthy yardstick before you can claim to beat it.

**Math/Statistical concepts**
- Equal Weight: `w_i = 1/N` — no estimation, the "free" benchmark.
- Global Minimum Variance: `min_w wᵀΣ̂w` s.t. `Σw_i = 1` — a convex QP, provably solvable.
- Classic Max Sharpe: same formula as your ML-driven optimizer, but with `μ̂` = naive
  historical sample mean — this is the "control group" that isolates the effect of using
  ML for μ̂ vs. not.
- Risk Parity: solved iteratively — no closed form in general; typically via
  sequential convex optimization or a fixed-point iteration on risk contributions.
- Covariance estimation issues: sample covariance is often ill-conditioned or singular
  when `N assets > T observations`. Introduce **shrinkage estimators** now (Ledoit-Wolf)
  since you'll need them again in every later phase.

**Libraries/modules**
- `PyPortfolioOpt.EfficientFrontier` for Max Sharpe and Min Variance (built-in, well
  tested — don't hand-roll the QP if a validated library exists).
- `PyPortfolioOpt.risk_models.CovarianceShrinkage().ledoit_wolf()` for a stable Σ̂.
- `cvxpy` as a manual fallback/cross-check — solve the same QP by hand and confirm
  `PyPortfolioOpt` agrees, as a correctness sanity check.
- Custom Risk Parity solver — iterative algorithm (e.g., Newton's method on risk
  contribution equations, or `scipy.optimize.minimize` on a risk-contribution-error
  objective).

**Pitfalls to avoid**
- **Covariance error maximization**: even at the baseline stage, an ill-conditioned Σ̂ can
  cause Max Sharpe to produce extreme concentrated/leveraged-looking weights. If this
  happens with *historical* inputs, it will only get worse with noisier ML inputs later —
  fix Σ̂ estimation (shrinkage) now, not later.
- **Constraint bugs**: verify `Σw_i = 1` and `0 ≤ w_i ≤ w_max` actually hold post-solve
  (floating point can leave weights summing to 0.9999998 — decide a tolerance).
- **Silent solver failure**: `cvxpy`/`scipy.optimize` can return a "successful" status
  while having actually failed to converge on a degenerate problem — always check solver
  status flags explicitly, never assume success.

**Deliverables / checkpoints**
- [ ] `src/optimizer.py` with one function per strategy (`equal_weight()`,
      `min_variance()`, `max_sharpe()`, `risk_parity()`), each returning a weights vector
      and each unit-tested against constraint satisfaction.
- [ ] `tests/test_optimizer_constraints.py` passing for all four strategies across at
      least 3 different synthetic covariance matrices (well-conditioned, ill-conditioned,
      near-singular).
- [ ] A single historical run (no walk-forward yet) computing all four strategies' weights
      on your full dataset, purely to sanity-check they look reasonable (no single asset
      at 100%, no negative weights unless short-selling is explicitly intended).

---

### PHASE 4 — ML MODEL DEVELOPMENT (Weeks 7–9)
**Goal:** Trained, validated Linear Regression / Random Forest / XGBoost models producing
μ̂ (and optionally σ̂) — evaluated honestly, with leakage impossible by construction.

**Math/Statistical concepts**
- Bias-variance tradeoff, explicitly connected to financial data: high-variance models
  (deep trees, high-degree polynomial features) will fit noise in a domain where the
  signal-to-noise ratio is notoriously low (~single-digit % of return variance is
  typically explainable at all).
- Regularization: L2 penalty in linear models; `max_depth`, `min_child_weight`,
  `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha` in XGBoost — these aren't
  optional tuning knobs here, they're the primary defense against overfitting to
  financial noise.
- `TimeSeriesSplit` cross-validation — and why **k-fold CV with shuffling is invalid**
  for this data (it trains on the future to predict the past, guaranteeing leakage).
- Feature importance / SHAP values as a diagnostic (not as the goal) — to sanity-check
  that the model isn't relying on a feature that itself contains leaked information.

**Libraries/modules**
- `sklearn.linear_model.LinearRegression`, `Ridge` (for regularized baseline).
- `sklearn.ensemble.RandomForestRegressor` — tune `n_estimators`, `max_depth`,
  `min_samples_leaf`.
- `xgboost.XGBRegressor` — use `early_stopping_rounds` against a validation fold, not the
  test fold.
- `sklearn.model_selection.TimeSeriesSplit` for all cross-validation — never
  `train_test_split(shuffle=True)`.
- `shap` (optional but recommended) for model interpretability diagnostics.

**Pitfalls to avoid — the highest-stakes phase for methodological integrity**
- **Data leakage via feature scaling**: if you standardize/normalize features, fit the
  scaler *only* on the training fold and apply it to the validation/test fold — fitting
  on the full dataset before splitting leaks future distributional information.
- **Leakage via target construction**: predicting `r_{t+1}` using a feature computed with
  a window that accidentally includes `t+1`'s data (re-verify Phase 2's leak tests here
  against the actual target alignment, not just the raw feature).
- **Hyperparameter tuning on the test set**: tune only on a validation fold *within* the
  training period; the final embargoed out-of-sample period (Phase 1) must never
  influence a single hyperparameter choice.
- **Overfitting to a specific asset**: decide explicitly whether you're training one
  global model across all assets (pooled panel) or one model per asset — pooled models
  generalize better with limited data per asset, which is usually the right call for a
  minor project's data volume.
- **Ignoring residual diagnostics**: after training, plot residuals over time — if
  residual variance changes dramatically across regimes (heteroskedasticity), your later
  Monte Carlo noise model (Phase 5) needs to account for this rather than assuming a
  single fixed residual distribution.

**Deliverables / checkpoints**
- [ ] `src/models.py` with a common interface (`fit(X_train, y_train)`,
      `predict(X_test)`) across all three model types, so the pipeline can swap models
      without changing downstream code.
- [ ] Cross-validated performance table (RMSE, R², directional accuracy) per model —
      documented as a diagnostic, explicitly labeled "not the success metric" in your
      write-up.
- [ ] Residual distributions extracted and saved per model (needed directly as the `ε_i`
      source in Phase 5) — include a normality test (`scipy.stats.jarque_bera` or
      `shapiro`) to justify empirical vs. parametric resampling choice.
- [ ] Final model selection decision documented (which model or ensemble becomes the
      pipeline's default μ̂ estimator going into Phase 5) with justification tied to
      *portfolio-relevant* criteria, not just RMSE.

---

### PHASE 5 — MONTE CARLO INTEGRATION (Weeks 10–11)
**Goal:** The estimation-error mitigation layer — the project's core methodological
contribution — fully operational and benchmarked against the naive single-point-estimate
approach.

**Math/Statistical concepts**
- Residual bootstrap: drawing `ε_i` with replacement from the model's empirical residuals,
  vs. parametric resampling (fit a `scipy.stats.t` or `norm` distribution to residuals and
  sample from it) — implement both, compare.
- Block bootstrap / stationary bootstrap for *time-series* resampling (used later for
  Phase 7's synthetic market paths, but the sampling mechanics should be built here):
  preserves autocorrelation structure that naive iid resampling destroys.
- Convergence behavior of the Monte Carlo mean: `w_final` variance shrinks as
  `O(1/sqrt(N))` — justify your choice of N (500–1000) with an actual convergence plot
  (`w_final` estimate vs. N), not just citing the literature's typical range.
- Mean vs. median aggregation of `w_i` — median is more robust to occasional degenerate
  optimizer solutions (numerical outliers); document which you use and why.

**Libraries/modules**
- `numpy.random.default_rng(seed=...)` — always seeded for reproducibility, never bare
  `np.random`.
- `arch.bootstrap.StationaryBootstrap` / `CircularBlockBootstrap` for block-resampling of
  residuals or returns.
- `joblib.Parallel(n_jobs=-1)` + `delayed()` to parallelize the N optimizer re-solves —
  this loop is the computational bottleneck of the entire pipeline; don't leave it
  single-threaded.
- `scipy.stats` distribution fitting (`t.fit`, `norm.fit`) if going parametric.

**Pitfalls to avoid**
- **Re-solving the optimizer N times naively in a Python for-loop** will be extremely
  slow at scale (N × rebalance dates × backtest length) — parallelize from the start, and
  consider a warm-start strategy (initializing each QP solve from the previous scenario's
  solution) if using `cvxpy`.
- **Confusing residual scale with return scale**: if your ML model predicts *daily*
  returns but your rebalance period is monthly, ensure `ε_i` is scaled/aggregated to the
  same horizon as `μ̂` before adding — mixing scales silently corrupts every downstream
  number.
- **Ignoring correlation between residuals across assets**: sampling each asset's `ε_i`
  independently destroys the actual cross-asset residual correlation structure (assets in
  the same sector likely have correlated forecast errors) — consider a **multivariate**
  residual resampling (jointly resample rows of the residual matrix across assets for the
  same historical date) rather than resampling each asset's residual independently.
- **Not validating that Monte Carlo actually helps**: run a controlled comparison —
  single-point-estimate optimizer vs. Monte Carlo-averaged optimizer — on the *same*
  historical window, and confirm the MC version's weights are measurably less concentrated
  / less volatile across adjacent rebalance dates before assuming the method works.

**Deliverables / checkpoints**
- [ ] `src/monte_carlo.py`: a `simulate_scenarios()` function (returns N perturbed μ
      vectors, respecting cross-asset residual correlation) and a `resample_weights()`
      function (re-solves optimizer N times, returns aggregated `w_final`).
- [ ] Convergence diagnostic plot: `w_final` stability vs. N, justifying the chosen N.
- [ ] A/B comparison notebook: point-estimate weights vs. MC-resampled weights on
      identical inputs, with a quantified "stability improvement" metric (e.g., weight
      turnover or variance of weights across a rolling set of adjacent rebalance dates).
- [ ] Parallelized MC loop benchmarked for runtime (document seconds per rebalance date at
      N=1000) — this number directly determines Phase 6's total backtest runtime budget.

---

### PHASE 6 — WALK-FORWARD BACKTESTING & EVALUATION (Weeks 12–13)
**Goal:** The full pipeline (ML → MC → Optimizer) running end-to-end, sequentially,
across the entire non-embargoed history, producing a clean performance comparison against
all four baselines.

**Math/Statistical concepts**
- Walk-forward scheme formalized: for each rebalance date `t`, train ML models on data up
  to `t - lookback`, generate `μ̂_t`, run MC + optimizer, hold weights until `t+1`, then
  roll forward and retrain. Decide and document: expanding window (all history to date)
  vs. rolling window (fixed lookback) — each has different regime-adaptation properties.
- Portfolio return accounting: `R_p,t = Σ_i w_i,t · r_i,t` for the holding period, with
  explicit handling of weight drift *between* rebalances (weights don't stay constant as
  asset prices move within a holding period unless you rebalance continuously).
- Deflated Sharpe Ratio / Probability of Backtest Overfitting (Bailey et al.) — since
  you're testing three ML models × MC vs. non-MC × your own strategy vs. four baselines,
  you are implicitly conducting multiple comparisons; a raw Sharpe Ratio comparison
  without correction risks a false-discovery-style overfitting conclusion.

**Libraries/modules**
- `vectorbt` for vectorized backtest simulation (fast, good for parameter sweeps) or
  `backtrader` for an event-driven engine (more realistic execution modeling, slower) —
  pick one and justify the choice (vectorbt recommended for a minor project's timeline).
- `sklearn.model_selection.TimeSeriesSplit` reused here to generate the actual sequence
  of train/rebalance date pairs driving the loop.
- Custom `src/backtest_engine.py` orchestrating: feature slice → model retrain → MC
  resampling → optimizer → weight application → return accrual, per rebalance date.

**Pitfalls to avoid**
- **Retraining leakage across the walk-forward loop**: at each step, confirm that the ML
  model retrain call only sees data strictly before the current rebalance date — a common
  bug is accidentally passing the *full* feature dataframe instead of a properly sliced
  one to the retrain function.
- **Ignoring transaction costs**: a backtest with zero transaction cost assumption will
  systematically favor strategies with high turnover (which the Monte Carlo-averaged
  strategy vs. a naive point estimate may differ on) — model at least a simple
  proportional cost (e.g., 10bps per unit of turnover) even in this phase, not just in
  Phase 7's robustness checks, since it affects your headline comparison.
- **Rebalance-date weight drift accounting bug**: failing to let weights drift with asset
  price movements between rebalances (i.e., incorrectly treating weights as constant
  throughout the holding period) silently misstates realized portfolio returns.
- **Benchmarking asymmetry**: make sure baselines are *also* walk-forward validated with
  their own naive parameter re-estimation at each rebalance date — comparing your
  walk-forward ML strategy against a baseline computed once on the full sample is an
  unfair, invalid comparison.

**Deliverables / checkpoints**
- [ ] `src/backtest_engine.py` fully operational, producing a time series of portfolio
      value for: your ML+MC strategy, and all four baselines, over the same walk-forward
      period.
- [ ] `src/metrics.py` computing Sharpe, Sortino, CAGR, MDD, Calmar, and Monte Carlo
      VaR/CVaR (via simulated portfolio return distribution) for every strategy.
- [ ] A single consolidated results table (strategies × metrics) plus equity curve and
      drawdown charts for all five strategies on one plot.
- [ ] Deflated Sharpe Ratio or a multiple-comparisons-aware significance check applied to
      the headline "ML+MC beats baselines" claim before it's treated as a finding.

---

### PHASE 7 — ROBUSTNESS & STRESS TESTING (Week 14)
**Goal:** Confirm the result isn't an artifact of one particular historical path,
transaction cost assumption, or market regime.

**Math/Statistical concepts**
- Block bootstrap / stationary bootstrap of historical returns to generate synthetic
  market paths — reusing the `arch.bootstrap` machinery built in Phase 5, applied now to
  *returns* rather than residuals, to regenerate entire alternate histories.
- Regime segmentation: split the backtest period into identifiable sub-regimes (e.g., by
  realized volatility terciles, or by known macro events) and report metrics
  *per-regime*, not just pooled — a strategy that only wins in calm markets is a very
  different finding from one that's robust across regimes.
- Sensitivity analysis on transaction cost assumptions (e.g., 0bps / 10bps / 25bps / 50bps
  per unit turnover) — report how much of any outperformance survives realistic costs.

**Libraries/modules**
- `arch.bootstrap.StationaryBootstrap` (again) applied at the *multi-asset return matrix*
  level to generate B synthetic histories (B ≈ 200–500 is typical, less than the N used
  for weight-resampling since this is a slower, full-backtest-per-path operation).
- `joblib.Parallel` again, since each synthetic path requires re-running the entire
  Phase 6 backtest loop — this is the most computationally expensive phase in the project.

**Pitfalls to avoid**
- **Bootstrap sample too short to break block structure**: choose block length for
  `StationaryBootstrap` based on the autocorrelation length of realized volatility
  (typically inferred from an ACF plot of squared returns) — a block that's too short
  destroys the volatility-clustering structure you're trying to preserve.
- **Running full stress tests before the core backtest (Phase 6) is fully debugged**:
  this phase multiplies Phase 6's runtime by B — do not start here until Phase 6's
  results are trusted, or you will spend the week re-running expensive stress tests after
  every core-pipeline bugfix.
- **Cherry-picking regimes**: define regime boundaries *before* looking at how each
  strategy performs within them, or the regime analysis becomes a post-hoc rationalization
  rather than a genuine robustness check.

**Deliverables / checkpoints**
- [ ] Distribution of Sharpe/Sortino/CAGR/MDD across B synthetic market paths, for every
      strategy — reported as a distribution (mean ± std, or a box plot), not a single
      number.
- [ ] Per-regime performance table (e.g., high-vol vs. low-vol sub-periods).
- [ ] Transaction-cost sensitivity table showing how the headline result changes across
      the cost assumptions tested.

---

### PHASE 8 — RESULTS COMPILATION & DOCUMENTATION (Weeks 15–16)
**Goal:** A complete, honest, reproducible write-up — including negative or mixed
findings if that's what the data actually shows.

**What "done" looks like**
- Every claim in the final report traceable to a specific notebook cell / script output.
- An explicit, undefended statement of limitations: universe size, survivorship bias,
  transaction cost model simplifications, the specific embargo period used.
- A clear verdict: did ML + Monte Carlo resampling improve realized, walk-forward,
  risk-adjusted performance over the four baselines, after accounting for multiple
  comparisons and transaction costs — yes, no, or "it depends on regime," with numbers
  backing whichever answer the evidence actually supports.

**Deliverables / checkpoints**
- [ ] `results/metrics_summary.csv` and `results/figures/` fully populated and referenced
      in the final write-up.
- [ ] `README.md` finalized with a narrative summary and headline result charts embedded.
- [ ] `CHANGELOG.md` complete, telling the week-by-week story of the project for anyone
      auditing the GitHub history.
- [ ] Final synopsis/report document assembled, formulas and diagrams cross-checked
      against the actual implemented code (not just the original proposal).

---

## 7. MASTER PITFALLS CHECKLIST (CONSOLIDATED — REVIEW BEFORE EVERY MERGE)

- [ ] No feature at time `t` uses information only available at `t` or later.
- [ ] No scaler/normalizer/model is fit on anything other than the current training fold.
- [ ] No cross-validation uses random shuffling on time-series data.
- [ ] Every rolling-window feature's initial NaN rows are explicitly handled.
- [ ] Corporate actions (splits/dividends) are adjusted for in price data.
- [ ] Covariance matrices are shrinkage-estimated, not raw sample covariance, whenever
      `N assets` approaches `T observations`.
- [ ] Every optimizer solve's status/convergence flag is checked, not assumed.
- [ ] Constraint satisfaction (`Σw=1`, bounds) is verified post-solve, not just requested.
- [ ] Monte Carlo residuals preserve cross-asset correlation structure (not sampled
      independently per asset) unless independence is explicitly justified.
- [ ] Transaction costs are modeled in the core backtest, not only in robustness checks.
- [ ] Baselines are walk-forward validated with the same rigor as the ML strategy.
- [ ] Multiple-comparisons awareness (Deflated Sharpe / PBO) applied before declaring a
      winner.
- [ ] Every RNG call is seeded for reproducibility.
- [ ] The embargoed out-of-sample period was never touched during model selection or
      hyperparameter tuning.

---

## 8. GLOSSARY (QUICK REFERENCE)

| Term | Meaning |
|---|---|
| μ̂ | Estimated expected return vector |
| Σ̂ | Estimated covariance matrix |
| Look-ahead bias | Using information not yet available at decision time |
| Data leakage | Any pipeline step that lets future/test information influence training |
| Walk-forward validation | Sequential train→predict→roll-forward validation scheme |
| Shrinkage estimator | A covariance estimator blended with a structured target to reduce estimation noise (e.g., Ledoit-Wolf) |
| Block bootstrap | Resampling contiguous blocks of a time series to preserve autocorrelation |
| VaR / CVaR | Value at Risk / Conditional VaR (Expected Shortfall) — tail-risk measures |
| Deflated Sharpe Ratio | Sharpe Ratio adjusted for the number of strategies/comparisons tested |
| Survivorship bias | Bias from only including assets that still exist today in a backtest universe |

---

## 9. REFERENCES

[1] H. Markowitz, "Portfolio Selection," *The Journal of Finance*, vol. 7, no. 1, pp. 77–91, 1952.
[2] R. O. Michaud, "The Markowitz Optimization Enigma: Is Optimized Optimal?" *Financial Analysts Journal*, vol. 45, no. 1, pp. 31–42, 1989.
[3] R. O. Michaud, *Efficient Asset Management: A Practical Guide to Stock Portfolio Optimization and Asset Allocation*. Boston, MA: Harvard Business School Press, 1998.
[4] W. F. Sharpe, "The Sharpe Ratio," *The Journal of Portfolio Management*, vol. 21, no. 1, pp. 49–58, 1994.
[5] R. T. Rockafellar and S. Uryasev, "Optimization of Conditional Value-at-Risk," *Journal of Risk*, vol. 2, no. 3, pp. 21–41, 2000.
[6] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
[7] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2016, pp. 785–794.
[8] S. Gu, B. Kelly, and D. Xiu, "Empirical Asset Pricing via Machine Learning," *The Review of Financial Studies*, vol. 33, no. 5, pp. 2223–2273, 2020.
[9] B. Kelly, S. Malamud, and K. Zhou, "The Virtue of Complexity in Return Prediction," *The Journal of Finance*, vol. 79, no. 1, pp. 459–503, 2024.
[10] O. Jin and H. El-Saawy, "Portfolio Management using Reinforcement Learning," Stanford University, CS229 Project Report, 2016.
[11] S. Perrin and T. Roncalli, "Machine Learning Optimization Algorithms & Portfolio Allocation," Amundi Quantitative Research, arXiv:1909.10233, 2019.
[12] M. López de Prado, *Advances in Financial Machine Learning*. Hoboken, NJ: John Wiley & Sons, 2018.
[13] D. H. Bailey, J. Borwein, M. López de Prado, and Q. J. Zhu, "The Probability of Backtest Overfitting," *Journal of Computational Finance*, vol. 20, no. 4, pp. 39–69, 2017.
[14] AQR Capital Management, "Can Machines Build Better Stock Portfolios?" Alternative Thinking Series, 2024.

*(Expand to 15–30 sources as the full literature review progresses.)*

---

## 10. GITHUB EXECUTION PRACTICE

- Commit per logical unit of work, not per file save — message style: `feat: add
  walk-forward split logic`, `fix: correct rolling-window lookahead in RSI feature`,
  `test: add no-lookahead unit test suite`.
- One GitHub Issue per Gantt-chart phase (Section 6) — close with a summary comment
  documenting what was found, not just "done."
- Tag milestones: `v0.1-baselines`, `v0.2-ml-models`, `v0.3-monte-carlo`,
  `v0.4-backtest-complete`, `v1.0-final`.
- `CHANGELOG.md` updated weekly in plain English — this becomes the most human-readable
  record of the project's actual evolution for anyone auditing it later.

---

*End of context file. This document should be updated as decisions are finalized during
each phase — treat Section 6's checkpoints as living checklists, not a static plan.*
