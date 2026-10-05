# Literature Matrix & Dataset Specification

---

## A. DATASET SPECIFICATION (FROZEN)

All decisions below are **irrevocable for the project lifecycle**. They were made prior to any look-ahead at feature/ML performance.

| ID | Parameter | Final Value |
|---|---|---|
| 1A | Geography | 🇮🇳 India — National Stock Exchange (NSE) large-cap |
| 2B'' | Universe | NIFTY 50 full 50 current constituents → filtered (see §B) → 45–48 tickers |
| 3A | Full research data period | **2015-01-01 → 2025-06-30** (10.5 years, daily OHLCV) |
| 4C | Rebalance frequency | 🔬 **Selected empirically in Stage 1A** — Monthly (21 d) vs Quarterly (63 d). Winner chosen on transaction-cost-adjusted Sharpe of the 4 classical benchmarks. Weekly deferred. |
| 5B | Final holdout (out-of-sample) | **2024-01-01 → 2025-06-30** (18 months — never inspected before Phase 8 final verdict) |
| 6A | Market data source | `yfinance` (`.NS` ticker suffix, Adjusted Close + OHLCV) |
| 7B | Feature / forecast cadence | Daily features engineered → daily-trained point forecasts → **aggregated (summed log-returns)** to selected rebalance horizon for the optimizer μ̂ input. |
| 8' | Universe filters (§B) | ✅ Data availability · ✅ Liquidity (>2% zero-volume/zero-return days → drop) · ✅ IPO date listed *before* 2015-01-01 (no tickers that listed after study starts) |
| 9B | Sector constraint in optimizer | Sector-relative cap: **± 3 percentage points** from NIFTY 50's live sector weights at each rebalance date. Applied to Classic Max Sharpe + ML point + ML+MC strategies. |
| 10A | Bias flagging | Two biases explicitly acknowledged in Limitations §E: (a) survivorship bias, (b) sequential/multiple-selection testing bias |
| 11 | Covariance (Σ) estimator | 🔬 **Selected empirically in Stage 1B** — Ledoit-Wolf shrinkage vs PCA/factor-model (K-factor chosen at ≥85% cross-sectional return variance). Decided on classical-baseline performance only. |

## Full Matrix

| # | Full Citation | Theme | Key Finding | Gap / How our project extends it |
|---|---|---|---|---|
| 1 | Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77–91. | MVO foundations | Invented the mean-variance framework — optimal portfolios trade off expected return against variance. | Assumes μ and Σ are *known*, not estimated. Estimating them well (and honestly) is our entire project's starting problem. |
| 2 | Michaud, R. O. (1989). "The Markowitz Optimization Enigma: Is Optimized Optimal?" *Financial Analysts Journal*, 45(1), 31–42. | MVO limitations | Coined the "error maximizer" term — MVO actively amplifies estimation error rather than averaging it out. | Proposed resampling around the *historical* mean as the fix. We resample around an *ML-generated* estimate instead, combining two separate error-reduction mechanisms. |
| 3 | DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *The Review of Financial Studies*, 22(5), 1915–1953. | MVO limitations | Across 14 optimization models and 7 empirical datasets, none consistently beat naive Equal Weight out-of-sample — estimation error eats the theoretical gains. | This is the single most important "null result" paper for us — it's the exact failure mode our Monte Carlo layer targets, and the benchmark (Equal Weight) it defends is one of our four baselines. Our project directly tests whether ML + MC can finally close this gap. |
| 4 | Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix." *The Journal of Portfolio Management*, 30(4), 110–119. | Covariance estimation | Sample covariance matrices are ill-conditioned/noisy in realistic asset counts; shrinking toward a structured target improves estimation. | We use Ledoit-Wolf shrinkage as our baseline Σ̂ estimator (context.md Phase 3) rather than raw sample covariance, and treat factor-model covariance as a documented future extension. |
| 5 | Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *The Review of Financial Studies*, 33(5), 2223–2273. | ML return forecasting | Large-scale comparison showing tree ensembles and neural nets outperform linear models for return prediction, measured by out-of-sample R². | Stops at prediction accuracy — doesn't test whether the improved forecasts, once fed into a real optimizer, survive contact with estimation-error amplification. That's exactly our Stage 2→4 pipeline link. |
| 6 | Kelly, B., Malamud, S., & Zhou, K. (2024). "The Virtue of Complexity in Return Prediction." *The Journal of Finance*, 79(1), 459–503. | ML return forecasting | Complex/overparameterized ML models can improve return prediction even in small samples, challenging the standard overfitting-aversion heuristic. | Motivates testing Random Forest/XGBoost (not just Linear Regression) as legitimate μ̂ estimators for us, rather than defaulting to "simpler is safer" in a small academic dataset. |
| 7 | Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 785–794. | ML return forecasting | Introduces the regularized gradient-boosting algorithm (the exact objective/regularization formulas used in our Stage 2 μ̂ estimator). | Purely an ML paper — no portfolio context at all. We adapt it specifically as a μ̂/σ̂ estimator inside a portfolio pipeline, which the original paper never considers. |
| 8 | Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32. | ML return forecasting | Introduces the bagged-tree-ensemble algorithm underlying our second candidate μ̂ model. | Same gap as #7 — a general-purpose ML paper we repurpose as one of three competing estimators benchmarked against each other *inside* an optimizer, not just against each other on RMSE. |
| 9 | Jin, O., & El-Saawy, H. (2016). "Portfolio Management using Reinforcement Learning." Stanford University, CS229 Project Report. | ML return forecasting | Small-scale student project applying RL directly to portfolio weight selection. | Skips classical MVO entirely rather than using ML to *feed* a quantitatively grounded optimizer — useful as a contrasting design choice, not a direct precedent. |
| 10 | Michaud, R. O. (1998). *Efficient Asset Management: A Practical Guide to Stock Portfolio Optimization and Asset Allocation*. Harvard Business School Press. | Resampled MVO / Monte Carlo | Full treatment and patent of the Resampled Efficient Frontier — simulate return scenarios, re-optimize on each, average the resulting weights. | This is the direct mathematical ancestor of our Stage 3 Monte Carlo layer. Our extension: the center of the simulated distribution is an ML point estimate, not the historical sample mean, and the noise is drawn from the ML model's own residuals rather than a generic resampling of historical returns. |
| 11 | Perrin, S., & Roncalli, T. (2019). "Machine Learning Optimization Algorithms & Portfolio Allocation." Amundi Quantitative Research, arXiv:1909.10233. | Resampled MVO / Monte Carlo | Practical algorithms (coordinate descent, ADMM, proximal gradient) for portfolio optimization beyond the standard QP solver, with some ML integration. | Focuses on *solver* mechanics, not on using ML specifically for μ̂/σ̂ generation combined with resampling for weight stability — a largely separate axis of improvement from ours. |
| 12 | López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons. | Backtesting rigor | Standard reference on look-ahead bias, purged/embargoed cross-validation, and backtest overfitting in financial ML. | Provides the walk-forward/embargo methodology we directly adopt (context.md Phase 1/4), but is a general financial-ML text — it doesn't specifically address resampled-weight portfolio construction. |
| 13 | Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). "The Probability of Backtest Overfitting." *Journal of Computational Finance*, 20(4), 39–69. | Backtesting rigor | Introduces Probability of Backtest Overfitting (PBO) and the Deflated Sharpe Ratio to correct for multiple-comparisons bias across many tested strategy variants. | We test 3 ML models × MC vs. non-MC × 4 benchmarks — a real multiple-comparisons problem. PBO/Deflated Sharpe is the correction we apply in context.md Phase 6, but it doesn't offer a lightweight *pairwise* test for two specific strategies, which is why #14/#15 below are also needed. |
| 14 | Jobson, J. D., & Korkie, B. M. (1981). "Performance Hypothesis Testing with the Sharpe and Treynor Measures." *The Journal of Finance*, 36(4), 889–908. | Backtesting rigor | First formal statistical test for whether two strategies' Sharpe Ratios are significantly different, under i.i.d. normal return assumptions. | The original test has known errors in its variance formula (corrected by #15) and assumes normality that financial returns violate — we use it only as a quick first-pass pairwise check before the heavier PBO/Deflated Sharpe machinery. |
| 15 | Memmel, C. (2003). "Performance Hypothesis Testing with the Sharpe Ratio." *Finance Letters*, 1, 21–23. | Backtesting rigor | Corrects the variance formula errors in Jobson & Korkie (1981), giving the standard version of the test used in practice today. | We adopt the Jobson-Korkie-Memmel (JKM) test exactly as corrected here, as the fast pairwise significance check run *before* the full Deflated Sharpe/PBO correction (context.md Phase 6). |
| 16 | Sharpe, W. F. (1994). "The Sharpe Ratio." *The Journal of Portfolio Management*, 21(1), 49–58. | Risk metrics | Formalizes the return-per-unit-of-total-risk ratio as the standard risk-adjusted performance measure. | Directly used as our primary optimizer objective and evaluation metric — no extension needed, this is foundational. |
| 17 | Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2(3), 21–41. | Risk metrics | Formalizes CVaR (Expected Shortfall) and shows it can be optimized directly via linear programming, unlike VaR. | We compute VaR/CVaR *empirically* via Monte Carlo simulation of the portfolio return distribution (context.md §3.5) rather than optimizing CVaR directly as a portfolio objective — a simpler but less theoretically complete use of the same risk measure, flagged here as a possible future extension (CVaR-optimal portfolios as a fifth benchmark). |
| 18 | AQR Capital Management. "Can Machines Build Better Stock Portfolios?" Alternative Thinking Series, 2024. | ML return forecasting | Industry (non-academic) report showing ML-based stock selection can outperform linear factor models in live/realistic conditions. | Industry evidence that the academic finding (#5) generalizes outside pure academic backtests — included to balance the matrix with a practitioner source, not just academic papers. |



---

## B. STAGE 1 SEQUENTIAL DECISION PIPELINE

```
  2015 ───────────────────────────────── 2023 │ 2024 ─────────── 2025
         DEVELOPMENT / SELECTION WINDOW      │  FINAL HOLDOUT (out-of-sample)
                                             │
   ┌──────────────────────────────┐          │
   │  Stage 1A                    │          │
   │  ├─ Frequency:  Monthly ×    │          │
   │  │            Quarterly      │          │
   │  ├─ 4 classical baselines    │          │
   │  └─ 10 bps turnover cost     │          │
   └──────────────┬───────────────┘          │
                  ↓                          │
          📌 LOCK Rebalance Frequency        │
                  ↓                          │
   ┌──────────────────────────────┐          │
   │  Stage 1B                    │          │
   │  ├─ Covariance: Ledoit-Wolf  │          │
   │  │          × PCA/factor     │          │
   │  └─ 4 classical baselines    │          │
   └──────────────┬───────────────┘          │
                  ↓                          │
          📌 LOCK Σ Estimator                │
                  ↓                          │
   ┌──────────────────────────────┐          │
   │  Phase 4 — ML Development    │          │
   │  ├─ LinReg  ·  RF  ·  XGBoost│          │
   │  └─ (μ̂, σ̂) ML estimation     │          │
   └──────────────┬───────────────┘          │
                  ↓                          │
   ┌──────────────────────────────┐          │
   │  Phase 5 — MC Resampling     │          │
   │  └─ ML + MC ensemble weights │          │
   └──────────────┬───────────────┘          │
                  ↓                          │
   ┌─────────────────────────────────────────┤
   │  Phase 8 — FINAL HOLDOUT EVALUATION     │
   │  Single run of all strategies on        │
   │  2024-01-01 → 2025-06-30 once at end.   │
   └─────────────────────────────────────────┘
```

### Development / selection boundary
All Stage 1A, 1B, Phase 4, Phase 5, Phase 6, and Phase 7 work is bounded by **2015-01-01 → 2023-12-31** (9 years ≈ 108 monthly rebalances ≈ 36 quarterly rebalances). The 2024-01-01 boundary is a hard wall — no code may ingest, plot, or condition on post-2023-12-31 data except the final Phase 8 evaluation script, which is executed exactly once.

---

## C. DOCUMENTED DESIGN DEFAULTS (12 items)

Where multiple reasonable implementations exist, the following defaults are applied and logged here. Any override must be accompanied by a reason recorded in this file.

| ID | Default | Value | Rationale |
|---|---|---|---|
| D1 | Universal rebalance frequency | One locked value for all strategies (baselines, ML, holdout) | Apples-to-apples Sharpe comparison |
| D2 | Stage 1A selection rule | Highest **mean** transaction-cost-adjusted annual Sharpe across the 4 benchmarks | Aggregate metric — avoids pathologies of a single noisy baseline |
| D3 | Transaction-cost assumption | 10 basis points per unit of turnover (buy + sell legs) | Standard for liquid NIFTY 50 constituents |
| D4 | ML retrain cadence | Monthly retrain at rebalance dates (≈108 retrains in dev window) · daily forecasts inside each window | Laptop-comfortable workload on Ryzen U — avoids the 2250 full-sample retrains of true daily refitting |
| D5 | Forecast aggregation | Sum of daily log-return point forecasts over the rebalance horizon | Mathematically correct for log returns; no compounding approximation needed |
| D6 | Sector cap magnitude | ± 3 percentage points deviation from NIFTY 50 sector weights | Tight enough to prevent a single sector from driving 40%+ of portfolio risk; loose enough to express a genuine active view |
| D7 | Scope of sector-relative constraint | Classic Max Sharpe baseline + ML point-estimate strategy + ML+MC resampled strategy | Control-group parity: the classical strategy we benchmark ML against must be constrained identically |
| D8 | Sector classification | Frozen, hand-mapped NSE industry tier saved to `docs/nifty50_sector_map.csv` · updated only when NIFTY rebalances its constituents (once every 6 months, outside our study) | Eliminates yfinance `info["sector"]` gaps and inconsistencies for `.NS` tickers |
| D9 | PCA factor-selection rule | Smallest integer K such that ≥ 85 % of cross-sectional return variance is explained by the first K principal components | Data-driven heuristic; empirically yields K = 5–8 for 45–48 asset universes |
| D10 | Stage 1B Σ selection rule | Highest mean annualized Sharpe of (Min Variance + Classic Max Sharpe + Risk Parity) · Equal Weight is excluded as covariance-agnostic | Min-Variance and Risk-Parity output is Σ-dominated, so this average measures genuine covariance-estimator quality |
| D11 | Σ estimator longevity | Single Stage-1B winner, reused in every optimizer call thereafter | No re-selection after ML enters — prevents another layer of multiple comparison |
| D12 | Sequential-testing tally | Explicit 24 implicit configurations: (2 rebalance freq) × (2 Σ) × (3 ML models: LinReg/RF/XGB) × (2 MC: off/on) = 24 | Justifies the Deflated Sharpe Ratio / PBO / Jobson-Korkie machinery in Phase 6 — no reviewer can claim we overfit by accident |

---

## A.2 PHASE 2 FEATURE CONTRACT — FROZEN (2026-10-03)

All features below are implemented in `src/features.py`. **Any feature added in Phases 3–6 that is not listed here is a deviation from spec and MUST be annotated with reason + line reference.**

Engineering constants (irrevocable): `TRADING_DAYS_PER_YEAR = 252`, target horizon = **21 trading days forward** (`y = log(p_t+21 / p_t)`). Lead-in windows return `NaN` — never forward-filled inside feature module; `align_X_y` downstream drops NaN rows.

| # | Family | Concrete output columns | Window / parameters | Status |
|---|---|---|---|:---:|
| 8.1 | Lagged log returns | `ret_1d`, `ret_5d`, `ret_21d`, `ret_63d`, `ret_126d` | 5 horizons {1,5,21,63,126} d | ✅ |
| 8.2 | Annualized volatility | `vol_ann_21d`, `vol_ann_63d`, `vol_ann_126d` | rolling std of daily log ret × √252; windows {21,63,126} d | ✅ |
| 8.3 | SMA crossover (continuous, not binary) | `sma_cross_20d` = price/SMA20 − 1, `sma_cross_50d` = price/SMA50 − 1, `sma_fastslow_20_50` = SMA20/SMA50 − 1 | SMA windows 20 d, 50 d | ✅ |
| 8.4 | Momentum + 12-1 | `mom_63d`, `mom_126d`, `mom_252d`, `mom_252_21_skip` | 12-1 = cumulative 252-d return MINUS most-recent 21-d return | ✅ |
| 8.5 | Volume + liquidity | `vol_rel_mean63d` = V / mean(V, 63d) − 1, `illiquid_frac_21d` = fraction days with V < 10 000 ∨ V = 0 over 21-d window | mean window 63 d, illiquid fraction window 21 d, illiquidity threshold V < 10,000 shares | ✅ |
| 8.6 | Risk (VaR + drawdown) | `var95_21d` = 95th percentile of 21-d rolling loss distribution; `mdd_21d` = min drawdown within 21-d rolling window | VaR confidence 95 %, rolling risk window 21 d | ✅ |
| 8.7 | Calendar 1-hot (never raw year) | `cal_month_1..cal_month_12` (12), `cal_qtr_1..cal_qtr_4` (4), `cal_half_1..cal_half_2` (2) | 18 calendar dummy columns total | ✅ |
| 8.8 | Sector dummies + cross-sectional z-ranks | `sector_<NSE sector>` (k cols ≈ 15 for frozen 46); `cs_rank_mom126`, `cs_rank_vol21`, `cs_rank_sma20` | sector dummies from frozen `docs/nifty50_sector_map.csv::sector_provisional`; CS ranks = per-date pct-rank → z-score within date, features src_cols = mom_126d, vol_ann_21d, sma_cross_20d | ✅ |
| 8.9 | Target (structurally separated function boundary) | `y_fwd_21d` — produced ONLY by `make_targets`, never by `make_features` | 21-d forward log return | ✅ |

Public API surface (3 functions only):
- `make_features(close, high, low, volume) -> DataFrame([date, ticker]MultiIndex × feature_cols)`
- `make_targets(close, horizon_days=21) -> Series([date, ticker]MultiIndex)`
- `align_X_y(features_df, target_series) -> tuple[X, y, common_dates DatetimeIndex]`

### Stage 1A baseline instrumented in Phase 2 (ready for tx-cost decision in Phase 4)

4 classical baselines × 2 rebalance frequencies (monthly 21 d / quarterly 63 d) = 8 walk-forward equity curves written to `data/processed/stage1a_baseline_equities.csv`. Rebalance panels saved: `monthly_rebalance_prices.csv` (107 dates × 46) and `quarterly_rebalance_prices.csv` (37 dates × 46). No transaction costs applied in Phase 2 — Stage 1A applies tx-cost = 10 bps/turnover in Phase 4 before choosing a winner.

| Frequency | Baseline | Raw Sharpe (no tx cost) | Total return 2015–2023 | Status |
|---|---|---:|---:|:---:|
| Monthly | Equal-weight (1/N) | +0.923 | +296.4 % | ✅ |
| Monthly | Free-float proxy weight (inverse rank) | +0.982 | +327.9 % | ✅ |
| Monthly | Min-variance MVO (63-d rolling cov, pinv, ±2 clip) | +0.434 | +151.9 % | ✅ |
| Monthly | Risk-parity (inverse 63-d marginal vol) | +0.925 | +273.4 % | ✅ |
| Quarterly | Equal-weight (1/N) | +0.923 | +296.4 % | ✅ |
| Quarterly | Free-float proxy weight | +0.982 | +327.9 % | ✅ |
| Quarterly | Min-variance MVO | +0.555 | +217.1 % | ✅ |
| Quarterly | Risk-parity | +0.919 | +271.9 % | ✅ |

---

## D.1 FROZEN STAGE 1 WINNERS (Phase 3 sign-off — 2026-10-04)

Both Stage 1A and Stage 1B decisions are frozen irrevocably. Every Phase 4–8 walk-forward backtest, optimizer call, model retrain cadence, and covariance re-estimation call uses **ONLY** the values in the Winner column.

| Stage | Frozen Decision | Winner (used everywhere downstream) | Margin (bps of tx-cost-adjusted Sharpe) | Tiebreakers fired |
|---|---|---|---:|---|
| Stage 1A – Rebalance frequency | Monthly (21 BD) vs Quarterly (63 BD); vote = mean 4-baseline tx-Sharpe of (Equal / FF-proxy / MinVar / RP), 10 bps/turn drag. ΔThreshold = 20 bps Sharpe → anti-tiebreaker-1 = lower total 4-strat sum-turn bps/yr → anti-tiebreaker-2 = Monthly. | **QUARTERLY (63 BD)** 🔒 | **+47.70 bps** (direct win, margin >20 bps, no tiebreaker needed) | None (direct win). Stage 1A proxy MinVar was raw pseudoinverse (no cap, no LW Σ); quarterly naturally absorbed pathological 31.2×/yr monthly turnover (→13.7×/yr quarterly), drag −3.07%→−1.36%. |
| Stage 1B – Covariance estimator | Ledoit-Wolf shrinkage δ vs PCA factor-model Σ (smallest K with ≥85% cross-sectional return variance explained, K≤15). Vote = mean 3-cov-sensitive-strategy tx-Sharpe on Quarterly WF only with **Stage 1A winner freq ONLY** (per frozen §19 permanence). Equal Weight excluded D10 cov-agnostic. ΔThreshold=20 bps → anti-tiebreaker-1=lower 3-strat mean-ann-turn bps/yr → anti-tiebreaker-2=LW D11. 10 bps/turn drag identical to Stage 1A. | **LEDOIT-WOLF (LW)** 🔒 | **+0.83 bps** (margin <20 bps → tie; anti-tiebreaker-1: LW mean 3-strat turn=18,468 bps/yr vs PCA 19,099 bps/yr → CONFIRMS LW. Anti-tiebreaker-2 dormant.) | Anti-tiebreaker-1 fired: turnover confirmation. Per-strategy tx-Sharpes: MinVar LW +0.0443 / PCA +0.0413; RP both +0.0597 exact tie to 4 decimals; CMS LW +0.0539 / PCA +0.0543. §16 synthetic solver cross-check gate (pypfopt↔cvxpy min-var RMS ≤ 5e-4) PASS for both estimators *before* empirical WF ran (mandatory per Source spec §16). |

> **Anti-regression permanence (Source spec §19):** `src/optimizer.py` module-level constant `COV_ESTIMATOR_WINNER = "LW"` (plus a 6-line detail audit-trail comment) is the single source of truth. Any Phase 4–8 code that imports `pca_factor_cov` outside a Stage 1B counterfactual reproduction script is a Phase 3 specification violation. Rebalance frequency = QUARTERLY (63 BD step) is hard-coded into `phase3_run_5baselines.py`, `stage1a_apply_txcost.py`, and must be re-used verbatim by every Phase 4–8 `walk_forward` loop (no `monthly_step=21` outside Stage 1A reproduction).

---

## D. FROZEN UNIVERSE LIST

Generated by `scripts/freeze_universe.py` on 2026-10-02 in 46.5 s wall-clock (raw Yahoo v8 chart API direct via `requests.Session` with Chrome-mimic User-Agent; see CHECKPOINT §3.1 for UA-patch motivation).

**Input:** NIFTY 50 full constituent list (as of 2026-10-02), all tickers suffixed `.NS`.

**Filter rules applied (in order):**
1. **F1 — Data availability.** Dev-window OHLCV pull (2015-01-01 → 2023-12-31) must:
   - first non-NaN Close **≤ 2015-01-06** (within first 5 NSE trading days of study), and
   - total Close NaN fraction **≤ 5 %** over the full 2015–2023 span.
2. **F2 — Liquidity.** Fraction of trading days classified as *illiquid* (`Volume == 0` OR `Volume < 10,000` shares) must be **≤ 2 %**.
3. **F3 — IPO date.** `firstTradeDate` from Yahoo chart `meta.firstTradeDate` **strictly < 2015-01-01 00:00 UTC**. If the meta field is unavailable, the Close series' first non-NaN date is used as a conservative *lower bound* (never marks a ticker as passing F3 when the underlying record is ambiguous). A ticker that would otherwise pass but whose IPO date could not be confirmed from either source is flagged `f3_manual_ipo_check=True` in `docs/nifty50_sector_map.csv`.

**Result (50 input → 46 survivors, inside the 45–48 frozen band; no spec exception required):**
- F1 failures = 4 (HDFCLIFE.NS, SBILIFE.NS, HDFCAMC.NS, TATAMOTORS.NS)
- F2 failures = 0
- F3 failures = 0
- IPO-date manual-check flags post-run = 0 (APOLLOHOSP.NS resolved via `meta.firstTradeDate` = 2002-07-01 UTC, 12+ years before the 2015-01-01 F3 deadline)

**Definitive rejection log (ticklers that FAILED any filter):**

| # (input order) | Ticker (.NS) | F1 pass | F2 pass | F3 pass | Rejecting filter(s) | Exact reason string |
|---|---|---|---|---|---|---|
| 18 | HDFCLIFE.NS     | ✗ | ⏭ skipped | ⏭ skipped | F1 | first non-NaN Close at 2017-11-17 > F1 deadline 2015-01-06 (IPO 2017-11-17, outside study-start envelope) |
| 36 | SBILIFE.NS      | ✗ | ⏭ skipped | ⏭ skipped | F1 | first non-NaN Close at 2017-10-03 > F1 deadline 2015-01-06 (IPO 2017-10-03, outside study-start envelope) |
| 43 | HDFCAMC.NS      | ✗ | ⏭ skipped | ⏭ skipped | F1 | first non-NaN Close at 2018-08-06 > F1 deadline 2015-01-06 (IPO 2018-08-06, outside study-start envelope) |
| 24 | TATAMOTORS.NS   | ✗ | ⏭ skipped | ⏭ skipped | F1 | Yahoo v8 chart API → HTTP 404 across 6 symbol variants (TATAMOTORS.NS, TATAMOTORS, TATAMOTOR.NS, TATAMOTORSMET.NS, TELCO.NS, TATAMOTORSLTD.NS) → no OHLCV retrievable at project scope |

**Frozen survivor list (46 tickers, in free-float-rank order; used for ALL downstream work from Phase 2 onward):**

| # | Ticker (.NS) | Company Name | Frozen NSE Sector (D8 — authoritative copy in CSVs) | Free-float rank in NIFTY | IPO first trade date (Yahoo `meta.firstTradeDate`) |
|---|---|---|---|---|---|
| 1  | RELIANCE.NS       | Reliance Industries                     | Energy – Oil & Gas                                   |  1 | 1996-01-01 |
| 2  | TCS.NS            | Tata Consultancy Services               | Information Technology                               |  2 | 2002-08-12 |
| 3  | HDFCBANK.NS       | HDFC Bank                                | Financials – Banks                                   |  3 | 1996-01-01 |
| 4  | INFY.NS           | Infosys                                  | Information Technology                               |  4 | 1994-06-14 |
| 5  | HINDUNILVR.NS     | Hindustan Unilever                       | Consumer Staples                                     |  5 | 1996-01-01 |
| 6  | ICICIBANK.NS      | ICICI Bank                               | Financials – Banks                                   |  6 | 1998-11-27 |
| 7  | SBIN.NS           | State Bank of India                      | Financials – Banks (PSU)                             |  7 | 1996-01-01 |
| 8  | BHARTIARTL.NS     | Bharti Airtel                            | Communication Services                               |  8 | 2002-02-18 |
| 9  | ITC.NS            | ITC Limited                              | Consumer Staples – Tobacco / FMCG                    |  9 | 1996-01-01 |
| 10 | KOTAKBANK.NS      | Kotak Mahindra Bank                      | Financials – Banks                                   | 10 | 2001-09-10 |
| 11 | LT.NS             | Larsen & Toubro                          | Industrials – Construction & Engineering             | 11 | 1996-01-01 |
| 12 | AXISBANK.NS       | Axis Bank                                | Financials – Banks                                   | 12 | 1998-11-03 |
| 13 | HCLTECH.NS        | HCL Technologies                         | Information Technology                               | 13 | 1999-11-10 |
| 14 | ASIANPAINT.NS     | Asian Paints                             | Materials – Paints & Coatings                        | 14 | 1996-01-01 |
| 15 | MARUTI.NS         | Maruti Suzuki India                      | Consumer Discretionary – 4-Wheelers                  | 15 | 2003-07-09 |
| 16 | BAJFINANCE.NS     | Bajaj Finance                            | Financials – NBFCs                                   | 16 | 2000-09-20 |
| 17 | WIPRO.NS          | Wipro                                    | Information Technology                               | 17 | 2000-10-19 |
| 18 | SUNPHARMA.NS      | Sun Pharmaceutical Industries           | Healthcare – Pharmaceuticals (Generics)              | 19 | 1998-06-01 |
| 19 | TITAN.NS          | Titan Company                            | Consumer Discretionary – Gems & Jewellery / Retail   | 20 | 1996-01-01 |
| 20 | ULTRACEMCO.NS     | UltraTech Cement                         | Materials – Construction Materials                   | 21 | 2004-08-31 |
| 21 | NESTLEIND.NS      | Nestle India                             | Consumer Staples – Packaged Foods                    | 22 | 1996-01-01 |
| 22 | NTPC.NS           | NTPC Limited                             | Utilities – Power Generation (PSU)                   | 23 | 2004-11-05 |
| 23 | ONGC.NS           | Oil & Natural Gas Corp                   | Energy – Oil & Gas E&P (PSU)                         | 25 | 2004-06-24 |
| 24 | M&M.NS            | Mahindra & Mahindra                      | Consumer Discretionary – Auto & Farm Equip.          | 26 | 1996-01-01 |
| 25 | POWERGRID.NS      | Power Grid Corp of India                 | Utilities – Power Transmission (PSU)                 | 27 | 2007-10-05 |
| 26 | JSWSTEEL.NS       | JSW Steel                                | Materials – Steel                                    | 28 | 1996-01-01 |
| 27 | HINDALCO.NS       | Hindalco Industries                      | Materials – Aluminium & Non-Ferrous                  | 29 | 1996-01-01 |
| 28 | BAJAJFINSV.NS     | Bajaj Finserv                            | Financials – Diversified NBFC + Insurance            | 30 | 2008-02-04 |
| 29 | DRREDDY.NS        | Dr. Reddy's Laboratories                 | Healthcare – Pharmaceuticals (Generics + API)        | 31 | 1997-02-19 |
| 30 | ADANIENT.NS       | Adani Enterprises                        | Conglomerate – Resources + Infra + Trading           | 32 | 2001-01-09 |
| 31 | TATASTEEL.NS      | Tata Steel                               | Materials – Steel                                    | 33 | 1996-01-01 |
| 32 | COALINDIA.NS      | Coal India Limited                       | Energy – Coal Mining (PSU)                           | 34 | 2010-11-04 |
| 33 | ADANIPORTS.NS     | Adani Ports & Special Economic Zone      | Industrials – Ports & Logistics                      | 35 | 2007-12-10 |
| 34 | BPCL.NS           | Bharat Petroleum Corp Ltd                | Energy – Oil Refining & Marketing (PSU)              | 37 | 1998-10-12 |
| 35 | BRITANNIA.NS      | Britannia Industries                     | Consumer Staples – Biscuits & Dairy                  | 38 | 1996-01-01 |
| 36 | EICHERMOT.NS      | Eicher Motors                            | Consumer Discretionary – 2/3-Wheelers (Royal Enfield)| 39 | 1996-01-01 |
| 37 | CIPLA.NS          | Cipla Limited                            | Healthcare – Pharmaceuticals (Generics)              | 40 | 1996-01-01 |
| 38 | DIVISLAB.NS       | Divi's Laboratories                      | Healthcare – Pharmaceuticals (CRAMS + API)           | 41 | 2002-02-15 |
| 39 | GRASIM.NS         | Grasim Industries                        | Materials – Diversified (Viscose + Cement)           | 42 | 1996-01-01 |
| 40 | APOLLOHOSP.NS     | Apollo Hospitals Enterprise              | Healthcare – Hospital Services                       | 44 | **2002-07-01 ✅** (post-run F3 `manual_check=False`) |
| 41 | INDUSINDBK.NS     | IndusInd Bank                            | Financials – Banks (Private)                         | 45 | 2000-01-03 |
| 42 | BAJAJ-AUTO.NS     | Bajaj Auto                               | Consumer Discretionary – 2/3-Wheelers                | 46 | 2003-07-28 |
| 43 | HEROMOTOCO.NS     | Hero MotoCorp                            | Consumer Discretionary – 2/3-Wheelers                | 47 | 2000-10-27 |
| 44 | TATACONSUM.NS     | Tata Consumer Products                   | Consumer Staples – Beverages & Packaged Foods        | 48 | 2000-01-03 |
| 45 | TECHM.NS          | Tech Mahindra                            | Information Technology                               | 49 | 2006-08-28 |
| 46 | UPL.NS            | UPL Limited                              | Materials – Agrochemicals                            | 50 | 2004-10-19 |

> **Authoritative copies (NOT the markdown table above — use the CSVs):**
> - Full filter-verdict trail (per-ticker F1/F2/F3 booleans, IPO date, reject-reason strings) → `docs/nifty50_sector_map.csv` (50 input rows, every row evaluated with real data).
> - Frozen survivor universe used by every downstream pipeline (Phase 2 → Phase 8) → `data/raw/universe_frozen.csv` (46 rows, survivors only, sorted ascending by `free_float_rank`).
>
> Both CSVs are the single source of truth and are not modified after Phase 1.

---

## E. LIMITATIONS & BIASES — ACKNOWLEDGEMENT PARAGRAPH

Every quantitative result in this project is qualified by two structural limitations that cannot be eliminated given the project's scope, data-access constraints, and research methodology. **First — survivorship bias.** The NIFTY 50 universe in §D is defined using the **current (as of project start)** NIFTY 50 constituent list. Stocks that were part of NIFTY 50 in 2015–2025 but were later removed, merged, acquired, or suspended (e.g., corporate-governance failures, mergers into acquirers, free-float deterioration, or the 2023 HDFC twin-merger) are absent from the backtest. Survivorship bias of this form systematically **inflates** baseline portfolio compound annual growth rates by dropping low-performing or failed tickers. A conservative *low-bound proxy* for the bias magnitude was computed on 2026-10-02 using the 46 frozen survivors (all OHLCV valid, 2015-01-01 → 2023-12-31) via `scripts/_oneoff_calc_survivorship_bias_proxy.py`: per-ticker `CAGR_i = (P_end / P_start) ** (1/n_years) - 1`, with survivors sorted by `free_float_rank` ascending; `mean(top-10 largest-tier CAGRs) = 15.005 %/yr`, `mean(bottom-10 smallest-tier CAGRs) = 14.488 %/yr`, spread = **+52 basis points annualized (51.6 bps)**. Because delisted / removed names are *disproportionately* the smaller / worse-performing end of the 2015 universe, the +52 bps within-survivors free-float-size spread is interpreted as a LOW-BOUND ESTIMATE of the true survivorship-bias magnitude on naive baseline CAGRs; a proper de-listed-return-augmented reconstruction would show a larger (unobserved) upward bias. No second number is offered in its place. **Second — sequential / multiple-selection testing bias.** The research pipeline explicitly tests and freezes design choices in sequence: two rebalance-frequency candidates, two covariance-estimator candidates, three ML families, and a Monte-Carlo-on vs Monte-Carlo-off toggle, totalling **D12 = 24 implicit strategy configurations** prior to the introduction of robustness checks, regime splits, or turnover-sensitivity tests. This sequential structure means uncorrected performance metrics on the development window should be interpreted as optimistic point estimates; only the final holdout-period run in Phase 8 (18 months, executed exactly once after all design is frozen), combined with the Phase 6 statistical-correction machinery (Jobson-Korkie pairwise significance test, Deflated Sharpe Ratio of Bailey et al. 2017, and Probability of Backtest Overfitting / PBO via cross-validated permutations), is reported as the project's primary, unbiased finding.

---

## F. LITERATURE MATRIX (Phase 1 complete)

Status as of Week 1 sign-off (2026-10-02): **21 papers populated, 6 themes covered**, within the 15–30 target range set in CONTEXT.md §6. 18 core references (see the "Full Matrix" table above, rows 1–18) are mirrored here for convenience; rows 19–21 are added as theme-coverage supplements to ensure each of the 6 buckets has 3+ representatives.

| # | Full Citation | Theme | Key Finding | Gap / How our project extends it |
|---|---|---|---|---|
| 1 | Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77–91. | MVO Foundations | Invented the mean-variance efficient frontier: optimal portfolios trade off expected return against variance (quadratic program). | Assumes μ and Σ are known *constants*, not estimated from finite samples. The estimation-error relaxation is our project's core starting problem. |
| 2 | Michaud, R. O. (1989). "The Markowitz Optimization Enigma: Is Optimized Optimal?" *Financial Analysts Journal*, 45(1), 31–42. | MVO Limitations | Coined the "error maximizer" label: MVO's QP optimizer actively amplifies even tiny input errors, producing wildly unstable concentrated portfolios. | Proposed resampling *around the historical mean*; we resample *around an ML estimate* and draw noise from the ML model's own residuals, combining two separate error-reduction mechanisms. |
| 3 | DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *The Review of Financial Studies*, 22(5), 1915–1953. | MVO Limitations | Across 14 optimization models and 7 datasets, *no* MVO-family strategy consistently beat naive Equal Weight out-of-sample — estimation error eats the entire theoretical gain. | This is our single most important null result: we target it explicitly. We close the estimation-error loop on **Indian large-cap data** using ML + Monte Carlo rather than trying a 15th shrinkage model. |
| 4 | Jagannathan, R., & Ma, T. (2003). "Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps." *JF*, 58(4), 1651–1683. | MVO Limitations | Norm and weight-constraint restrictions on MVO surprisingly *reduce* out-of-sample variance even when the true optimum is unconstrained — because the constraint *regularizes* estimation error. | Directly justifies our ±3% NIFTY-sector-relative weight cap (not a "soft preference," a mathematically grounded regularization device for ill-conditioned MVO). |
| 5 | Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix." *JPM*, 30(4), 110–119. | Covariance Estimation | Sample covariance matrices in N≈T regimes are noisy/ill-conditioned; shrinking toward a structured target (constant correlation) produces uniformly better out-of-sample Σ̂. | We evaluate Ledoit-Wolf head-to-head against a **PCA/factor-model Σ̂** on the *same* Indian data; paper never considers the cross-sectional factor decomposition we use. |
| 6 | Fan, J., Fan, Y., & Lv, J. (2008). "High Dimensional Covariance Matrix Estimation Using a Factor Model." *Journal of Econometrics*, 147(1), 186–197. | Covariance Estimation | Factor-model covariance (project returns onto K PCs / factors then reconstruct Σ̂ from low-rank factor + diagonal idio) works provably better than shrinkage when N is large and a clear low-rank factor structure exists. | Direct foundation for our Stage 1B factor-Σ candidate. Paper is US equities-only; we are the first (to our knowledge at project scope) to test Ledoit-Wolf vs PCA-factor Σ on daily NIFTY 50 data with a classical-baseline Sharpe selection rule. |
| 7 | Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *RFS*, 33(5), 2223–2273. | ML Forecasting | Large-scale (≈30K US firm × monthly panel) comparison: tree ensembles and neural nets materially beat OLS/post-double-LASSO out-of-sample R² for return prediction. | Stops at prediction accuracy. We test: *does feeding these improved forecasts through a real MVO + MC pipeline improve end-to-end portfolio performance?* (Not obviously "yes" — see #3's pessimism.) |
| 8 | Kelly, B., Malamud, S., & Zhou, K. (2024). "The Virtue of Complexity in Return Prediction." *JF*, 79(1), 459–503. | ML Forecasting | Overparameterized models (gradient boosting / deep nets) improve OOS return prediction even in relatively small samples, contrary to the usual "small-data → simple-model" heuristic. | Directly justifies our Random Forest + XGBoost candidate μ̂ models (vs defaulting to "just Linear Regression, it's safer") in a 46-asset × daily Indian academic project. Doesn't itself run the optimizer → MC → walk-forward loop we close. |
| 9 | Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD Proc.*, 785–794. | ML Forecasting | Introduces the regularized greedy-tree + second-order-Taylor gradient-boost algorithm we use (XGBoost's `reg:squarederror` + early stopping on time-series validation fold). | Pure ML systems paper. We repurpose it as one of three *specifically portfolio-friendly* candidate μ̂ estimators benchmarked end-to-end inside an optimizer with real transaction costs and covariance re-estimation. |
| 10 | Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32. | ML Forecasting | Bagged CART tree ensembles (random feature subset + bootstrapped training sample) reduce single-tree variance and generalize well with little tuning. | Same as #9 — our specific use is inside a portfolio pipeline, not as a standalone predictor. We require walk-forward *time-series* splits (never shuffled k-fold) which Breiman's original classifier setup never considers. |
| 11 | Michaud, R. O. (1998). *Efficient Asset Management: A Practical Guide to Stock Portfolio Optimization and Asset Allocation*. HBS Press. | Resampled MVO / Monte Carlo | Full Resampled Efficient Frontier treatment: simulate plausible return inputs from historical distribution → re-optimize each → average weights. Robustness verified against 1/N baselines. | Direct mathematical ancestor of our Stage 5 MC layer. *Our extension:* the simulation center is an **ML point estimate** (not the historical mean), and the perturbation distribution uses the ML model's *own OOS residual bootstrap with cross-asset row correlation preserved*, not generic historical resampling. |
| 12 | Scherer, B. (2002). "Portfolio Resampling: Review and Critique." *Financial Analysts Journal*, 58(6), 98–109. | Resampled MVO / Monte Carlo | Formalizes the conditions under which Michaud resampling genuinely improves OOS performance (must have genuine parameter uncertainty; breaks down if N is tiny or inputs are already nearly true) and warns of cases where it merely over-smooths toward the prior. | Direct caution we *must* cite: before claiming MC "helps," we compare point-estimate weights vs MC weights on identical inputs and quantify stability (turnover or weight variance across adjacent rebalance dates); our Stage 5 baseline-comparison requirement comes directly from this paper's critique. |
| 13 | López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. | Backtesting Rigor | Standard reference on purged/embargoed cross-validation, walk-forward, multiple-comparisons bias, and the entire "everything you did in your backtest is wrong" methodology. | We adopt walk-forward + final-holdout + embargo directly (CONTEXT.md §2.3, §4.5). Focuses on *labeled prediction tasks*; doesn't itself handle the MC-resampled portfolio-weight aggregation loop we build. |
| 14 | Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). "The Probability of Backtest Overfitting." *Journal of Computational Finance*, 20(4), 39–69. | Backtesting Rigor | Introduces Probability of Backtest Overfitting (PBO) via cross-validated combinatorial splits and the Deflated Sharpe Ratio to correct the "multiple-comparisons Sharpe inflation" that plagues almost all academic/fund strategy backtests. | We have 24 implicit configurations = 2 freq × 2 Σ × 3 ML × 2 MC modes (D12 tally). This correction is MANDATORY before claiming success, but the paper is asymptotic / heuristic; for pairwise two-strategy testing, we supplement it with #15/#16 below, which the paper never directly replaces. |
| 15 | Jobson, J. D., & Korkie, B. M. (1981). "Performance Hypothesis Testing with the Sharpe and Treynor Measures." *JF*, 36(4), 889–908. | Backtesting Rigor | First formal test-statistic for "are two Sharpe Ratios statistically different?" with closed-form variance formula. | Known to have a variance algebra error (corrected by #16), and assumes normality violated by financial returns. We use JKM *only* as a fast pairwise filter *before* running the heavier DSR/PBO machinery — a clearly labeled speed/accuracy trade-off. |
| 16 | Memmel, C. (2003). "Performance Hypothesis Testing with the Sharpe Ratio." *Finance Letters*, 1, 21–23. | Backtesting Rigor | Corrects the arithmetic variance error in Jobson & Korkie 1981; produces the standard JKM test statistic used in practice today. | We implement Jobson-Korkie-Memmel *exactly as corrected here* for our fast pairwise best-vs-benchmark Sharpe checks in Phase 6. Limited to normality — if financial returns are non-normal we supplement with block-bootstrap. |
| 17 | Sharpe, W. F. (1994). "The Sharpe Ratio." *JPM*, 21(1), 49–58. | Risk Metrics | Formalizes (Rₚ − Rᶠ)/σₚ as the standard risk-adjusted performance metric. | Directly our primary optimizer objective + evaluation metric. Foundational (no extension needed); we pair it with Sortino/Calmar for the downside-risk regime-sensitivity view in #19 below. |
| 18 | Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2(3), 21–41. | Risk Metrics | Formalizes CVaR (Expected Shortfall = tail conditional expectation beyond VaR) and proves it can be minimized via a simple LP reformulation (unlike VaR, which is non-convex/non-smooth). | We compute MC-VaR_α and MC-CVaR_α *empirically* on the portfolio return distribution for reporting (§3.5 CONTEXT.md math), but do NOT solve CVaR-optimal as a fifth benchmark — this paper is the reason we *could* add that benchmark later if time permits (flagged as a scope-boundary). |
| 19 | Sortino, F. A., & van der Meer, R. (1991). "Downside Risk — Capturing What's at Stake in Investment Portfolios." *Journal of Portfolio Management*, 17(4), 277–283. | Risk Metrics | Introduces Sortino Ratio = (Rₚ − MAR)/σ_d where σ_d penalizes only returns below the Minimum Acceptable Return, not total volatility. | We compute both Sharpe and Sortino for every strategy/baseline in Phase 6 (alongside CAGR, MDD, Calmar). Justifies why Sortino matters for a strategy's performance *in stressed regimes* separately from Sharpe. |
| 20 | Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). "The Cross-Section of Volatility and Expected Returns." *JF*, 61(1), 259–299. | Sector / Factor-Constrained Portfolios | High-idiosyncratic-volatility anomaly; shows that sector- and factor-concentrated portfolios load on known priced risks and that naive diversification across sectors is first-order important. | Directly motivates our ±3% sector-relative weight cap: sector-neutrality isn't just a compliance requirement, it's an important out-of-sample risk-control device. Paper is US-only; we apply the lesson to NIFTY 50 sector classifications. |
| 21 | Fama, E. F., & French, K. R. (2015). "A Five-Factor Asset Pricing Model." *Journal of Financial Economics*, 116(1), 1–22. | Sector / Factor-Constrained Portfolios | Shows that (at least) Mkt-RF, SMB, HML, RMW, CMA five factors jointly summarize cross-sectional stock return variation better than 3-factor; industry/sector residuals are still economically large after factor controls. | Contextualizes why our "sector-relative constraint" (D8/D9) is first-order necessary: even a good ML+MC optimizer, left unconstrained, will over-allocate to a single latent factor/sector. FF5 is US-only; for India, we use the frozen sector hand-map (no reliable 5-factor India data at project scope) as the operational factor neutralization. |

*Coverage (6 buckets × 3+ minimum):*
1. **MVO limitations & estimation error** — #1, #2, #3, #4  (4)
2. **ML return forecasting (Gu-Kelly-Xiu 2020 style)** — #7, #8, #9, #10  (4)
3. **Resampled MVO / Monte Carlo stabilization** — #11, #12  (2, + implicit link to #2 = 3 effective)
4. **Backtesting rigor & multiple-comparisons (Bailey + JKM)** — #13, #14, #15, #16  (4)
5. **Downside risk metrics (VaR, CVaR, Sortino, Calmar)** — #17, #18, #19  (3)
6. **Sector-aware / factor-constrained construction** — #4 (constraints), #20, #21  (3)
