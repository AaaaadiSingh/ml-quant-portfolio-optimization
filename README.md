# ML-Based Portfolio Optimization for Risk-Aware Investment Decision Making
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adi-quant-portfolio.streamlit.app)

[![Status: v1.0-final](https://img.shields.io/badge/Status-v1.0--final-brightgreen.svg)](README.md)
[![Tests](https://img.shields.io/badge/Tests-43%2F43%20Passing-success.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.12.3-blue.svg)](requirements.txt)
[![Phase](https://img.shields.io/badge/Pipeline-Phases%201--8%20Signed%20Off-darkgreen.svg)](checkpoint.md)

An institutional-grade quantitative finance research codebase investigating whether machine-learning return estimates combined with Monte Carlo resampling and uncertainty modelling can improve portfolio robustness and stability under strict walk-forward out-of-sample conditions.

---

## Executive Summary

When constructing investment portfolios with classical Markowitz Mean-Variance Optimization (MVO), sample estimates of expected returns ($\hat{\mu}$) and covariance ($\hat{\Sigma}$) contain substantial estimation error. As established by Michaud (1989), mean-variance optimizers act as "error maximizers", heavily overweighting securities with upward-biased return estimates and underweighting those with downward-biased estimates.

This project implements a multi-layer quantitative framework to address this challenge:
1. **Machine Learning Return Engine:** Pooled walk-forward predictive models (Ridge Regression, Random Forest, XGBoost) using 8 feature families (77 engineered predictors) to estimate conditional expected returns $\hat{\mu}_{t+1}$.
2. **Monte Carlo Resampling (Michaud Resampling):** Resampling returns via empirical multivariate-row block bootstrap preserving fat-tailed residual properties ($\nu \approx 4.30$) and cross-asset correlation.
3. **Walk-Forward Convex Optimization:** Solving continuous convex risk-budgeted and turnover-constrained optimization problems with CVXPY across $K=500$ resampled scenarios per rebalance date, computing the centroid portfolio $\bar{\mathbf{w}}$.
4. **Statistical Rigor & Anti-P-Hacking:** Strict look-ahead bias elimination, transaction cost modeling (10 bps/turn), Jobson-Korkie pairwise hypothesis testing (Memmel 2003 asymptotic correction), Deflated Sharpe Ratio (Bailey & López de Prado 2014, $N=24$), Combinatorially Symmetric Cross-Validation Probability of Backtest Overfitting (PBO), stationary block bootstrap ($L=19$, $B=200$ synthetic paths), and a strictly embargoed out-of-sample holdout test (2024-01-01 to 2025-06-30).

---

## Headline Results

### Walk-Forward Development Window (2015-01-01 → 2023-12-29, 2,222 Trading Days)

The consolidated performance across all 8 walk-forward strategies evaluated on the development dataset is summarized below (derived from `results/metrics_summary.csv`):

| Strategy | Ann Return (%) | Ann Vol (%) | Sharpe (Tx-Adj) | Sortino | Max DD (%) | Calmar | Ann Turnover (bps) | DSR Prob | P(Strat > EW) | PBO |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MC Centroid (Mean, LW $\Sigma$, $\pm$3pp)** | **23.75%** | **17.34%** | **1.1152** | **1.6177** | **-36.70%** | **0.6472** | **6,148.9** | **0.9967** | **64.00%** | **0.300** |
| Random Forest Point Est. | 24.01% | 19.32% | 1.0307 | 1.4776 | -40.10% | 0.5989 | 13,787.0 | 0.9939 | 85.00% | 0.300 |
| Equal Weight (1/N) | 20.13% | 16.61% | 0.9739 | 1.3678 | -36.48% | 0.5516 | 1,810.8 | 0.9899 | Baseline | 0.300 |
| Risk Parity (LW $\Sigma$) | 18.90% | 15.48% | 0.9655 | 1.3574 | -34.06% | 0.5548 | 5,816.2 | 0.9891 | 63.50% | 0.300 |
| Minimum Variance (LW $\Sigma$) | 17.27% | 14.28% | 0.9349 | 1.3258 | -29.16% | 0.5921 | 19,993.6 | 0.9868 | 65.00% | 0.300 |
| XGBoost Point Est. | 21.11% | 18.72% | 0.9286 | 1.2974 | -38.39% | 0.5499 | 12,799.3 | 0.9867 | 82.50% | 0.300 |
| Classic Max Sharpe (LW $\Sigma$, $\pm$3pp) | 19.07% | 17.64% | 0.8762 | 1.2023 | -35.60% | 0.5358 | 24,362.0 | 0.9805 | 68.50% | 0.300 |
| Ridge LinReg Point Est. | 20.35% | 19.27% | 0.8742 | 1.1974 | -45.73% | 0.4449 | 16,758.5 | 0.9798 | 44.00% | 0.300 |

### Key Research Findings

1. **Transaction-Adjusted Outperformance:** The Monte Carlo Centroid strategy achieved the highest transaction-cost-adjusted Sharpe ratio of **1.1152**, outperforming the 1/N Equal Weight benchmark (0.9739) and the classical un-resampled Maximum Sharpe baseline (0.8762).
2. **Dramatic Turnover Reduction (~63%):** Resampling and averaging optimal portfolios in weight space reduced annual portfolio turnover from 16,758.5 bps/year (Ridge point-estimate) and 24,362.0 bps/year (Classic Max Sharpe) down to **6,148.9 bps/year**—a ~63% to ~75% reduction in portfolio churn.
3. **Statistical Significance & DSR:** The Centroid strategy's Deflated Sharpe Ratio probability (Bailey & López de Prado 2014) is **0.9967** ($p = 0.0033$) under $N=24$ trials, rejecting the hypothesis of data-mining luck. The Jobson-Korkie pairwise test vs 1/N yielded $p = 0.0037$.
4. **PBO & Overfitting Control:** Combinatorially Symmetric Cross-Validation (CSCV) with $S=6$ non-overlapping time slices resulted in a Probability of Backtest Overfitting (PBO) of **0.300**, well below the rejection threshold of 0.500.
5. **Synthetic Bootstrap Robustness:** Across $B=200$ synthetic market paths generated by stationary block bootstrap ($L=19$ days), the Centroid portfolio outperformed Equal Weight in **64.00%** of resampled universes, with a 5th-percentile worst-case Sharpe of **+0.4119**.

---

### Strictly Embargoed Holdout Window (2024-01-01 → 2025-06-30, 368 Trading Days)

Per project protocol, the 18-month holdout window was accessed exactly once on the frozen pipeline without post-hoc re-tuning. The strategies were held at their final frozen weights (rebalance 2023-11-07):

| Strategy | CAGR (%) | Ann Vol (%) | Sharpe (Tx-Adj) | Max DD (%) | Sortino |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 14.85% | 15.06% | 0.7516 | -14.07% | 1.0532 |
| Equal Weight (1/N) | 13.40% | 14.91% | 0.6713 | -17.21% | 0.9125 |
| Risk Parity | 12.66% | 13.97% | 0.6589 | -17.35% | 0.9115 |
| **MC Centroid** | **12.67%** | **14.95%** | **0.6257** | **-17.15%** | **0.8386** |
| Minimum Variance | 10.86% | 12.13% | 0.6029 | -17.45% | 0.8675 |
| Classic Max Sharpe | 13.04% | 16.76% | 0.5952 | -21.42% | 0.7670 |
| XGBoost | 12.58% | 16.13% | 0.5865 | -17.35% | 0.7559 |
| Ridge LinReg | 12.20% | 16.81% | 0.5487 | -16.32% | 0.7020 |

**Honest Out-of-Sample Verdict:**
In the 2024–2025 Indian equities rally, broad market momentum favored equal weighting (Sharpe 0.6713) and non-linear tree models (Random Forest Sharpe 0.7516). The MC Centroid delivered consistent risk controls (Sharpe 0.6257, Max Drawdown -17.15%), comfortably beating Classic Max Sharpe (0.5952) and Ridge regression (0.5487), while confirming that un-rebalanced fixed-weight allocations in strong bull markets experience compression against cap-weighted and equal-weighted momentum.

---

## Research Figures

All figures are automatically generated by `scripts/phase8_generate_figures.py` and `scripts/phase8_holdout_eval.py` into `results/figures/`:

- **Figure 1 — Equity Curves (Dev Window 2015–2023):** Cumulative total return of all 8 walk-forward strategies (`results/figures/fig1_equity_curves.png`).
- **Figure 2 — Rolling Drawdowns:** Drawdown overlay highlighting the COVID crash of Feb–Mar 2020 (`results/figures/fig2_drawdown_overlay.png`).
- **Figure 3 — Sharpe Ratio Ranking & DSR:** Transaction-adjusted Sharpe comparison with deflated significance annotations (`results/figures/fig3_performance_bar.png`).
- **Figure 4 — Jobson-Korkie Heatmap:** Pairwise test $p$-values with Memmel (2003) correction (`results/figures/fig4_jk_heatmap.png`).
- **Figure 5 — Bootstrap Sharpe Distributions:** Boxplots of $B=200$ synthetic block-bootstrap paths (`results/figures/fig5_bootstrap_boxplot.png`).
- **Figure 6 — Macro Regime Performance:** Performance matrix across 6 distinct economic regimes (`results/figures/fig6_regime_heatmap.png`).
- **Figure 7 — Transaction Cost Sensitivity:** Performance decay curves across 0–50 bps cost levels (`results/figures/fig7_txcost_sensitivity.png`).
- **Figure 8 — 4-Factor Risk Attribution:** Fama-French style Carhart 4-factor regression exposures (`results/figures/fig8_factor_attribution.png`).
- **Figure 9 — Holdout Period Performance:** Strictly out-of-sample cumulative equity trajectory 2024–2025 (`results/figures/fig9_holdout_equity.png`).

---

## Pipeline Architecture

```text
┌────────────────────────────────────────────────────────┐
│            Universe & Historical Prices                │
│       46 Nifty 50 Survivors • 2015-01-01 to 2023-12-29 │
└───────────────────────────┬────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│            Feature Engineering (77 Features)           │
│   Momentum • Volatility • Technical • Liquidity • Cross│
│   Target: 63-day forward return • Anti-lookahead guard │
└───────────────────────────┬────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│             Pooled Walk-Forward ML Engine              │
│  Ridge Regression • Random Forest • XGBoost Regressors │
│  Retrained quarterly • Strictly out-of-sample mu_hat   │
└───────────────────────────┬────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│          Monte Carlo Uncertainty Resampling            │
│  Empirical fat-tailed residuals (nu=4.30) bootstrap    │
│  K = 500 resampled mu vectors per rebalance date       │
│  Convex optimization (CVXPY) with Ledoit-Wolf Sigma    │
│  Sector drift bounds (+-3pp), 10% name cap             │
└───────────────────────────┬────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│            Centroid Portfolio Aggregation              │
│  Weight centroid w_bar = (1/K) sum(w_k)                │
│  Turnover reduced by ~63% • Convexity preserved        │
└───────────────────────────┬────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│             Walk-Forward Backtesting Engine            │
│  Daily drifting portfolio • 10 bps transaction cost    │
│  Jobson-Korkie • Deflated Sharpe (N=24) • CSCV PBO     │
│  Stationary block bootstrap (B=200, L=19)              │
└───────────────────────────┬────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│         Strict Holdout Evaluation (2024-2025)          │
│  Single irreversible run on frozen pipeline weights    │
└────────────────────────────────────────────────────────┘
```

---

## Methodological Details & Formulae

### Deflated Sharpe Ratio (Bailey & López de Prado, 2014)

To correct for data snooping across $N=24$ implicit parameter configurations, the Deflated Sharpe Ratio computes:

$$\text{DSR} = \Phi\left( \frac{(\widehat{SR} - SR^*) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{SR}^2}} \right)$$

where $SR^*$ is the expected maximum Sharpe ratio under the null hypothesis of no skill:

$$SR^* \approx \sqrt{2 \ln N} \left(1 - \frac{\gamma}{\sqrt{2 \ln N}}\right) + \frac{\ln(4\pi \ln N)}{2\sqrt{2 \ln N}}$$

For the Centroid portfolio, $\text{DSR} = 0.9967$ ($p = 0.0033$).

### Probability of Backtest Overfitting (PBO)

Using Combinatorially Symmetric Cross-Validation (CSCV) with $S=6$ time slices generating $\binom{6}{3} = 20$ training/testing combinatorial splits, PBO measures the probability that the in-sample optimal strategy underperforms the median out-of-sample:

$$\text{PBO} = \frac{1}{\binom{S}{S/2}} \sum_{c=1}^{\binom{S}{S/2}} \mathbb{I}\left( \text{Rank}_{\text{OOS}}(c) < \frac{N+1}{2} \right) = 0.300 < 0.500$$

---

## 🖥️ Interactive Web Dashboard (Recruiter-Ready)

An institutional-grade interactive Streamlit web dashboard is included to explore the research findings without requiring manual command execution or terminal scripting.

### Dashboard Architecture & Pages
- **📊 Page A: Executive Overview:** High-level scorecard, research problem formulation, and honest out-of-sample holdout summary.
- **📈 Page B: Strategy Performance:** Interactive ₹1,00,000 capital growth simulator, rolling drawdowns overlay with COVID crash highlights, and sortable risk-adjusted performance table.
- **⚖️ Page C: Development vs. Holdout:** Transparent side-by-side comparison between the 9-year walk-forward development window and the 18-month embargoed holdout period (2024–2025), explaining why momentum outpaced centroid risk controls during the bull market rally.
- **🛡️ Page D: Risk & Robustness:** Stationary block bootstrap distributions ($B=200$ synthetic paths), Deflated Sharpe Ratio (DSR), CSCV Probability of Backtest Overfitting (PBO), 6 macro-regime stress tests, and transaction cost sensitivity (0–50 bps).
- **💼 Page E: Portfolio Weights Explorer:** Interactive quarterly rebalance date selector (37 dates), stock weights table with sector breakdown, whole-share capital sizing estimator, and convexity constraint auditor (sum=100%, 10% name cap).
- **📖 Page F: Methodology & Limitations:** Comprehensive mathematical equations, Carhart 4-factor models, survivorship bias quantification (+51.6 bps proxy), and reproducibility guide.

### 🧭 Suggested Recruiter Walkthrough (5-Minute Tour)

For quant recruiters, portfolio managers, and hiring teams reviewing this repository:

1. **Executive Overview (Page A):**
   - **What to look for:** Immediate empirical scorecard and problem formulation.
   - **Key takeaway:** Monte Carlo weight resampling reduced portfolio turnover by **~63%** (6,148.9 bps vs 16,758.5 bps for Ridge ML) while generating the top transaction-adjusted Sharpe of **1.1152** ($p = 0.0033$, $\text{DSR} = 0.9967$).
2. **Strategy Performance (Page B):**
   - **What to look for:** Interactive cumulative growth of ₹1,00,000 across 2,222 trading days.
   - **Key takeaway:** Toggle between linear and log scales, inspect rolling drawdowns across the COVID-19 crash, and review the sortable risk-adjusted performance table.
3. **Development vs. Holdout (Page C):**
   - **What to look for:** Honest out-of-sample attribution rather than selective reporting.
   - **Key takeaway:** Understand why broad momentum favored Random Forest (14.85% CAGR) and Equal Weight (13.40% CAGR) during the 2024–2025 Indian market bull rally, while the Centroid portfolio preserved superior downside risk controls (-17.15% Max DD vs -21.42% Classic Max Sharpe).
4. **Risk & Robustness (Page D):**
   - **What to look for:** $B=200$ synthetic market paths generated by stationary block bootstrap ($L=19$ days).
   - **Key takeaway:** The Centroid achieved a **64.00% win rate** over 1/N Equal Weight across synthetic histories, with a strictly positive 5th-percentile worst-case Sharpe of **+0.4119**. Includes Combinatorially Symmetric Cross-Validation ($\text{PBO} = 0.300 < 0.500$) and 6 macro-regime stress matrices.
5. **Portfolio Weights Explorer (Page E):**
   - **What to look for:** Sector allocations, convexity constraint auditor, and capital sizing engine.
   - **Key takeaway:** Select any of the 37 quarterly rebalance boundaries (2015–2023) to inspect stock weights enriched with company names and sectors, audit 100% convexity compliance (sum=100%, 0 negative weights, 10% cap), and export allocations as CSV.
6. **Methodology & Limitations (Page F):**
   - **What to look for:** Full mathematical rigor, Carhart 4-factor formulations, and limitations disclosure.
   - **Key takeaway:** Quantified survivorship bias (+51.6 bps proxy), multiple testing corrections ($N=24$ implicit configs), and exact reproducibility commands.

### Local Execution (Windows PowerShell)

To launch the dashboard locally in your existing environment:

```powershell
# Activate project virtual environment
.venv\Scripts\Activate.ps1

# Launch the Streamlit dashboard
streamlit run app.py
```
The application will immediately open in your browser at `http://localhost:8501`.

### Public Deployment Guide (Streamlit Community Cloud)

To deploy the dashboard for public web access:

1. **Repository & Branch:** Select your GitHub repository `ml-quant-portfolio-optimization` and branch `main`.
2. **Main File Path:** Set to `app.py`.
3. **App Settings (Advanced):** Under Python dependencies, specify `requirements-dashboard.txt`.
4. **Required Committed Artifacts:** The dashboard loads precomputed research outputs and does NOT retrain models on startup. Ensure the following directories remain committed:
   - `results/metrics_summary.csv` and `results/holdout_eval.csv`
   - `results/figures/*.png`
   - `data/processed/*.csv` (precomputed weights and bootstrap distributions)
   - `docs/nifty50_sector_map.csv`
5. **Security & Secrets:** No API keys, credentials, or secrets are required to run the dashboard.
6. **Regulatory Disclaimer:** *This project is strictly for quantitative research and educational evaluation. It does not constitute investment advice or live trading signals.*

---

## Reproducibility Guide

The entire research pipeline is 100% deterministic and runnable via PowerShell on Windows:

```powershell
# 1. Environment & Dependencies Setup
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run Test Suite & Sanity Checks
pytest tests/ -v
python scripts/sanity_check_features.py

# 3. Execute Pipeline Phases (1 to 7)
python scripts/_p4_exit_gates.py
python scripts/_p5_exit_gates.py
python scripts/_p6_exit_gates.py
python scripts/_p7_exit_gates.py

# 4. Generate Consolidated Phase 8 Deliverables
python scripts/phase8_compile_metrics.py
python scripts/phase8_generate_figures.py
python scripts/phase8_holdout_eval.py

# 5. Run Final Phase 8 Exit Gates (All 7 Gates)
python scripts/_p8_exit_gates.py

# 6. Run the Interactive Web Dashboard
streamlit run app.py
```

---

## Academic Limitations

In accordance with institutional guidelines and `docs/literature_matrix.md §E`:

1. **Survivorship Bias:** The asset universe uses current Nifty 50 constituents back-cast to 2015. A real survivorship bias proxy of **+51.6 bps annualized CAGR spread** was quantified between top-10 and bottom-10 market-cap survivors.
2. **Multiple Hypothesis Testing:** While the Deflated Sharpe Ratio ($N=24$) and CSCV PBO ($0.300$) mitigate false discoveries, sequential model exploration inherently inflates test statistics.
3. **Single-Country Universe:** Findings are specific to Indian equities (NSE) characterized by high economic growth and strong retail/institutional domestic inflows over 2015–2025.
4. **Transaction Cost Simplification:** A flat 10 bps per one-way turnover model was used. Real institutional execution involves non-linear market impact, bid-ask spread widening during crises, and securities transaction taxes (STT).

---

## Repository Structure

```text
ml-quant-portfolio-optimization/
├── .venv/                         # Pinned Python 3.12.3 virtual environment
├── app.py                         # Recruiter-ready Streamlit dashboard entry point
├── dashboard/                     # Modular dashboard components & pages
│   ├── data_loader.py             # Cached data loader and weight validators
│   ├── styles.py                  # Financial design system & Plotly layouts
│   └── pages/                     # 6 dedicated dashboard page modules
│       ├── overview.py            # Page A: Executive Overview
│       ├── performance.py         # Page B: Strategy Performance & ₹1L growth
│       ├── holdout.py             # Page C: Development vs Holdout evaluation
│       ├── robustness.py          # Page D: Risk, Bootstrap & Robustness
│       ├── weights.py             # Page E: Portfolio Weights Explorer
│       └── methodology.py         # Page F: Methodology & Limitations
├── data/
│   ├── raw/
│   │   ├── universe_frozen.csv    # 46 surviving NSE tickers and metadata
│   │   └── nifty50_sector_map.csv # Sector mapping and filter trails
│   └── processed/                 # 64 cached intermediate research CSVs
├── docs/
│   ├── synopsis.pdf               # Academic project synopsis
│   └── literature_matrix.md       # Literature review (21 papers) & design defaults
├── notebooks/                     # Analytical and visualization notebooks
│   ├── 01_feature_engineering.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_monte_carlo_resampling.ipynb
│   ├── 04_backtesting_results.ipynb
│   └── 05_robustness_stress_testing.ipynb
├── results/
│   ├── metrics_summary.csv        # Authoritative 8-strategy performance table
│   ├── holdout_eval.csv           # 18-month embargoed holdout test results
│   └── figures/                   # 9 high-resolution research figures (fig1-fig9)
├── scripts/
│   ├── _p4_exit_gates.py          # Phase 4 automated verification
│   ├── _p5_exit_gates.py          # Phase 5 automated verification
│   ├── _p6_exit_gates.py          # Phase 6 automated verification
│   ├── _p7_exit_gates.py          # Phase 7 automated verification
│   ├── _p8_exit_gates.py          # Phase 8 final exit verification
│   ├── phase8_compile_metrics.py  # Consolidated metric compiler
│   ├── phase8_generate_figures.py # Publication figure generator
│   └── phase8_holdout_eval.py     # Embargoed holdout evaluation script
├── src/                           # Core research modules
│   ├── backtest_engine.py         # Daily walk-forward portfolio simulator
│   ├── data_loader.py             # Data ingestion and anti-leak guards
│   ├── metrics.py                 # Financial performance & risk calculations
│   ├── ml_models.py               # Pooled walk-forward ML models
│   ├── monte_carlo.py             # Empirical resampling & centroid optimizer
│   └── portfolio_optimizer.py     # CVXPY convex optimization layer
├── tests/                         # Pytest automated test suite (54 unit tests)
│   ├── test_dashboard_loader.py   # Dashboard loader & weight validator unit tests
│   └── test_*.py                  # Core pipeline unit tests
├── checkpoint.md                  # Comprehensive session handoff & state tracking
├── CONTEXT.md                     # Research formulation & 16-week project plan
├── CHANGELOG.md                   # Systematic changelog across all phases
├── requirements.txt               # Pinned Python package dependencies
└── requirements-dashboard.txt     # Dashboard standalone deployment dependencies
```

---

## Authors & Supervision

**Adi Aditya Singh**  
Roll No: 12419051723, IIOT-B2  
University School of Automation and Robotics (USAR), GGSIPU  

**Faculty Supervisor:** Dr. Ruchika Sehgal  
Department of Automation and Robotics, GGSIPU  
  
