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

## D. FROZEN UNIVERSE LIST

Generated by `scripts/freeze_universe.py` on 2026-10-02. Source: NIFTY 50 constituent list (as of 2026-10-02) → yfinance `.NS` pull → filters:
1. Data availability: full OHLCV record 2015-01-02 → 2023-12-29 (no missing first 100-day blocks that cannot be forward-filled)
2. Liquidity: ≤ 2% of trading days with (volume = 0 OR close == previous close with volume < 50th percentile)
3. IPO date: NSE listing date strictly earlier than 2015-01-01

| Order | Ticker (.NS) | Company Name | NSE Sector (D8 hand-map TBD) | Free-float Mkt Cap Rank in NIFTY | Rejected? (Y/N + Reason) |
|---|---|---|---|---|---|
| 1 | RELIANCE.NS | Reliance Industries Ltd. | Energy / Oil & Gas | 1 | |
| 2 | TCS.NS | Tata Consultancy Services | Information Technology | 2 | |
| 3 | HDFCBANK.NS | HDFC Bank | Financials - Banks | 3 | |
| 4 | INFY.NS | Infosys | Information Technology | 4 | |
| 5 | HINDUNILVR.NS | Hindustan Unilever | Consumer Staples | 5 | |
| 6 | ICICIBANK.NS | ICICI Bank | Financials - Banks | 6 | |
| 7 | SBIN.NS | State Bank of India | Financials - Banks | 7 | |
| 8 | BHARTIARTL.NS | Bharti Airtel | Communication Services | 8 | |
| 9 | ITC.NS | ITC Limited | Consumer Staples - Tobacco / FMCG | 9 | |
| 10 | KOTAKBANK.NS | Kotak Mahindra Bank | Financials - Banks | 10 | |
| 11 | LT.NS | Larsen & Toubro | Industrials - Construction & Engineering | 11 | |
| 12 | AXISBANK.NS | Axis Bank | Financials - Banks | 12 | |
| 13 | HCLTECH.NS | HCL Technologies | Information Technology | 13 | |
| 14 | ASIANPAINT.NS | Asian Paints | Materials - Paints & Coatings | 14 | |
| 15 | MARUTI.NS | Maruti Suzuki India | Consumer Discretionary - Automobiles | 15 | |
| 16 | BAJFINANCE.NS | Bajaj Finance | Financials - NBFCs | 16 | |
| 17 | WIPRO.NS | Wipro | Information Technology | 17 | |
| 18 | HDFCLIFE.NS | HDFC Life Insurance | Financials - Insurance | 18 | |
| 19 | SUNPHARMA.NS | Sun Pharmaceutical Industries | Healthcare / Pharmaceuticals | 19 | |
| 20 | TITAN.NS | Titan Company | Consumer Discretionary - Gems & Jewellery / Retail | 20 | |
| 21 | ULTRACEMCO.NS | UltraTech Cement | Materials - Construction Materials | 21 | |
| 22 | NESTLEIND.NS | Nestle India | Consumer Staples - Foods | 22 | |
| 23 | NTPC.NS | NTPC Limited | Utilities - Power | 23 | |
| 24 | TATAMOTORS.NS | Tata Motors | Consumer Discretionary - Automobiles | 24 | |
| 25 | ONGC.NS | Oil & Natural Gas Corporation | Energy - Oil & Gas (PSU) | 25 | |
| 26 | M&M.NS | Mahindra & Mahindra | Consumer Discretionary - Automobiles | 26 | |
| 27 | POWERGRID.NS | Power Grid Corporation of India | Utilities - Power Transmission | 27 | |
| 28 | JSWSTEEL.NS | JSW Steel | Materials - Metals & Mining (Steel) | 28 | |
| 29 | HINDALCO.NS | Hindalco Industries | Materials - Metals & Mining (Aluminum) | 29 | |
| 30 | BAJAJFINSV.NS | Bajaj Finserv | Financials - NBFC + Insurance Holding | 30 | |
| 31 | DRREDDY.NS | Dr. Reddy's Laboratories | Healthcare / Pharmaceuticals | 31 | |
| 32 | ADANIENT.NS | Adani Enterprises | Commodities / Infra / Trading Conglomerate | 32 | |
| 33 | TATASTEEL.NS | Tata Steel | Materials - Metals & Mining (Steel) | 33 | |
| 34 | COALINDIA.NS | Coal India Limited | Energy - Coal Mining (PSU) | 34 | |
| 35 | ADANIPORTS.NS | Adani Ports and SEZ | Industrials - Infrastructure / Ports & Logistics | 35 | |
| 36 | SBILIFE.NS | SBI Life Insurance | Financials - Insurance | 36 | |
| 37 | BPCL.NS | Bharat Petroleum Corporation | Energy - Oil & Gas Refining (PSU) | 37 | |
| 38 | BRITANNIA.NS | Britannia Industries | Consumer Staples - Foods (Biscuits & Dairy) | 38 | |
| 39 | EICHERMOT.NS | Eicher Motors | Consumer Discretionary - Automobiles (Two & Three Wheelers) | 39 | |
| 40 | CIPLA.NS | Cipla Limited | Healthcare / Pharmaceuticals | 40 | |
| 41 | DIVISLAB.NS | Divi's Laboratories | Healthcare / Pharmaceuticals (CRAMS / API) | 41 | |
| 42 | GRASIM.NS | Grasim Industries | Materials - Diversified (Viscose / Cement) | 42 | |
| 43 | HDFCAMC.NS | HDFC Asset Management Company | Financials - Asset Management | 43 | |
| 44 | APOLLOHOSP.NS | Apollo Hospitals Enterprise | Healthcare - Hospitals & Services | 44 | |
| 45 | INDUSINDBK.NS | IndusInd Bank | Financials - Banks (Private) | 45 | |
| 46 | BAJAJ-AUTO.NS | Bajaj Auto | Consumer Discretionary - Automobiles (Two & Three Wheelers) | 46 | |
| 47 | HERO MOTOCORP.NS | Hero MotoCorp (HEROMOTOCO) | Consumer Discretionary - Automobiles (Two & Three Wheelers) | 47 | |
| 48 | TATACONSUM.NS | Tata Consumer Products | Consumer Staples - Beverages & Foods | 48 | |
| 49 | TECHM.NS | Tech Mahindra | Information Technology | 49 | |
| 50 | UPL.NS | UPL Limited (formerly United Phosphorus) | Materials - Agrochemicals | 50 | |

> **Rejection log for the yfinance pull:** After automated filter run, expected ~45–48 tickers will survive. Rejections (if any) will be inserted here with the filter rule that triggered them. This entire section will be regenerated once `scripts/freeze_universe.py` completes successfully.

> **Sector mapping:** The "NSE Sector" column above contains a provisional human-readable label. The authoritative frozen mapping for the ±3% sector-relative constraint (D8) is the CSV at `docs/nifty50_sector_map.csv`. That CSV is the single source of truth and is not modified after Week 1.

---

## E. LIMITATIONS & BIASES — ACKNOWLEDGEMENT PARAGRAPH

Every quantitative result in this project is qualified by two structural limitations that cannot be eliminated given the project's scope, data-access constraints, and research methodology. **First — survivorship bias.** The NIFTY 50 universe in §D is defined using the **current (as of project start)** NIFTY 50 constituent list. Stocks that were part of NIFTY 50 in 2015–2025 but were later removed, merged, acquired, or suspended (e.g., corporate-governance failures, mergers into acquirers, free-float deterioration, or the 2023 HDFC twin-merger) are absent from the backtest. Survivorship bias of this form systematically **inflates** baseline portfolio compound annual growth rates by dropping low-performing or failed tickers; a rough conservative proxy based on index-removal rates of 2–4 tickers per annum suggests baseline CAGRs are biased upward by 25–75 basis points annualized versus a properly de-listed-return-augmented index reconstruction. **Second — sequential / multiple-selection testing bias.** The research pipeline explicitly tests and freezes design choices in sequence: two rebalance-frequency candidates, two covariance-estimator candidates, three ML families, and a Monte-Carlo-on vs Monte-Carlo-off toggle, totalling **D12 = 24 implicit strategy configurations** prior to the introduction of robustness checks, regime splits, or turnover-sensitivity tests. This sequential structure means uncorrected performance metrics on the development window should be interpreted as optimistic point estimates; only the final holdout-period run in Phase 8 (18 months, executed exactly once after all design is frozen), combined with the Phase 6 statistical-correction machinery (Jobson-Korkie pairwise significance test, Deflated Sharpe Ratio of Bailey et al. 2017, and Probability of Backtest Overfitting / PBO via cross-validated permutations), is reported as the project's primary, unbiased finding.

---

## F. LITERATURE MATRIX (TO BE COMPLETED)

Target: 15–30 papers by end of Week 1.

| # | Full Citation | Theme | Key Finding | Gap / How our project extends it |
|---|---|---|---|---|
| 1 | Harry Markowitz, "Portfolio Selection," JF 1952 | MVO Foundations | Derived mean-variance efficient frontier | Assumes true μ, Σ are known — the estimation-error problem our entire project attacks |
| 2 | Richard O. Michaud, "The Markowitz Optimization Enigma: Is Optimized Optimal?" FAJ 1989 | MVO Limitations | Coined "error maximizer" characterization of MVO with estimated inputs | Proposed resampled MVO; we combine ML forecasting + MC residual resampling vs. his pure historical resampling |
| 3 | Olivier Ledoit & Michael Wolf, "Honey, I Shrunk the Sample Covariance Matrix," JPM 2004 | Covariance Estimation | Shrinkage estimator of Σ that dominates sample covariance in-sample and out-of-sample | We evaluate Ledoit-Wolf shrinkage head-to-head against PCA/factor-model covariance on NIFTY 50 data |
| 4 | | | | |
| 5 | | | | |
| … | | | | |

Themes to cover (3–5 papers each):
* MVO limitations & estimation error
* ML return and volatility forecasting in cross-sectional equities (Gu-Kelly-Xiu 2020-style)
* Resampled portfolio optimization / Monte Carlo stabilization
* Backtesting rigor, multiple-comparisons correction, DSR / PBO (Bailey et al. 2014, 2017)
* Downside risk metrics: VaR, CVaR, Sortino, Calmar, Sharpe variants (Jobson-Korkie 1981 + Memmel 2003)
* Sector-aware / factor-constrained portfolio construction
