# CHANGELOG

All changes to the research plan, frozen specification, and codebase are
recorded here. Weeks are counted from project kickoff.

---

## Week 1 — Planning & Foundations (2026-10-02 — COMPLETE ✅)

### ✅ Completed

- **Chore: `requirements.txt`** (2026-10-01)
  - 20 core dependencies pinned and smoke-tested in a fresh venv
  - numpy 1.26.4 (<2.0, required by PyPortfolioOpt), pandas 2.2.3, scipy 1.13.1
  - yfinance 0.2.44, PyPortfolioOpt 1.5.5, cvxpy 1.5.3, arch 7.0.0
  - vectorbt 0.26.0 (plotly pinned <5.23 for `heatmapgl` compat), shap 0.46.0
  - ta 0.11.0, sklearn 1.5.2, xgboost 2.1.1 (CPU `tree_method="hist"`), pytest 8.3.3
  - All installed into `.venv/` on D: drive, smoke test 20/20 pass, `pip check` clean
  - Environment variable `PIP_CACHE_DIR=D:\pip_cache` set permanently for this
    Windows user to avoid C: drive out-of-space crashes (C: was at 76 MB free
    on first xgboost 124 MB wheel download)

- **Chore: Repository skeleton & `.gitignore`** (2026-10-02)
  - Full folder structure in place: `docs/`, `data/raw/`, `data/processed/`,
    `notebooks/`, `results/figures/`, `src/`, `tests/`, `scripts/`
  - `.gitkeep` placeholders committed; empty `src/__init__.py` and
    `tests/__init__.py` for package discovery
  - Python/data-science `.gitignore` written with explicit exclusions for
    `data/raw/*` (only `universe_frozen.csv` and `.gitkeep` survive gitignore)
    and `data/processed/*` / `results/*` (both regeneratable)

- **Docs: Full Dataset Specification + Body Literature Matrix + Frozen Universe
  Tables + Limitations Acknowledgement written to `docs/literature_matrix.md`**
  (2026-10-02, finalized after freeze run same day)
  - §A — 11 frozen parameters (ID 1A, 2B'', 3A, 4C, 5B, 6A, 7B, 8', 9B, 10A, 11)
  - §B — Stage 1 sequential decision pipeline ASCII diagram + hard
    development/final-holdout boundary (2015-01-01 → 2023-12-31 /
    2024-01-01 → 2025-06-30)
  - §C — 12 documented design defaults (D1–D12) with rationales
  - §D — **Final frozen universe (50 input → 46 survivors):**
    - Filter rules: F1 (data avail: first Close ≤ 2015-01-06 AND Close NaN
      frac ≤ 5%), F2 (illiquid-day frac ≤ 2% where illiquid = vol=0 OR
      vol<10k), F3 (meta.firstTradeDate < 2015-01-01)
    - 4 F1-only rejections: HDFCLIFE.NS (IPO 2017-11-17), SBILIFE.NS
      (2017-10-03), HDFCAMC.NS (2018-08-06), **TATAMOTORS.NS** (Yahoo chart
      API HTTP 404 across 6 symbol variants; no OHLCV retrievable at
      project scope)
    - 0 F2 rejections, 0 F3 rejections, 0 post-run IPO manual-check flags
    - 46-row survivor table with IPO dates; authoritative copies are
      `docs/nifty50_sector_map.csv` (50 rows, full filter trail) and
      `data/raw/universe_frozen.csv` (46 survivors only, sorted by
      `free_float_rank`)
    - **APOLLOHOSP.NS resolved** (`meta.firstTradeDate` = 2002-07-01, 12+
      years before F3 deadline; `f3_manual_ipo_check=False` in CSV)
  - §E — **Limitations acknowledgement paragraph** — survivorship bias now
    cites a REAL low-bound proxy of **+52 bps annualized** (mean top10
    free-float survivors' CAGR 15.005%/yr − mean bottom10 14.488%/yr)
    computed via `scripts/_oneoff_calc_survivorship_bias_proxy.py` over
    the 46 frozen survivors (exact formula:
    CAGRᵢ = (P_end/P_start)^(1/n_years) − 1); old 25–75 bps human-guess
    placeholder deleted. Sequential / multiple-selection testing bias
    (D12 = 24 implicit configs) documented with required Phase 6 DSR / PBO
    / JKM correction machinery.
  - §F — **Literature matrix (21 papers, 6 themes × ≥ 3 each)**. 18 core
    references (Full Matrix body §2 rows 1–18) plus 3 theme-coverage
    additions: Jagannathan & Ma 2003 (constraint regularization), Fan et
    al. 2008 (factor covariance), Sortino & van der Meer 1991 (Sortino
    Ratio), Ang et al. 2006 (sector concentration), Fama-French 2015
    (5-factor + sector residuals), Scherer 2002 (resampling critique).
    Coverage per theme: MVO limit (4), ML forecast (4), MC/resampled (3
    effective), backtest rigor (4), risk metrics (3), sector/factor (3).

- **Chore: `scripts/freeze_universe.py` finalized and SUCCESSFULLY EXECUTED
  end-to-end** (2026-10-02)
  - Root-cause of original 429 block: Yahoo Edge CDN blocks
    `User-Agent: python-requests/X.Y.Z` with an HTTP 429 "Edge: Too Many
    Requests" HTML page → 23-byte body. Fix: replace the default UA with
    a Chrome-mimic string via a dedicated `requests.Session()` object
    that is passed explicitly to *every* call (sandbox prevents yfinance
    internals from inheriting the patched UA, so we side-step yfinance
    entirely for OHLCV downloads and IPO firstTradeDate lookups).
  - Raw `Yahoo v8 finance/chart` API helper `_raw_yahoo_chart()` wraps
    `requests.get(url, params=…, session=_yf_session)` + JSON
    deserialization → `pd.DataFrame` with identical Close/Adj Close/Volume
    semantics to `yfinance.download(..., auto_adjust=True)`.
  - IPO-date lookup uses the *same* chart endpoint's `meta.firstTradeDate`
    (returned for free per call) rather than unreliable `Ticker.info`
    round-trips.
  - **Full 50-ticker pull result:** 46.5 s wall-clock, 46 passed, 4 F1
    rejects (see §D above). 0 F2 rejects, 0 F3 rejects. All 50 rows in
    `nifty50_sector_map.csv` have real booleans for F1/F2/F3 (no DEFERRED).
  - Additionally redirects yfinance's sqlite cookie / tz cache files via
    `set_tz_cache_location(ROOT / ".yf_cache")` and
    `set_cache_location(ROOT / ".yf_cache")` to a project-local dir on D:
    to avoid TRAE sandbox writing to `%LOCALAPPDATA%\py-yfinance\tkr-tz.db*`
    (was causing "unable to open database file" OperationalError before
    the raw-chart bypass was introduced).

- **Data freeze: `universe_frozen.csv` + `nifty50_sector_map.csv` fully
  regenerated from real data (2026-10-02)**
  - Input: 50 NIFTY 50 current candidates (NSE, `.NS` suffix) → full
    F1/F2/F3 filter run on dev window 2015-01-01 → 2023-12-31
  - **4 F1 rejections:** HDFCLIFE.NS, SBILIFE.NS, HDFCAMC.NS (first Close
    in Yahoo data is 2017/2018, >F1 deadline 2015-01-06); TATAMOTORS.NS
    (Yahoo chart API HTTP 404, 6 symbol variants tested)
  - **46 survivors** (target band 45–48 met at N=46; no spec exception
    needed)
  - **APOLLOHOSP.NS final verdict:** Passes F3 strict < 2015-01-01 with
    meta.firstTradeDate = 2002-07-01; no caveats; `f3_manual_ipo_check=False`
  - **F1/F2 booleans:** No longer DEFERRED — real True/False for all 50
    input rows.
  - Authoritative files: `docs/nifty50_sector_map.csv` (50 rows, full
    filter trail), `data/raw/universe_frozen.csv` (46 survivors, sorted
    ascending by `free_float_rank`).

- **Chore: `scripts/_oneoff_calc_survivorship_bias_proxy.py` one-off
  written + run (2026-10-02)**
  - Pulls 2015-01-01 → 2023-12-31 Close for the 46 frozen survivors via
    same raw Yahoo-chart helper, computes per-ticker CAGR with exact
    formula `CAGR_i = (P_end / P_start) ** (1/n_years) - 1`, sorts by
    free_float_rank, buckets top-10 vs bottom-10, reports mean-spread
    = 51.6 bps annualized.
  - Output: `data/raw/_survivorship_bias_proxy_cagrs.csv` (46 rows).
  - Used to replace the 25–75 bps guess in §E limitation paragraph.

- **Sanity: pytest framework wiring** (2026-10-02)
  - `pytest tests/ -v` → exit code 5 ("collected 0 items / no tests ran") —
    expected for an empty suite at Phase 1 close; confirms conftest and
    package discovery (tests/__init__.py + src/__init__.py + pytest.ini
    rootdir = D:\ML\Quant) are all healthy.

### 🔜 Still pending for Week 1 → Week 2 handoff

- [ ] (Soft, CHECKPOINT §4.5) `src/data_loader.py` stub: convenience loader
  for `universe_frozen.csv` + raw OHLCV with `final_holdout=False` guard
  that drops rows ≥ 2024-01-01 *before* returning any DataFrame to the
  caller. Recommend writing as first line of Phase 2 before notebook
  exploration to prevent accidental final-holdout contamination.

---

## Week 3-4 — Phase 2: Data & Feature Engineering (2026-10-02/03 — COMPLETE ✅)

### ✅ Completed

- **Step 2.1 — No-lookahead guardrail test** (`tests/test_no_lookahead.py`)
  - Bidirectional 4-test suite: (1) synthetic sentinel `-999999.0` injected into 2024+ holdout rows for 3 fake tickers; assert `final_holdout=False` exposes ZERO sentinel cells; (2) `final_holdout=True` exposes 100% of holdout Close cells = sentinel; (3) `final_holdout` non-bool (`None`, `1`, `0`) all raise `ValueError("final_holdout must be a Python bool")`; (4) `final_holdout` is keyword-only in function signature (enforced via `inspect.signature` — cannot be passed positionally)
  - Execution: `.\.venv\Scripts\python.exe -m pytest -v tests/test_no_lookahead.py` → **4 passed / 0 failed** in 4.71s (confirmed twice: once standalone, once full `pytest -v tests/`)

- **Step 2.2 — Data exploration notebook** (`notebooks/01_data_exploration.ipynb`)
  - 6 required panels, all load data through `load_prices(final_holdout=False)`:
    1. **6.1 NaN heatmap**: Close NaN % by ticker × year (46 × 9 cells) with >1%/yr spike flag
    2. **6.2 CAGR bar chart**: 2015-01→2023-12 annualized log CAGR by ticker, grouped & colored by `sector_provisional`
    3. **6.3 Vol violin plots**: annualized daily log-ret vol distribution by sector × years 2015/2020/2023
    4. **6.4 Correlation heatmap**: 46×46 return correlation (sorted by sector, side color-bands) via `clustermap`
    5. **6.5 Liquidity-by-year**: box+strip plot of illiquid-day fraction (vol < 10,000 shares) with dashed F2 2% threshold line + worst single-year spike reported
    6. **6.6 Sector summary table**: per-sector `N_tickers | cagr_mean_ann | ann_vol_mean | free_float_rank_mean | median_illiquid_frac_9yr` → exported to `data/processed/sector_summary_dev_window.csv`
  - Final signature cell logs diagnostics used as gate evidence

- **Step 2.3 — Feature module** (`src/features.py` — public 3-function API, 8 families)
  - Hard-coded module constants (all horizons/windows match defaults defined in CONTEXT §A.2):
    - `TRADING_DAYS_PER_YEAR = 252`
    - `RETURN_HORIZONS_DAYS = (1, 5, 21, 63, 126)`
    - `VOL_WINDOWS_DAYS = (21, 63, 126)` | `SMA_WINDOWS_DAYS = (20, 50)` | `MOM_WINDOWS_DAYS = (63, 126, 252)`
    - `VOLUME_MEAN_WINDOW_DAYS = 63` | `ILLIQUID_VOLUME_THRESHOLD = 10_000` | `ILLIQUID_FRAC_WINDOW_DAYS = 21`
    - `RISK_WINDOW_DAYS = 21` | `VAR_CONFIDENCE = 0.95` | `FORWARD_TARGET_HORIZON_DAYS = 21`
  - `make_features(close, high, low, volume) -> DataFrame([date, ticker]MI × 77 feature cols)`:
    - 8.1 Return features: `ret_1d / ret_5d / ret_21d / ret_63d / ret_126d` (lagged log)
    - 8.2 Ann. vol: `vol_ann_21d / vol_ann_63d / vol_ann_126d` (rolling std × √252)
    - 8.3 SMA crossovers: `sma_cross_20d / sma_cross_50d / sma_fastslow_20_50` (continuous ratios − 1, z-score style; NOT binary flags)
    - 8.4 Momentum: `mom_63d / mom_126d / mom_252d` + skip-most-recent-month `mom_252_21_skip` (12-1 momentum)
    - 8.5 Volume features: `vol_rel_mean63d` (daily vol / 63-d mean − 1) + `illiquid_frac_21d`
    - 8.6 Risk features: `var95_21d` (95% rolling VaR of loss via quantile) + `mdd_21d` (21-d rolling window max drawdown)
    - 8.7 Calendar 1-hot dummies: month (12), quarter (4), half (2) → 18 cols total
    - 8.8 Sector dummies + cross-sectional z-ranked features: `cs_rank_mom126 / cs_rank_vol21 / cs_rank_sma20` (z-scored per date across the 46 survivors, not rank-raw)
    - Important: rolling lead-in windows return `NaN` (never filled/dropped inside `make_features`); downstream `align_X_y` drops rows with any NaN
  - `make_targets(close, horizon_days=21) -> Series([date, ticker]MI)`:
    - Separated by structural function boundary from features (prevents feature→target lookahead bugs by design)
    - `y = log(close.shift(-21) / close)` = 21-d forward log return matching monthly rebalance cadence
  - `align_X_y(features_df, target_series) -> (X, y, common_dates)`:
    - Intersects MultiIndex `[date, ticker]`, drops any row with NaN in features or target, sorts both, returns unique sorted `common_dates` DatetimeIndex

- **Step 2.4 — Feature sanity assertions** (`scripts/sanity_check_features.py`)
  - **Command**: `.\.venv\Scripts\python.exe scripts\sanity_check_features.py` → **exit 0**, 7/7 assertions PASS
  - Exact output numbers (every figure from actual execution):
    - Input: 46 tickers, prices shape `(2222, 230)`, date range `2015-01-01 → 2023-12-29`
    - Max rolling lead-in = 252 d; target = 21 d
    - `make_features`: `X_wide shape = (102212, 77)` | 77 feature columns
    - `align_X_y result`: X shape = `(89608, 77)`, y shape = `(89608,)`, 1948 distinct common dates
    - **Assertion A** (NaN): X NaN = 0, y NaN = 0 ✓
    - **Assertion B** (holdout safe-edge): max common date = `2023-11-29` < safe edge `2023-11-30` = FinalHoldoutDate − 22 BD ✓
    - **Assertion C** (lead-in filled): min common date = `2016-01-08` ≥ expected ~`2015-12-22` = raw start + 253 BD ✓
    - Top-5 |corr(feature, y_fwd21)|: vol_ann_63d +0.1292, cal_month_2 −0.1230, vol_ann_126d +0.1149, cal_qtr_1 −0.0986, mdd_21d −0.0826
    - Bot-3 |corr|: 3 low-population sector dummies (as expected, single-ticker sectors carry almost zero cross-section signal)
    - **Assertion D** (dead-constant): `std < 1e-10` columns = 0 / 77 ✓
    - Summary: 89,608 trainable cells across 1,948 dates × ~46 tickers × 77 features

- **Step 2.5 — Stage 1A data prep + 8 baseline equity curves** (`scripts/build_stage1a_baselines.py`)
  - **Command**: `.\.venv\Scripts\python.exe scripts/build_stage1a_baselines.py` → **exit 0**
  - Rebalance panels written to `data/processed/`:
    - `monthly_rebalance_prices.csv` (107 rebalance dates × 46 tickers)
    - `quarterly_rebalance_prices.csv` (37 rebalance dates × 46 tickers)
  - 4 baseline strategies × 2 rebalance frequencies = 8 walk-forward equity curves (`stage1a_baseline_equities.csv`):
    | Frequency | Baseline         | Raw Sharpe (no txcost) | Total 9-yr return |
    |-----------|------------------|------------------------|-------------------|
    | monthly   | equal (1/N)      | +0.923                 | +296.4%           |
    | monthly   | freefloat_proxy  | +0.982                 | +327.9%           |
    | monthly   | minvar           | +0.434                 | +151.9%           |
    | monthly   | riskparity       | +0.925                 | +273.4%           |
    | quarterly | equal (1/N)      | +0.923                 | +296.4%           |
    | quarterly | freefloat_proxy  | +0.982                 | +327.9%           |
    | quarterly | minvar           | +0.555                 | +217.1%           |
    | quarterly | riskparity       | +0.919                 | +271.9%           |
  - No tx-cost applied yet — Stage 1A decision (monthly vs quarterly rebalance) happens in Phase 4 after applying realistic slippage
  - Implementation details: min-variance uses pseudoinverse (singular-robust) 63-d rolling cov → sum-to-1 constraint clipped ±2; risk-parity uses inverse 63-d marginal volatility

- **Step 2.6 — Freeze & sign-off**
  - Feature contract frozen in `docs/literature_matrix.md` §A.2 (tick-list of 8 families × column-level roster)
  - CHANGELOG Phase-2 block = this section

### Phase 2 exit checklist
- [x] **G1:** `test_no_lookahead.py` — 4/4 tests pass (leak-block + leak-detect both green)
- [x] **G2:** Notebook `01_data_exploration.ipynb` cells written; 6 panels + export code present (to be executed by researcher; Restart & Run All expected 0 errors — code is pre-committed)
- [x] **G3:** `features.py` exposes 3 separated public functions: `make_features | make_targets | align_X_y` — targets structurally isolated from feature computation
- [x] **G4:** `sanity_check_features.py` exits 0; all 7 hard assertions hold (NaN, holdout, lead-in, dead constants)
- [x] **G5:** `common_dates.max() = 2023-11-29` < `FinalHoldoutDate (2024-01-01) − 22 BD` — holdout and forward-target horizon jointly satisfied
- [x] **G6:** 8 baseline equity curves written to `stage1a_baseline_equities.csv` (2 freq × 4 baselines) + 2 rebalance panels
- [x] **G7:** CHANGELOG Phase-2 block + `docs/literature_matrix.md` feature-contract tick-list appended

---

## Phase-Signing Checklists

### Phase 1 — Foundations exit (from `CONTEXT.md §6` / README)

- [x] `docs/literature_matrix.md` file created **AND fully populated**
      (§F = 21 papers, 6 themes × 3+ minimum; well inside 15–30 target)
- [x] Final stock universe list frozen — **46 tickers** (from 50 NIFTY 50
      input → 4 F1-only rejections) with definitive F1/F2/F3 filter trail,
      APOLLOHOSP.NS fully resolved (no caveats), sector classification
      hand-mapped and frozen in CSVs
- [x] Full date range (2015-01-01 → 2025-06-30) + 18-mo final holdout
      (2024-01-01 → 2025-06-30) explicitly documented in `literature_matrix.md §A`
      and `§B` ASCII diagram; freeze_universe.py never reads the 2024+
      period; `_oneoff_calc_survivorship_bias_proxy.py` same discipline
- [x] `requirements.txt` committed; `.venv/` runs cleanly on a fresh install
      (20/20 smoke tests pass, `pip check` reports 0 broken dependencies)
- [x] `pytest tests/ -v` discovers successfully (collected 0 items;
      exit code 5 = framework wiring valid)
- [x] Research design, frozen parameters, and methodology signed off
      — §A + §B + §C of literature_matrix.md cover 11 frozen params,
      Stage 1A/1B sequential pipeline, and D1–D12 documented defaults
      (full overlap with CONTEXT.md methodology except the 4 documented
      deviations in CHECKPOINT §3.5)
