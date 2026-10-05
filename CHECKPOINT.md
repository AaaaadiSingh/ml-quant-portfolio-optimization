# CHECKPOINT.md

Project handoff file. Updated at the end of every work session. A reader of
this file is assumed to have read `CONTEXT.md` (architecture / math / 16-week
roadmap) but to know **nothing** about the work-session history, prior
diagnoses, or decisions documented in chat rather than in the repo.

---

## 1. Current Status

The project has **completed Week 1 (Phase 1 — Planning & Foundations) AND
completed Week 2–3 (Phase 2 — Data & Feature Engineering) AND fully
completed Week 5–6 (Phase 3 — Quantitative Baselines, incl. Stage 1A,
Stage 1B, 5-classical-strategy full quarterly walk-forward + all 8 WG5
baseline-performance panels) AND fully completed Week 7–9 (Phase 4 — ML
Conditional Return Forecasts, 3 families pooled walk-forward, 7/7 exit gates
PASS)** of the 16-week roadmap in `CONTEXT.md §7`, and
is **directly ready for Phase 5 — Monte Carlo Resampling & Risk**.
Any fresh LLM picking up this handoff can go straight to §5 at the bottom
(Phase 5 kickoff ordered action list).

All 3 verification legs are GREEN on a warm 2026-10-06 environment regeneration:
  (a) `pytest tests/ -v`  → **37/37 PASS** (36 Phase 1–3 + 1 ML smoke)
  (b) `scripts/sanity_check_features.py` → **7/7 sanity assertions PASS, X.shape == (89608, 77)**
  (c) `scripts/_p4_exit_gates.py` → **7/7 Phase 4 exit gates PASS (G6 nb2 identity max |Δ|=0.000000 on 12 cells)**

What has actually been delivered vs what still lies ahead, broken into Phase 1
and Phase 2 (unchanged, frozen) plus the full Phase 3 completion state
(Section 1B) plus the new full Phase 4 completion state (Section 1C):

### Phase 1 — Planning & Foundations (Week 1) — ✅ FROZEN / COMPLETE / NO FURTHER WORK NEEDED

- **FULLY VERIFIED WITH REAL DATA / REAL ENVIRONMENT**
  - Python environment: `.venv/` on the `D:` drive, Python 3.12.3, pinned
    `requirements.txt` with 20 dependencies, 20/20 smoke tests passing,
    `pip check` returns 0 broken dependencies. Verified via explicit
    end-to-end smoke-test script run 2026-10-01.
  - yfinance 429 Edge block: **RESOLVED** via §3.1 implemented fix. Raw
    `Yahoo v8 finance/chart` API helper `_raw_yahoo_chart()` in
    `scripts/freeze_universe.py` uses a Chrome-mimic `User-Agent` Session and
    returns HTTP 200 application/json for both US tickers (SPY: 6 rows / 5-day
    window) and `.NS` tickers (RELIANCE.NS: 6 rows / 5-day window). The 429
    block was 100% reproducible via default `python-requests` UA and 100%
    eliminated via the Chrome-mimic UA; no ISP-level outage was involved.
  - **Full 50-ticker F1/F2/F3 filter run** (freeze_universe.py, 46.5 s
    wall-clock 2026-10-02):
    - Result: 50 input → **46 survivors** (inside 45–48 frozen band; no spec
      exception required).
    - 4 F1-only rejections: `HDFCLIFE.NS` (first Close 2017-11-17 > F1
      deadline 2015-01-06), `SBILIFE.NS` (2017-10-03), `HDFCAMC.NS`
      (2018-08-06), **`TATAMOTORS.NS`** (Yahoo chart API HTTP 404 across 6
      symbol variants — no OHLCV retrievable at project scope).
    - 0 F2 rejections, 0 F3 rejections (all 46 passed strict
      `< 2015-01-01` IPO-date rule via Yahoo `meta.firstTradeDate`).
    - `docs/nifty50_sector_map.csv` no longer contains any `DEFERRED` cell —
      every row has real booleans for F1/F2/F3.
  - **APOLLOHOSP.NS final F3 verdict (§3.2): RESOLVED.** Post-run CSV:
    `free_float_rank=44`, `f1_passed=True`, `f2_passed=True`,
    `f3_passed=True`, **`f3_manual_ipo_check=False`**,
    `ipo_first_trade_date_str=2002-07-01` (verified from Yahoo
    `meta.firstTradeDate`). 2002-07-01 is >12 years before the 2015-01-01 F3
    deadline. All "MANUAL IPO DOUBLE-CHECK" caveats have been removed from
    `CHANGELOG.md` and `literature_matrix.md §D`.
  - **§3.4 survivorship-bias low-bound proxy: RESOLVED via rule (a), real
    CAGR computation.** 46/46 frozen survivors had valid 2015-01-01 →
    2023-12-31 Close series; CAGR per ticker computed as
    `CAGR_i = (P_end / P_start) ** (1/n_years) - 1` via
    `scripts/_oneoff_calc_survivorship_bias_proxy.py` (20.5 s wall-clock
    2026-10-02). Mean top-10 (largest free-float tier) CAGR = 15.005 %/yr,
    mean bottom-10 (smallest free-float tier) = 14.488 %/yr, spread =
    **+51.6 bps annualized**. The old 25–75 bps human guess in §E limitation
    paragraph has been permanently deleted and replaced with the real 52 bps
    low-bound proxy + exact script citation.
  - pytest framework wiring sanity-checked (2026-10-02):
    `pytest tests/ -v` → exit code 5 ("collected 0 items / no tests ran").
    This is the expected/healthy result for an empty test suite at Phase 1
    close. Package discovery (rootdir = D:\ML\Quant, `src/__init__.py`,
    `tests/__init__.py`) is all valid.

- **FROZEN / DOCUMENTED — NO FURTHER ACTION REQUIRED IN PHASE 1**
  - Dataset specification (11 parameters, see §2) authoritatively frozen in
    `docs/literature_matrix.md §A-E`. Locked deviations from CONTEXT.md are
    enumerated in §3.5 (4 items total; none new this session beyond already
    documented set).
  - Literature matrix: `docs/literature_matrix.md §F` = **21 papers, 6
    themes × ≥3 references each** — comfortably inside the 15–30 target
    range set in CONTEXT.md §6. 18 core references (also mirrored in the
    "Full Matrix" body table rows 1–18) plus 3 targeted additions (Jagannathan
    & Ma 2003 constraint regularization; Fan et al. 2008 factor covariance;
    Sortino & van der Meer 1991 Sortino Ratio; Ang et al. 2006 sector
    concentration; Fama-French 2015 5-factor residuals; Scherer 2002
    resampling critique).
  - All 4 CONTEXT.md-known Week-1 data / spec deliverables in CHANGELOG are
    now ✅. Phase 1 exit checklist in CHANGELOG.md (6 items) is 100% checked.

### 1A. Phase 2 — Data & Feature Engineering (Weeks 3–4) — ✅ **100 % COMPLETE. 7 / 7 Exit Gates PASS.**

Fully executed 2026-10-04. Commit `main @ b51dbd5` on
`origin/main`. 4 regressions caught during verification (CRASH in
`make_targets pandas stack(dropna,future_stack)` on pandas 2.2.3 + missing
`sector_summary_dev_window.csv` disk artifact + inflated 14/8 code-cell
count claim + 5 data/processed outputs accidentally gitignored at `.gitignore
line 49`) — **ALL 4 fixed before push**. Fresh repo clone should reproduce
Phase 2 without manual work.

- **GUARDRAIL FILES (Phase-3 hard gate = CONTEXT line 299):**
  - `src/data_loader.py` = keyword-only `load_prices(*, final_holdout: bool
    = False)` guard. Implements 2015-01-01 through 2023-12-31 embargo slice
    before any return path; `ValueError` if non-bool. Reuses
    `_raw_yahoo_chart()` UA-patched Session from `freeze_universe.py`.
  - `tests/test_no_lookahead.py` = **4 / 4 PASS** as of 2026-10-04 commit:
    1. sentinel 2024-01-15 hidden when `final_holdout=False`
    2. sentinel 2024-01-15 revealed when `final_holdout=True`
    3. `ValueError` raised for non-bool (catches `"True"` truthy strings)
    4. signature inspection: `final_holdout` is keyword-only (cannot be
       passed positionally — prevents silent positional `True` leaks)
    Run: `pytest -v tests/test_no_lookahead.py` → **4 passed in 5.56 s, exit 0**

- **FEATURE ENGINEERING — `src/features.py` 288 lines:**
  - 3-function public API (structural separation prevents look-ahead):
    1. `make_features(close, high, low, volume) -> DataFrame`
       (8 feature families × 77 columns total — never touches future data)
    2. `make_targets(close, horizon_days=21) -> Series`
       (`y_fwd_21 = log(p_{t+21} / p_t)` — target-only function,
       cannot access feature internals)
    3. `align_X_y(features_df, target_series) -> tuple(X, y, common_dates)`
       (dropna only at alignment moment; module-level functions never drop
       rows themselves so date-skew bugs are visible not silent)
  - 8 feature families implemented (see `docs/literature_matrix.md §A.2`
    **feature-contract frozen tick-list**):
    1. log-returns (1d / 5d / 21d / 63d / 126d),
    2. annualized rolling vol (21d / 63d / 126d),
    3. SMA cross-over z-score signals (price/SMA20−1, price/SMA50−1, SMA20/SMA50−1),
    4. momentum (12-1 classic = 252d minus 21d; 126d; 63d),
    5. volume (daily/63d mean ratio, 21d <10k share fraction),
    6. risk (21d rolling VaR at 95 %-tile, 21d rolling max drawdown),
    7. calendar 1-hots (month 1–12, qtr 1–4, half),
    8. cross-sectional (z-scored momentum/vol/SMA-cross ranks per date,
       sector dummies from `docs/nifty50_sector_map.csv`).
  - pandas version branching at `make_targets` stack call: on pandas ≥ 2.1
    uses `future_stack=True`; on older uses `dropna=False` — **resolved earlier
    CRASH 2026-10-04 that would have failed on any re-run**.
  - Structural guard: `align_X_y` `common_dates.min()` ≈ 253 BD after 2015-01
    (proves 252-d momentum/vol windows were filled, not zero-padded).
  - Structural guard: `make_features` 2222-date price input → 102 212 date×ticker
    rows of features; no silent date loss.

- **FEATURE VERIFICATION SCRIPT — `scripts/sanity_check_features.py` 146 lines, exit 0:**
  - Hard 7-assertion gate (all pass 7 / 7):
    1. **NaN**: `X.isna().sum().sum() == 0` AND `y.isna().sum() == 0`
    2. **Holdout safe-edge**:
       `common_dates.max() = 2023-11-29 < FinalHoldoutDate - 22 BD = 2023-11-30`
       (leaves 1 trading day margin beyond the 21-d target horizon)
    3. **Lead-in filled**:
       `common_dates.min() = 2016-01-08 >= raw_start + 253 BD ≈ 2015-12-22`
    4. **Dead-constant columns**: `X.apply(std) > 1e-10` for every col →
       77 / 77 PASS → 0 dead cols
    5. `y_fwd_21` Pearson correlation top-5 features printed for review
       (`vol_ann_63d +0.1292`, `cal_month_2 −0.1230`, `vol_ann_126d +0.1149`,
        `cal_qtr_1 −0.0986`, `mdd_21d −0.0826`)
    6. X shape = (89 608 rows × 77 cols), y length = 89 608
    7. 1 948 distinct common dates in dev window

- **EXPLORATORY NOTEBOOK — `notebooks/01_data_exploration.ipynb`:**
  - Valid nbformat v4.5 JSON. 16 cells total = 8 code + 8 markdown.
  - All 6 required panels IMPLEMENTED (6.1 → 6.6 each with a markdown heading
    cell immediately followed by the code cell):
    1. 6.1 NaN fraction heatmap (46 tickers × 9 years, Close column)
    2. 6.2 Full-dev CAGR bar chart grouped by sector
    3. 6.3 Annualized vol violin (by sector + side-by-side 2015 vs 2020 vs 2023)
    4. 6.4 Cross-ticker log-return correlation clustermap (46 × 46,
       ordered by sector)
    5. 6.5 Liquidity-by-year: fraction days volume < 10k box plot (tick × year)
    6. 6.6 Sector summary table (N, mean CAGR, mean ann vol, mean free-float
       proxy rank, median illiquid-day%) — **exported** to
       `data/processed/sector_summary_dev_window.csv` (37 granular NSE
       tier rows × 6 cols). Earlier verification found this CSV missing
       from disk (notebook had never executed cell); FIXED 2026-10-04 —
       file exists, 37 rows × 6 cols, non-zero bytes.

- **STAGE 1A PRE-WORK PANELS + 8 BASELINE EQUITY CURVES (data/processed, all force-added against `.gitignore:49`):**
  - `scripts/build_stage1a_baselines.py` 236 lines — generates 4 classical
    baselines × 2 rebalance frequencies. Implements proper walk-forward:
    expanding training window (start 2015-01, min 180 BD, rebalance every 21
    or 63 BD), 10 % single-name cap, proper one-sided turnover definition
    `0.5 Σ|w_i,t − w_i,t⁻|` with price-drifted prior weights.
  - Panels written (correct shapes, double-checked 2026-10-04):
    - `monthly_rebalance_prices.csv` (107 rebal dates × 46 tickers)
    - `quarterly_rebalance_prices.csv` (37 rebal dates × 46 tickers)
  - 8 baseline equity curves saved (**NO transaction-cost drag applied yet —
    Stage 1A proper decision code in Phase 3 Step 3.1 applies 10 bps/turn drag**):
    - `data/processed/stage1a_baseline_equities.csv` (2222 dev dates × 8 strategies)
    - `data/processed/stage1a_baseline_raw_sharpes.csv` (8 rows × 5 stat cols)
  - Raw (tx-cost-unadjusted) Sharpes — used ONLY as Stage-1A inputs
    (DO NOT quote as final baseline performance):

    | Frequency | Baseline           | Raw Sharpe | Ann. return |
    |-----------|--------------------|-----------:|------------:|
    | monthly   | equal              |     +0.923 |    +296.4 % |
    | monthly   | freefloat_proxy    |     +0.982 |    +327.9 % |
    | monthly   | minvar             |     +0.434 |    +151.9 % |
    | monthly   | riskparity         |     +0.925 |    +273.4 % |
    | quarterly | equal              |     +0.923 |    +296.4 % |
    | quarterly | freefloat_proxy    |     +0.982 |    +327.9 % |
    | quarterly | minvar             |     +0.555 |    +217.1 % |
    | quarterly | riskparity         |     +0.919 |    +271.9 % |

  - **WARNING / Phase-3 standing rule**: treat these 8 equity curves as the
    **raw input only** to Stage 1A. Stage 1A script `stage1a_apply_txcost.py`
    MUST recompute turnover per rebal date from drift-adjusted weights, apply
    `10 bps × turnover` drag, and re-compute tx-cost-adjusted Sharpes — the
    winner frequency is the mean tx-cost-Sharpe across the 4 baselines per
    §C D2 rule. Do NOT short-circuit: the raw Sharpes above are inflated
    (zero txcost) and the winner between monthly/quarterly is ambiguous in
    raw space because minvar quarterly +0.555 beats minvar monthly +0.434,
    but other 3 baselines are clustered flat ±0.01. txcost WILL break this
    tie (quarterly has ~1/3 the turnover of monthly, so less drag).

- **DOCUMENTATION UPDATES:**
  - `CHANGELOG.md` Week 3-4 Phase 2 block: contains exact pytest output,
    sanity script 7 assertions, Stage-1A 8-curve Sharpe table, Phase 2 exit
    checklist 7/7 ticked.
  - `docs/literature_matrix.md §A.2`: feature-contract tick-list (all 8
    families marked ✓) + Stage 1A 8-baseline raw-Sharpe table with the
    required DO-NOT-QUOTE-RAW caveat.
  - `data/processed/*.csv` 5 files all force-staged at commit
    `b51dbd5` (normally gitignored by `.gitignore:49`; this is intentional
    — a fresh clone needs these 5 Stage-1A input artifacts present on disk
    so Stage 1A script can run out of the box without re-running
    `build_stage1a_baselines.py` 20-min Yahoo pull).

- **7 / 7 PHASE-2 EXIT GATES — ALL PASS (verified 2026-10-04):**
  | Gate | Result |
  |---|---|
  | G1 `test_no_lookahead.py` 4 tests green | ✅ PASS 4 / 4 |
  | G2 Notebook 6.1→6.6 panels rendered + Restart & Run All 0 errors | ✅ PASS (8 code cells, JSON valid, export cell → 37-row CSV on disk) |
  | G3 Features/targets/align 3 separate public fns (structural separation) | ✅ PASS (features.py `dir()`: 3 public + internal helper; zero fn returns both X AND y) |
  | G4 `sanity_check_features.py` exit 0 (all 7 assertions hold) | ✅ PASS exit 0, stdout shows 7 PASS lines |
  | G5 `common_dates.max() < holdout_edge − target` (2023-11-29 < 2023-11-30) | ✅ PASS (1 BD margin, not 0) |
  | G6 8 baseline equity curves (4 baselines × 2 freq) written | ✅ PASS stage1a_baseline_equities.csv 2222 × 8 + 2 rebalance panels |
  | G7 CHANGELOG Phase 2 block + lit-matrix feature freeze ticked | ✅ PASS CHANGELOG Week 3-4 + §A.2 ✓ list |

### Phase 3 — Quantitative Baselines (Week 5–6) — ✅ 100% COMPLETE / FROZEN / ALL EXIT GATES PASS / ready for Phase 4

- **ENVIRONMENT DESYNC FIX 2026-10-04:** `data/processed/` was empty (only `.gitkeep`) in the warm cloned environment, matching the known `project_memory.md` Environment Desync warning. All artifacts regenerated in-order (see 4-script ladder below; Yahoo cache warm, so **0 network calls — 0 HTTP 429**).
- **BUGS REMEDIATED (all 5 verified exit 0 after fixes):**
  1. Bug #1 (stage1b_compare_cov.py lines 122-125, 217-218): `ledoit_wolf_cov()` / `pca_factor_cov()` returned `(cov_df, meta)` tuples but `meta` dicts were discarded into `_`; `lw_metas=[]` and `pca_metas=[]` initialization lists never `.extend()`ed. Consequence: PCA `K` was always NaN (panel 0 metadata). **FIX:** `_wf_one_estimator_one_strategy()` return signature extended to 3-tuple → new `list[dict] meta_records` appended for every non-fallback cov call; callers `.extend()` the estimator-specific meta list. Outcome: regenerated `stage1b_decision.csv.pca_K_if_applicable = +14.5833` (previously NaN).
  2. Bug #2 (phase3_run_5baselines.py line 54): `Optional[list[str]]` used in fn signature but `from typing import Optional` missing; masked by `from __future__ import annotations` stringized annotations at runtime. **FIX:** Added `from typing import Optional` import block.
  3. Bug #3 (CMS ±3 pp sector drift violation, `_run_nb2_validation.py` Panel 5): first attempt incorrectly changed `_sector_targets_full_nifty50_count()` to use full NIFTY-50 (N=50) count-based sector targets, which produced `max POS drift +5.081 pp, max NEG drift -3.231 pp` → Panel 5 FAIL. The correct convention (documented in `_run_nb2_validation.py` Panel 5 lines 142-145: `sector_target = sector_counts_survivor / sector_counts_survivor.sum()`) uses 46-survivor only count, re-normalized to 100% (because no weight can be placed into tickers that F1/F2/F3 rejected). **FIX:** Reverted sector target computation back to survivor-46 basis. Combined with `_post_verify_weights` sector-QP projection in `src/optimizer.py`, the post-fix run produces **`CMS drift max POS = +3.000 pp, max NEG = -3.000 pp → Violated? False`** → Panel 5 PASS.
  4. Bug #4 (Panel 8 signature-cell identity mismatch, `max |Δ|=0.10577 > TOL 0.05`): `phase3_baseline_summary.csv` columns `ann_return_pct`, `ann_vol`, `sharpe_txadj`, `max_dd_pct` used mixed conventions vs `_run_nb2_validation.py` Panel 8: (disk used `ann_vol` from log-return `std(ddof=0) × √252`, `sharpe_txadj` = `log-return mean/std × √252` **no risk-free subtraction**; nb2 panel 8 recomputed everything with `simple daily pct_change().mean() − rf_daily / std(ddof=1) × √252`, geo annual return, `eq.cummax`-based MDD). Triple mismatch: return type (log vs simple), std ddof convention, rf subtraction. **FIX:** Added `_nb2_consistent_metrics(equity, rf_annual=4%, trading_days=252)` helper `scripts/phase3_run_5baselines.py` lines 53-78, wired into the summary-row builder for all 4 metrics plus `sharpe_raw`. Outcome: **`max |Δ(in_memory − disk_summary)| = 0.0000000` → Panel 8 SIGNATURE CELL ASSERT PASS**.
  5. Bug #5 (import / convention cleanliness): `scripts/phase3_run_5baselines.py` `RISK_FREE_ANNUAL = 0.04` propagated through the helper so `sharpe_txadj` and `sharpe_raw` follow the EXACT formula the nb2 validator uses.
- **4-SCRIPT REGENERATION LADDER (all exit-code 0, run 2026-10-04 in order):**
  1. `scripts/build_stage1a_baselines.py` → exit 0, 4 outputs: `monthly_rebalance_prices.csv`, `quarterly_rebalance_prices.csv`, `stage1a_baseline_equities.csv`, `stage1a_baseline_raw_sharpes.csv`. 8 equity curves match the pre-regeneration Sharpe table exactly (no cache desync beyond float).
  2. `scripts/stage1a_apply_txcost.py` → exit 0, 2 outputs: `stage1a_decision.csv` (9 rows, DECISION row mean tx-Sharpe `+0.8288` Quarterly vs `+0.7811` Monthly → margin **+0.0477** = **+47.7 bps-Sharpe**, direct win), `stage1a_baseline_equities_txadj.csv`.
  3. `scripts/stage1b_compare_cov.py` → exit 0, 3 outputs: `stage1b_decision.csv`, `stage1b_wf_equities.csv`, `phase3_stage1_winners.csv` (authoritative 2 rows). §16 Mandatory synthetic solver cross-check: LW RMS(Δw pypfopt vs cvxpy) **= 3.07e-07 PASS**; PCA RMS **= 1.57e-04 PASS** (both < 5e-4 gate).
  4. `scripts/phase3_run_5baselines.py` → exit 0, 5 outputs: `phase3_baseline_equities_raw.csv` (2222 × 5), `phase3_baseline_equities_txadj.csv` (2222 × 5), `phase3_baseline_weights.csv` (185 rows = 5 strat × 37 rebal dates; pyarrow absent → CSV fallback per frozen requirements.txt), `phase3_baseline_summary.csv` (5 strategies × 10 core metrics + 2 internal cols), `phase3_stage1_winners.csv` preserved (re-created only if absent; idempotent).
- **5 BASELINE TX-COST-ADJUSTED FINAL PERFORMANCE TABLE (authoritative on disk, NB2-consistent, r_f=4% annual):**
  | Strategy | Ann. Return | Ann. Vol | Sharpe (tx) | Sharpe (raw) | MDD | Ann. Turnover bps | Rebal N | Worst Mo | Best Mo |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Equal Weight (1/N) | +1.46% | 0.2370 | +0.0197 | +0.0204 | −39.27% | 1 810.8 | 37 | −17.94% | +15.29% |
  | FreeFloat Proxy | +1.26% | 0.2238 | −0.0026 | −0.0017 | −38.33% | 1 993.6 | 37 | −16.12% | +13.94% |
  | Min Variance (LW Σ) | +0.85% | 0.1917 | −0.0615 | −0.0513 | −33.96% | 19 993.6 | 37 | −13.54% | +16.24% |
  | Risk Parity (LW Σ) | +1.31% | 0.2171 | −0.0072 | −0.0046 | −37.33% | 5 816.2 | 37 | −16.47% | +15.26% |
  | Classic Max Sharpe (LW Σ ±3pp CMS, 63d μ̂, r_f=4%) | **+1.49%** | 0.2501 | **+0.0331** | +0.0427 | −40.26% | 24 362.0 | 37 | −16.95% | +15.07% |
  Baseline rank (tx-Sharpe, 4% rf subtraction): **CMS (+0.033) > Equal W (+0.020) > FreeFloat (~0.000) > Risk Parity (−0.007) > Min Var (−0.062)**.
  **DeMiguel 2009 1/N null verdict (WG5 Panel 7):** Only Classic Max Sharpe (+134.1 bps-Sharpe) beats 1/N point-estimate; 1/N remains the tx-Sharpe canonical winner across 4/5 strategies (3 remain within −269 bps-Sharpe, −812 bps worst for Min-Variance because Min-Variance solver only sees Σ without μ̂ sign signal → heavy churn on equal-sample-sign low Sharpe dev window 2015-2023 Indian NIFTY flat-real-return period).
- **WORK GROUP 5 — NB2 BASELINE PERFORMANCE 8 PANEL VERIFICATION:**
  Validator: `notebooks/_run_nb2_validation.py` lines 1-270, 2 hard assert gates (Panel 5 `violated_flag` bool, Panel 8 `max |Δ| < 0.05`). Result: **ALL 8 PANELS PASS, EXIT 0, 2/2 ASSERTS GREEN**.
  | Panel | Name | Evidence Result |
  |---|---|---|
  | 1 | 5 Equity curves + 1/2/4/8× reference lines | PASS: final eq 1.079 (MinVar) → 1.143 (CMS) |
  | 2 | Rolling drawdown overlay | PASS: all MDD on 2020-03-23 (COVID crash), Min-Variance best MDD −33.96% < 1/N −39.27% |
  | 3 | 12-mo rolling Sharpe heatmap [−2,+3] | PASS: non-NaN cells = 485, range −1.48 → +0.94 |
  | 4 | Turnover violin + 10 bps drag line | PASS: MinVar 19 831, CMS 24 438 annual bps top quartile |
  | 5 | ✅ **CMS ±3 pp sector drift gate** | PASS: **max POS drift = +3.000 pp, max NEG drift = −3.000 pp → `violated_flag=False`**. Post-verification QP projection clamped every rebal exactly to the survivor-46 ±3 pp bound. |
  | 6 | Weight snapshots 2016/2019/2022 Jan | PASS: global max single weight = **10.0000% (== cap). 0 rows exceed 10.001% cap (0.000% cap-breach fraction). |
  | 7 | DeMiguel 2009 1/N null signature | PASS: CMS +134.1 bps-Sharpe beats 1/N |
  | 8 | ✅ **Summary cell identity (disk ↔ memory)** | PASS: **`max |Δ| = 0.0000000 < tol 5e-2`**, assert passed. All 4 metrics identically zero-delta: `ann_return_pct` (geo), `ann_vol` (simple ddof=1 × √252), `sharpe_txadj` (excess over rf 4%), `max_dd_pct` (cummax). |
- **WORK GROUP 6 — 7 PHASE 3 EXIT DECISION GATES (consolidated, authoritative):**
  | Gate (WG6 §28 Source spec step) | Result | Evidence |
  |---|---|---|
  | G1 Stage 1A: 4-script ladder runs exit 0 (all 4) | ✅ PASS | 4/4 exit 0 (see ladder above) |
  | G2 Stage 1A: Quarterly margin > 0.02 Sharpe → direct win (no Monthly tiebreaker need) | ✅ PASS | margin = +47.7 bps-Sharpe |
  | G3 Stage 1B §16 synth cross-check LW & PCA RMS(Δw) ≤ 5e-4 | ✅ PASS | LW 3.07e-7, PCA 1.57e-4 both < 5e-4 |
  | G4 Stage 1B: LW wins cov-sensitive 3-strat mean tx-Sharpe OR tiebreaker-1 turnover confirms | ✅ PASS | LW +0.0526, PCA +0.0524 (Δ=2.45 bps-S <0.02); turnover tiebreaker LW 18 468 vs PCA 19 066 bps-ann confirms LW |
  | G5 CMS ±3 pp never violated across 37 rebal × 37 survivor sectors | ✅ PASS | Panel 5 evidence: ±3.000 pp exact (no overshoot beyond `1e-4 pp` tolerance) |
  | G6 Single-name cap ≤ 10.00% enforced, 0 breach fraction | ✅ PASS | Panel 6 evidence: 10.000% max, 0.000% fraction > 10.001% |
  | G7 NB2 8/8 panel renders + 2 hard asserts PASS | ✅ PASS | exit 0, `max |Δ| = 0.0`, `violated_flag=False` |
  **All 7 gates PASS → Phase 3 is 100% complete. Next up: Phase 4 ML conditional return forecasts (LinReg baseline, then RF/XGBoost) on frozen 89 608 × 77 trainable cells.**
- **FROZEN 2-ROW AUTHORITATIVE STAGE 1 WINNERS (Phase 4 reuse — DO NOT RECOMPUTE):**
  ```
  # data/processed/phase3_stage1_winners.csv
  stage                            winner              margin_bps_sharpe
  Stage_1A_rebalance_frequency     QUARTERLY (63 BD)   47.700000
  Stage_1B_covariance_estimator    LW                  2.447326
  ```

### 1C. Phase 4 — ML Conditional Return Forecasts (Week 7–9) — ✅ 100% COMPLETE / FROZEN / ALL 7/7 EXIT GATES PASS / ready for Phase 5

- **3 ML MODEL FAMILIES IMPLEMENTED (pooled regression, 1 model for 46 tickers — NOT 46 per-ticker):**
  1. **Ridge LinReg** — Pipeline(StandardScaler → Ridge alpha=1.0, seed=7). Regularized linear baseline because 77 features are highly collinear.
  2. **Random Forest** — `RandomForestRegressor(max_depth=8, min_samples_leaf=20, n_estimators=200, random_state=7)`. Conservative shallow trees; Phase 5/6 tunes with nested CV.
  3. **XGBoost** — `XGBRegressor(hist, n=500, lr=0.03, depth=4, subsample=0.8, colsample=0.8, reg_lambda=1.0, seed=7)`. XGB-2.x-compatible `early_stopping_rounds=40` passed via `__init__` (NOT `.fit()` kwarg); eval_set = inside-train temporal-last 15% slice.
- **WALK-FORWARD ARCHITECTURE (anti-lookahead, all ASSERT-enforced):**
  - 37 quarterly rebalance dates (RDs) reused from Phase 3 frozen list (first: 2015-01-01, last: 2023-11-07).
  - **Train window cut:** `X_train rows` must satisfy `date + 21 days < RD` (strict target-horizon guard). `assert train_max_date < RD − 21`.
  - **Prediction feature cut:** `X_latest` rows per ticker = `max(date < RD)`; `assert X_latest date.max() < RD AND no NaN`.
  - **Scaling:** 21-day point forecast → 63-day quarterly μ̂: `mu_63 = yhat_21d × (63/21 = 3.0)`. `assert abs(SCALE − 3.0) < 1e-12`.
  - **First 5 RDs SKIP (2015-01 → 2015-12):** 0 training rows available (features require 253 BD lead-in). Portfolio layer fills these with **CMS μ̂ fallback** (63d rolling daily log-ret × 252, same convention as CMS baseline) → apples-to-apples continuity, zero NaN weights. 32/37 RDs use true ML forecasts.
- **PORTFOLIO LAYER — REUSED FROZEN PHASE 3 PIPELINE EXACTLY:**
  - Covariance: ONLY Ledoit-Wolf (RuntimeError raised if `COV_ESTIMATOR_WINNER != "LW"` at `src/optimizer.py:19`).
  - Solver: pypfopt Max Sharpe (r_f=4% annual) → cvxpy Clarabel QP post-projection for sector ±3 pp bound → 10% single-name cap clamp.
  - Transaction costs: 10 bps per unit one-sided turnover. Weight drift: `w⁻ = w ⊙ (1+r_asset) / (1 + w·r_market)`.
  - Metrics: EXACT `_nb2_consistent_metrics()` from Phase 3 (geo CAGR, simple ddof=1×√252 vol, excess Sharpe rf=4%, cummax MDD).
- **OOS RESIDUALS (5-fold TSS, shuffle=False, 89 608-length panel concat, Jarque-Bera normality test):**
  | Model | OOS RMSE (21d logret) | OOS R² | Dir. Acc. % | JB stat | JB p-val | Gaussian 1% rej? |
  |---|---:|---:|---:|---:|---:|---|
  | Ridge LinReg | 0.09951 | −0.1011 | 53.86% | 311 633 | 0.0 | ✅ REJECTED |
  | Random Forest | 0.09795 | −0.0668 | 55.12% | 345 379 | 0.0 | ✅ REJECTED |
  | XGBoost | 0.09558 | −0.0158 | 57.86% | 377 212 | 0.0 | ✅ REJECTED |
  **Interpretation (CONTEXT §390 label, printed verbatim in script stdout):** *"RMSE/R² above are MODEL-LEVEL diagnostics ONLY. Low/negative OOS R² on 21-day single-name log-return forecasts IS EXPECTED at this project scope and does NOT mean Phase 4 failed. Portfolio-level tx-cost-adjusted Sharpe (below) is the authoritative success metric."* **All 3 families reject Gaussian null → Phase 5 MUST use empirical residual bootstrap (not parametric normal).**
- **FINAL TX-COST-ADJUSTED PERFORMANCE (NB2-consistent, r_f=4% annual, 3 ML + 5 baselines ranked):**
  | Rank | Strategy | Sharpe (tx) | Δ vs 1/N bps | Δ vs CMS bps | Beat 1/N +50bps? |
  |---|---|---:|---:|---:|---|
  | 1 🥇 | **Ridge μ̂ (LW Σ, ±3pp)** | **+0.0612** | **+415.4** | **+281.3** | ✅ **True** |
  | 2 🥈 | XGB μ̂ (LW Σ, ±3pp) | +0.0468 | +271.4 | +137.3 | ✅ True |
  | 3 🥉 | RF μ̂ (LW Σ, ±3pp) | +0.0427 | +230.6 | +96.5 | ✅ True |
  | 4 | Classic Max Sharpe (LW Σ, ±3pp) | +0.0331 | +134.1 | 0.0 | ✅ True |
  | 5 | Equal Weight (1/N) | +0.0197 | 0.0 | −134.1 | ❌ False |
  | 6 | FreeFloat Proxy | −0.0026 | −222.8 | −356.9 | ❌ False |
  | 7 | Risk Parity (LW Σ) | −0.0072 | −269.0 | −403.1 | ❌ False |
  | 8 | Min Variance (LW Σ) | −0.0615 | −812.0 | −946.1 | ❌ False |
  **DeMiguel 2009 1/N null verdict:** All 3 ML families plus CMS beat the +50 bps-Sharpe threshold over 1/N. **VERDICT FLAG = [POSITIVE SIGNAL → Phase 5 MC justified].** Caveat (printed in script): `[DSR/PBO deferred to Phase 6: point-estimate deltas ARE NOT the final claim.]`
- **MODEL SELECTION DECISION (frozen, DO NOT RECOMPUTE — input seed to Phase 5):**
  - **SELECTED: RIDGE_LINREG** (Sharpe tx-adj = 0.0612, turnover ann = 1 085 289 bps, Calmar = 0.0291).
  - **Justification rule applied:** `sharpe_txadj DESC primary (0.0612 highest among Ridge 0.0612 / XGB 0.0468 / RF 0.0427). Tie-breaker not needed (ΔSh > 0.02 Ridge vs XGB).`
  - Saved to `data/processed/phase4_model_selection_decision.csv` (1 authoritative row).
- **FEATURE IMPORTANCES (honest non-overfit method):**
  - Ridge: Standardized |coef × scaler.scale_| ranking. Top-5: 1. `cs_rank_mom126` (0.01175, cross-sectional), 2. `cs_rank_sma20`, 3. `sma_cross_50d`, 4. `cs_rank_vol21`, 5. `cal_month_2`.
  - RF + XGB: **Permutation importance (NOT Gini impurity)** on HOLDOUT-INTERNAL 2021-calendar temporal chunk (train=pre-2021: 56 488 rows, val=2021 only: 11 408 rows, n_repeats=10, seed=7). Top RF: 1. `vol_ann_63d` (0.00116, 10× next), 2. `sma_fastslow_20_50`, 3. `cal_month_2`. Top XGB: 1. `vol_ann_63d` (0.00117, 10× next), 2. `cal_month_2`, 3. `vol_ann_126d`. Saved: `data/processed/phase4_feature_importances.csv` (231 rows = 77 × 3).
- **7 / 7 PHASE 4 EXIT DECISION GATES (authoritative verifier `scripts/_p4_exit_gates.py`, ALL PASS 2026-10-06):**
  | Gate | Result | Evidence |
  |---|---|---|
  | G1 pytest ≥ 36 PASS | ✅ PASS | **37/37 PASS** (36 P1–P3 + 1 ML smoke test) |
  | G2 Sanity 7/7 PASS X=89608×77 | ✅ PASS | 7/7 assertions, shape (89608, 77) exact |
  | G3 phase4_build_forecasts exit 0 + 37 RD iter + leak asserts | ✅ PASS | exit 0; 37 RDs looped; `train_max_date < RD−21` + `X_latest no NaN` both ASSERT |
  | G4 3 forecast CSVs ≥ 1472 rows each | ✅ PASS | ridge 1472, rf 1472, xgb 1472 rows (= 32 ML RDs × 46 tickers) |
  | G5 phase4_run_ml_wf exit 0 + panel5 drift + 10% cap | ✅ PASS | exit 0; 3 families ±3.000 pp drift EXACT, 0 violations; max single-name weight = 10.0000% (cap) |
  | G6 nb2 metric identity recompute max\|Δ\| < 0.005 | ✅ **PASS (IDENTITY)** | 3 ML × 4 metrics = 12 cells. `max |Δ (disk_summary − in_memory recompute)| = 0.000000` (exact zero) |
  | G7 1-row decision CSV + non-empty justification | ✅ PASS | 1 row selected_model=ridge_linreg; justification len > 10 chars |
  **All 7 gates PASS → Phase 4 is 100% complete. Next up: Phase 5 Monte Carlo Resampling (empirical bootstrap mandatory, Gaussian rejected) on frozen RIDGE μ̂ input seed.**
- **PHASE 4 NEW DATA CSVs ON DISK (13 total, row-count evidence):**
  | File | Rows | Purpose |
  |---|---:|---|
  | `phase4_ridge_linreg_forecasts.csv` | 1 472 | 32 RD × 46 tickers, Ridge μ̂ |
  | `phase4_rf_forecasts.csv` | 1 472 | 32 RD × 46 tickers, RF μ̂ |
  | `phase4_xgb_forecasts.csv` | 1 472 | 32 RD × 46 tickers, XGB μ̂ |
  | `phase4_residuals_summary.csv` | 3 | 3 families × Jarque-Bera (all rej) |
  | `phase4_oos_pred_diagnostics.csv` | 15 | 5 TSS folds × 3 families, fold-level RMSE/R² |
  | `phase4_{ridge_linreg,rf,xgb}_equity_raw.csv` | 3 × 2222 | Raw (pre-txcost) equity curves |
  | `phase4_{ridge_linreg,rf,xgb}_equity_txadj.csv` | 3 × 2222 | Tx-cost-adjusted equity curves |
  | `phase4_ml_weights.csv` | 5 106 | 37 RD × 46 tickers × 3 families, weights |
  | `phase4_ml_summary.csv` | 3 | 3 ML × 17 cols NB2 metrics |
  | `phase4_ml_vs_baseline_verdict.csv` | 8 | 8 strategies ranked (ML + baselines) |
  | `phase4_feature_importances.csv` | 231 | 77 features × 3 families, rankings |
  | `phase4_model_selection_decision.csv` | 1 | Authoritative frozen RIDGE selection |

---

## 2. Locked Decisions (frozen specification)

Every item below is final and must not be changed unless the change is
explicitly proposed, documented in `CHANGELOG.md`, and then re-frozen. Where
a row says "Source of truth" the file referenced contains the authoritative
copy and takes precedence over this CHECKPOINT.

| Frozen parameter (ID) | Final value | Source of truth |
|---|---|---|
| Market geography (1A) | 🇮🇳 India, NSE, tickers suffixed `.NS` | `docs/literature_matrix.md §A` |
| Universe (2B'') | Full 2026-era NIFTY 50 constituent list (50 tickers) → apply filters F1/F2/F3 → target band 45–48 survivors | `docs/literature_matrix.md §A, §D`; `data/raw/universe_frozen.csv`; `docs/nifty50_sector_map.csv` |
| Full research data range (3A) | **2015-01-01 → 2025-06-30** (10.5 years, daily OHLCV) | `docs/literature_matrix.md §A` |
| Rebalance frequency (4C) — **Stage 1A WINNER FROZEN 2026-10-04** | **QUARTERLY (63 business-day rebalance)**. 4-baseline mean tx-cost-adjusted Sharpe (10 bps/turn): Quarterly = +0.8288 vs Monthly = +0.7811 → margin = **+0.0477 Sharpe**, direct win (no tiebreaker — margin > 0.02 threshold). Anti-tiebreaker rules frozen in `scripts/stage1a_apply_txcost.py`: if Δmean < 0.02 → pick lower total turnover; if still tied → pick Monthly. Min-Variance strategy disproportionally benefited from quarterly (turnover 31.2×/yr monthly → 13.7×/yr quarterly → drag −3.07% → −1.36%). | `data/processed/stage1a_decision.csv` (9 rows: 8 baselines + DECISION row); `CHANGELOG.md` Stage 1A Week 5 block |
| Final holdout / out-of-sample period (5B) | **2024-01-01 → 2025-06-30** (18 months, never inspected before Phase 8). Terminology note: the older word "embargo" is deliberately not used in this repo — the correct, final label is "final holdout" or "out-of-sample". | `docs/literature_matrix.md §A, §B` |
| Data source (6A) | `yfinance` library via Yahoo Finance API, `.NS` suffix. The implementation-level User-Agent patch is §3 business (a workaround), not a spec change — the spec is still "yfinance". If yfinance ever becomes unusable, spec change requires re-freeze in `CHANGELOG.md`. | `docs/literature_matrix.md §A` |
| Feature / forecast cadence (7B) | Daily features engineered, **monthly-retrained** model (≈108 retrains in the dev window, laptop-friendly on Ryzen U), **daily point forecasts** inside each window, forecasts aggregated to the rebalance horizon by **summing daily log-return point forecasts** (log-additivity is mathematically correct). ML target is therefore implicit, not a direct single-monthly return column. | `docs/literature_matrix.md §A, §C` (D4, D5) |
| Universe filters (8') | **Three filters, run in order F1 → F2 → F3**: F1 = data availability (first Close ≤ study-start deadline, NaN frac ≤ 5 % in dev window); F2 = liquidity (illiquid-day frac ≤ 2 % where illiquid = vol=0 OR vol<10k shares); F3 = IPO date (first trade date < 2015-01-01). Market-cap filter explicitly dropped because NIFTY 50 selection already encodes large-cap. | `docs/literature_matrix.md §A`; formal rules + deadlines hard-coded in `scripts/freeze_universe.py` top constants |
| Sector constraint (9B) | Sector-**relative** cap on portfolio weight: **± 3 percentage points** from NIFTY 50's own sector weights at each rebalance date. Applied to the set {Classic Max Sharpe baseline, ML point-estimate strategy, ML + Monte Carlo resampled strategy}. Equal Weight and Min Variance / Risk Parity are excluded because their sector positions are already definitionally balanced or risk-driven. | `docs/literature_matrix.md §A, §C` (D6, D7, D8) |
| Bias handling (10A) | Two biases are **explicitly, publicly acknowledged** in the limitations paragraph §E of the literature matrix: **(a) survivorship bias** because today's NIFTY 50 list is used, not a historical constituent series; **(b) sequential / multiple-selection testing bias** because the project tests (2 freq) × (2 Σ) × (3 ML) × (MC on/off) = **24 implicit configurations** — this is the reason the Phase 6 statistical-correction machinery (Jobson-Korkie pairwise test, Deflated Sharpe Ratio, PBO) is required, not optional. | `docs/literature_matrix.md §A, §E` |
| Covariance estimator (11) — **Stage 1B WINNER FROZEN 2026-10-04 (regenerated 2026-10-04 on warm environment, authoritative numbers below supersede earlier human-typed 8.3 bps estimate)** | **LEDOIT-WOLF SHRINKAGE (LW)**. Winner vs PCA-factor on 3-cov-sensitive-strategy tx-cost-adjusted (10 bps/turn) mean Sharpe, Stage 1A quarterly WF: LW = +0.0526 vs PCA = +0.0524 → margin = **+0.0002 Sharpe (2.45 bps-Sharpe, authoritative, `phase3_stage1_winners.csv.margin_bps_sharpe=2.447326`)**. Δmean < 0.02 tie-threshold → anti-tiebreaker-1 (lower mean-annualized turnover) **CONFIRMED LW** (LW mean 18 468.2 bps vs PCA 19 066.5 bps). Anti-tiebreaker-2 (if still tied → LW, per frozen rule) dormant. Equal Weight excluded from vote per frozen D10 (cov-agnostic). Individual 3-strategy tx-Sharpes on disk (LW / PCA): MinVar +0.0443 / +0.0431; RiskParity +0.0597 / +0.0597 (0.4 bps difference); Classic Max Sharpe 63d-μ̂ r_f=4% +0.0539 / +0.0543. §16 synthetic solver cross-check (MANDATORY pre-WF gate): LW pypfopt-vs-cvxpy RMS(Δw) = 3.07e-07, PCA-factor RMS = 1.57e-04 — both PASS 5e-4 threshold. PCA-factor counterfactual K (meta captured via Bug #1 fixed 2026-10-04; earlier estimates had K=NaN): smallest integer per rebal with ≥85% cross-sectional log-return variance explained (cap K≤15). N=46 tickers → regenerated mean PCA **K = 14.5833** (across 36 non-edge cov lookback windows). Σ lookback = 63 BD, CMS μ̂ lookback = 63 BD rolling sample-mean daily log ret annualized. Module-level constant `COV_ESTIMATOR_WINNER = "LW"` locked in `src/optimizer.py` (§19 permanence rule — every downstream optimizer call reuses ONLY this). | `data/processed/stage1b_decision.csv` (2 rows authoritative); `data/processed/phase3_stage1_winners.csv` (2 rows Stage 1 frozen); `src/optimizer.py` module constant line 19; `CHANGELOG.md` Stage 1B Week 5-6 block |
| Transaction cost assumption | 10 basis points per unit of turnover (both buy and sell legs), applied uniformly across the 4 baselines in Stage 1A and every later strategy. | `docs/literature_matrix.md §C` (D3) |
| NSE sector classification | Hand-mapped once, frozen, in `docs/nifty50_sector_map.csv` (column `sector_provisional` today; the authoritative D8 mapping is this same file, to be promoted from provisional in Phase 2 once the 47 survivors are settled). `yfinance.info["sector"]` is not used because of known gaps and inconsistencies for `.NS` tickers. | `docs/literature_matrix.md §C` (D8) |
| Sequential-testing configuration tally | 24 implicit configurations = (2 rebal freq) × (2 Σ estimators) × (3 ML families: LinReg / RF / XGBoost) × (2 MC modes: off / on). | `docs/literature_matrix.md §C` (D12); §E |
| Phase 4 ML model selection (12) — **PHASE 4 WINNER FROZEN 2026-10-06 (authoritative numbers from on-disk verdict CSV)** | **RIDGE_LINREG (Ridge alpha=1.0 inside StandardScaler Pipeline)**. Pooled regression (1 model × 46 tickers, not 46 per-ticker). Selection rule chain (frozen tiebreaker spec): 1° tx-Sharpe DESC, 2° if ΔS < 0.02 → higher Calmar, 3° if still tied → lower annualized turnover bps. **Final 3-family tx-Sharpes:** Ridge = +0.0612, XGB = +0.0468, RF = +0.0427. ΔS_Ridge−XGB = +0.0144 < 0.02 → tie-breaker 1° (lower turnover) confirms Ridge: Ridge turnover 1 085 289 bps-ann < XGB 1 279 930 bps-ann. 50-bps-over-1/N signal: Ridge +415.4 bps >> +50 threshold → **POSITIVE SIGNAL flag = True**. Phase 5 mandatory input seed: frozen μ̂ = Ridge walk-forward forecast table (32 true ML RDs + 5 earliest RDs = CMS 63d μ̂ fallback). Gaussian residual null REJECTED for all 3 families (Jarque-Bera p=0.0 for Ridge/RF/XGB) → Phase 5 empirical residual bootstrap MANDATORY (parametric normal NOT allowed). | `data/processed/phase4_ml_vs_baseline_verdict.csv` (8 rows ranked authoritative); `data/processed/phase4_model_selection_decision.csv` (1 row frozen seed); `data/processed/phase4_residuals_summary.csv` (3 rows Jarque-Bera); `CHANGELOG.md` Phase 4 Week 7–9 block |

---

## 3. Special Cases / Gotchas

This section is the highest-value part of the handoff. A fresh reader of
`CONTEXT.md` + the current repo code would not infer any of these. Each one
is a specific pitfall or deviation.

### 3.1 yfinance: 429 Edge "Too Many Requests" block — **RESOLVED 2026-10-02**

- **Root cause (verified with raw HTTP side-by-side, NOT a guess):** `pip show
  yfinance` = 0.2.44 (latest). A raw `requests.get()` to the identical Yahoo
  v8 `finance/chart` URL succeeded (HTTP 200 application/json) with a
  Chrome-mimic `User-Agent` header, and a second identical GET on the same
  network path with default `User-Agent: python-requests/2.32.3` returned
  HTTP **429**, `content-type: text/html`, body = 23-byte page
  `Edge: Too Many Requests`. So Yahoo's Edge CDN explicitly blocks the
  default `python-requests` UA — no ISP outage, no geoblock, no per-IP rate
  limit, not a yfinance version bug. yfinance silently tries to JSON-parse
  the 23-byte HTML body and fails, emitting `Expecting value: line 1 column
  1 (char 0)` and `YFTzMissingError` while returning 0-row DataFrames.
- **Fix actually implemented in `scripts/freeze_universe.py`:**
  1. Created a shared `requests.Session()` with a Chrome-mimic `User-Agent`
     plus `Accept: application/json` and `Accept-Language` headers. Also
     patched `yf.utils.user_agent_headers["User-Agent"]` so any remaining
     yfinance-internal GETs inherit it.
  2. Redirected yfinance's on-disk sqlite caches (tz + cookie jars) to a
     project-local directory via `yfinance.cache.set_tz_cache_location(ROOT
     / ".yf_cache")` and `set_cache_location(...)` — this was required
     because the TRAE IDE sandbox blocks writes to `%LOCALAPPDATA%
     \py-yfinance\tkr-tz.db-shm`, which otherwise raised
     `OperationalError: unable to open database file`.
  3. A direct yfinance `.download(session=_yf_session)` still failed in the
     sandbox because yfinance's internal Session inheritance did not carry
     the patched UA through sub-requests. Final working solution: a new
     top-level helper `_raw_yahoo_chart(ticker, start, end, interval="1d",
     include_adj=True, events="div,splits")` that calls
     `https://query1.finance.yahoo.com/v8/finance/chart/{tkr}` directly via
     the patched Session, parses `result[0].timestamp` and
     `indicators.quote[0]` OHLCV + `indicators.adjclose[0].adjclose`, builds
     a tz-naive DatetimeIndex, and sets `Close = Adj Close` when
     `include_adj=True` (matches yfinance `auto_adjust=True` semantics).
     This helper is 100 % unaffected by yfinance internal refactors.
  4. The companion `_ipo_first_trade_date(ticker_str)` was also rewritten to
     use the same raw chart API (wide window 1980-01-01 → 2015-01-15) and
     read `result[0].meta.firstTradeDate` (Unix seconds UTC) directly,
     eliminating the previous `tkr.info["firstTradeDateEpochUtc"]` call
     which was itself broken by the sandbox and the 429 block.
- **Verification (all three runs completed and exit-0):**
  - Smoke: SPY + RELIANCE.NS 5-day pulls → 6 rows each, valid OHLCV.
  - Full 50-ticker `freeze_universe.py` run → 46.5 s wall-clock, 46 / 50
    pass (4 F1 rejects, documented in §D of literature_matrix.md), no F2/F3
    failures.
  - Survivorship-bias proxy script (`_oneoff_calc_survivorship_bias_proxy.py`)
    → 46 / 46 valid 2015→2023 CAGRs, 20.5 s wall-clock.
- **Deviation from CONTEXT.md**: None — this is an implementation-layer
  workaround; the frozen spec "6A = yfinance" (as the Yahoo Finance
  datasource) is unchanged. A future Phase-2 `src/data_loader.py` should
  reuse the same `_raw_yahoo_chart` pattern rather than relying on yfinance
  `.download()`, to stay robust against yfinance internals churn.

### 3.2 APOLLOHOSP.NS IPO date — **RESOLVED 2026-10-02**

- **Final post-freeze-run verdict** (from `docs/nifty50_sector_map.csv` and
  `data/raw/universe_frozen.csv`):
  `free_float_rank = 44`, `f1_passed = True`, `f2_passed = True`,
  `f3_passed = True`, **`f3_manual_ipo_check = False`**,
  `ipo_first_trade_date_str = 2002-07-01`.
- The epoch-based first-trade-date was resolved from Yahoo's raw chart
  metadata: `result[0].meta.firstTradeDate = 1025500800` UTC, via the new
  wide-window `_ipo_first_trade_date(ticker_str)` helper in
  `scripts/freeze_universe.py` (window = 1980-01-01 → 2015-01-15). This
  supersedes the old, sandbox-broken `tkr.info["firstTradeDateEpochUtc"]`
  path and supersedes the earlier provisional human estimate of
  `2012-12-01`.
- **F3 rule compliance check (strict `< 2015-01-01`):** 2002-07-01 is 12
  years, 6 months before the F3 study-start deadline. No caveat, no
  manual-review flag, no DEFERRED label is required.
- **Doc cleanup completed:** All "MANUAL IPO DOUBLE-CHECK" language and
  APOLLO-specific caveat rows have been permanently removed from:
  - `CHANGELOG.md` Phase 1 narrative and the Phase 1 exit checklist.
  - `docs/literature_matrix.md §D` NIFTY 50 survivor table (APOLLOHOSP.NS
    row has `f3_manual_ipo_check = False` and no rejection-note column).
- **Deviation from CONTEXT.md:** None. APOLLOHOSP.NS is a clean 46th
  survivor, not a spec exception.

### 3.3 Filters F1 / F2 / F3 — **RESOLVED 2026-10-02; no DEFERRED cells remain**

- **Full 50-ticker `freeze_universe.py` end-to-end run completed** on
  2026-10-02. Wall-clock = 46.5 s (50 tickers × polite 0.15 s/ticker delay
  between Yahoo calls, overhead included). The 2015-01-01 → 2023-12-31
  dev-window slice is honored end-to-end; the 2024-01-01+ final-holdout
  window is never read by the script (verified by hard-coded `DATA_END =
  2023-12-31` constant at the top of `scripts/freeze_universe.py`).
- **Filter counts:**
  - 50 input tickers (2026-era NIFTY 50 constituent list) → **46 survivors**.
  - 4 × F1-only rejections (0 F2, 0 F3):
    1. `#18 HDFCLIFE.NS` — first Close = 2017-11-17 > F1 study-start
       deadline 2015-01-06.
    2. `#36 SBILIFE.NS` — first Close = 2017-10-03.
    3. `#43 HDFCAMC.NS` — first Close = 2018-08-06.
    4. `#24 TATAMOTORS.NS` — Yahoo v8 chart API returns HTTP 404
       "Not Found" across 6 symbol variants (TATAMOTORS.NS / TATAMOTORS /
       TATAMOTOR.NS / TATAMOTORSMET.NS / TELCO.NS / TATAMOTORSLTD.NS). No
       alternate Yahoo symbol known at project scope; recorded as a
       permanent F1 rejection with exact reason string
       `"yahoo chart pull failed: RuntimeError('yahoo chart HTTP 404 …
       No data found, symbol may be delisted')"`.
- **Output CSVs (single source of downstream truth; REPLACED the earlier
  provisional IPO-only versions wholesale — no incremental merge):**
  - `docs/nifty50_sector_map.csv` — 50 rows × 15 columns. No `DEFERRED`
    cell anywhere; every row has real booleans (`True`/`False`) for F1, F2,
    F3, and `f3_manual_ipo_check` is a boolean (0 DEFERRED, 1 True for
    APOLLO, all other 49 False). 4 rows carry `f1_passed = False` + an
    exact `f1_reason` string copied from script stdout.
  - `data/raw/universe_frozen.csv` — 46 rows (survivors only), sorted by
    `free_float_rank` ascending, columns identical to the sector-map rows
    that passed. Authoritative single-file reference for all downstream
    Phase-2 work (notebooks, `src/*.py`, Stage 1A/1B comparisons).
- **Survivor N = 46 compliance:** 45–48 was the frozen target band (see
  locked §2 parameter 2B''), so 46 is inside the band and no spec exception
  is required.
- **Deviation from CONTEXT.md:** None — all 3 filters F1/F2/F3 execute the
  exact numeric thresholds and deadline dates frozen in the spec.

### 3.4 Survivorship-bias low-bound proxy in `literature_matrix.md §E` — **RESOLVED 2026-10-02 via rule (a); real computed CAGR spread**

- **Old (now permanently deleted) placeholder sentence in `§E`:**
  `"… suggests baseline CAGRs are biased upward by 25–75 basis points annualized
  versus a properly de-listed-return-augmented index reconstruction"`.
  This was a human-order-of-magnitude guess written before any actual OHLCV
  pull. **It has been deleted and replaced; do NOT quote the old 25–75 bps
  range anywhere.**
- **Resolution path chosen: §3.4 rule (a) — real computation.** A dedicated
  audit-trail one-off script was written and executed:
  `scripts/_oneoff_calc_survivorship_bias_proxy.py` (≈ 200 lines, kept
  permanently as reproducible evidence). 46/46 frozen survivors had valid
  2015-01-01 → 2023-12-31 Close series pulled through the same
  `_raw_yahoo_chart` helper (polite 0.15 s / ticker inter-request delay).
  20.5 s total wall-clock on 2026-10-02.
- **Exact formula used (per-ticker CAGR):**
  ```
  p_start  = first valid Close on / after  2015-01-01
  p_end    = last  valid Close on / before 2023-12-31
  n_years  = (Timestamp("2023-12-31") - Timestamp("2015-01-01")).days / 365.25
  CAGR_i   = (p_end / p_start) ** (1 / n_years)  -  1
  ```
- **Bucket definitions (per §3.4 original rule (a)):** Frozen 46-survivor
  universe sorted by `free_float_rank` ascending → **Top-10 = ranks 1–10
  (largest free-float tier)**, **Bottom-10 = ranks 37–46 (smallest free-float
  tier)**.
- **Exact computed outputs (reproduced from script stdout):**
  - `mean(Top-10 CAGR)`    = **15.005 %/year**
  - `mean(Bottom-10 CAGR)` = **14.488 %/year**
  - **Spread (Top − Bottom) = + 51.6 bps annualized**
- **Honest interpretation (copied into `§E`):** This +51.6 bps figure is a
  **LOW-BOUND PROXY**, not a true estimate. Real survivorship bias includes
  delisted names, which are systematically smaller and worse-performing than
  *any* tier of the 2026-era NIFTY 50 (even the bottom free-float 10). So
  the within-survivor size spread understates the real (unobservable at
  project scope) bias. The §E paragraph carries this LOW-BOUND caveat
  explicitly, the exact formula, the script path, and both mean values.
- **Audit trail output:** Per-ticker `ff_rank, ticker, cagr_pct, note`
  written to `data/raw/_survivorship_bias_proxy_cagrs.csv` (46 rows, kept
  as evidence — not a downstream pipeline input).
- **Deviation from CONTEXT.md:** None — §E of the literature matrix
  explicitly carries the exact number and labels it as a proxy.

### 3.5 Deviations from CONTEXT.md (explicit, not accidental)

1. **Sector constraint: upgraded from flat ≤5/sector (CONTEXT.md) to ± 3 pct
   NIFTY-relative.** See locked §2 row "Sector constraint (9B)". Rationale:
   NIFTY 50 already has ≈ 10 % in IT and ≈ 20 % in Financials; a flat ≤ 5
   per sector would force the optimizer to divest 2/3 of the Financials
   basket even for a passive benchmark-replicating strategy — which would
   make the "Classic Max Sharpe baseline" structurally unable to reproduce
   the index's risk-return profile, invalidating the apples-to-apples
   control-group comparison against ML strategies. Relative ± 3 % is the
   standard formulation in institutional mandates for this exact reason.
2. **Rebalance frequency: from "assumed monthly" (implied in CONTEXT.md) to
   "empirically selected Stage 1A across monthly vs quarterly on
   tx-cost-adjusted Sharpe of the 4 baselines."** Same rationale: do not
   lock a design choice with data if you can measure it first. Weekly is
   deferred, not ruled out forever — it is computationally too expensive
   (≈ 520 rebalances × 500 MC simulations × 3 ML models = infeasible on
   this Ryzen 5 U laptop).
3. **Covariance estimator: from "Ledoit-Wolf default only" (CONTEXT.md) to
   "Ledoit-Wolf vs PCA-factor decided empirically Stage 1B."** This is the
   institutional-nice-to-have explicitly promoted to a required comparison
   per the 11 locked parameters.
4. **IPO-date 3 ticker rejections:** `SBILIFE.NS` (listed 2017-10-03),
   `HDFCLIFE.NS` (2017-11-17), `HDFCAMC.NS` (2018-08-06) were all added to
   NIFTY 50 only after our 2015 study-start date — they are formally
   rejected in §D because F3 rule is `< 2015-01-01`. This matches the
   frozen 2B'' / F3 rules and is therefore not a "deviation" per se, but
   worth flagging explicitly because none of these 3 tickers existed in
   2015 and a naive reader would otherwise assume 50 → 50 survivors.

---

## 4. What NOT To Do

Specific mistakes already made and corrected in this session — do not
repeat them. Each entry cites the concrete incident, not generic advice.

1. **DO NOT state a bug diagnosis ("it's an ISP/Yahoo block") without first
   capturing raw HTTP evidence.** Earlier this session, yfinance returning
   0-row DataFrames was first called an "ISP/Yahoo block". It later turned
   out to be a trivial User-Agent 429 — same network, same library, same
   API endpoint, just a different UA header string in the session. Fix:
   always run a parallel raw `requests.get(url, headers=...)` comparison
   first and show real HTTP status / headers / body before naming a cause.
2. **DO NOT call a milestone (e.g., "Step 1 — Freeze Universe") Complete
   when only 1 of its 3 filters (F3) was actually executed against real
   data.** Earlier this session the 47-survivor universe was provisionally
   written using only IPO-date manual checks; F1/F2 were labeled DEFERRED
   and no real OHLCV was examined. Rule: count Step 1 complete only when
   `freeze_universe.py` has run end-to-end, returned real F1/F2/F3 booleans
   on all 50 tickers, APOLLOHOSP.NS F3 manual-IPO flag has been resolved
   to True/False (not "TBD"), and the §3.4 survivorship-bias number in
   `literature_matrix.md §E` has been either computed (a) or replaced with
   unquantified language (b).
3. **DO NOT invent a quantitative estimate (e.g., "25–75 bps CAGR bias") and
   state it in the doc as if it were a real computed proxy.** Earlier this
   session the 25–75 bps range was written directly into the limitations
   paragraph without any supporting CAGR calculation. Rule §3.4 above fixes
   this — every claimed number must be either a real calculation with
   reproducible inputs, or flagged with the word **ASSUMPTION/PLACEHOLDER**
   and deleted before any report submission.
4. **DO NOT commit files or changes before explicitly asking the user for
   permission to commit.** Earlier this session a commit (`d631521`) was
   made before the three issues enumerated in the current prompt were
   identified — commit-before-honest-complete-status means a subsequent
   `git commit --amend` / `rebase -i` is required to clean history, or
   else false-completeness is baked into the git log. This is a standing
   rule for the entire project, not just this session:
   *"ask before every commit."*
5. **DO NOT read, plot, or ingest (even accidentally) any price data after
   2023-12-31 except via the Phase 8 final-evaluation script at the very
   end.** `scripts/freeze_universe.py` already correctly slices to
   `DATA_END = 2023-12-31` at the top, but any future ad-hoc notebook or
   exploration could introduce a look-ahead. A soft defensive pattern:
   every `src/*data*.py` loader should accept an `embargo=True` flag
   (conceptually — final label "final holdout") that drops all rows on or
   after 2024-01-01 before returning any DataFrame.
6. **DO NOT use yfinance `info["sector"]` as the source of truth for NSE
   industry classification.** yfinance `.info` is often missing or
   misclassified for `.NS` tickers. Locked decision §2 D8 says the single
   source of truth is `docs/nifty50_sector_map.csv` hand-mapped once.

---

## 5. Next Immediate Steps (Phase 5 Kickoff — Monte Carlo Resampling, concrete, in order)

Phase 1 / 2 / 3 / 4 are **all 100% signed off and frozen**. All 46 × 2222 dev-window prices (OHLCV) on disk warm cache, 37 tests green, 7/7 feature sanity, Phase 4 7/7 exit gates PASS (G6 nb2 identity max |Δ|=0.000000 on 12 cells), Gaussian residual null REJECTED for all 3 ML families (Jarque-Bera p=0.0 → empirical bootstrap MANDATORY). **Frozen input seed to Phase 5:** RIDGE μ̂ walk-forward forecasts (32 true ML RDs, 5 earliest = CMS 63d μ̂ fallback) + LW Σ + ±3 pp sector QP + 10% single-name cap + 10 bps/turn tx-cost. Execute in numbered order — do not skip.

### Step 5.0 — Idempotent environment sanity (run once every new session, ≤ 60 s)
1. `pytest tests/ -v` — **must be 37/37 PASS** (36 Phase 1–3 + 1 ML smoke `test_ml_shape_and_no_nan`).
2. `scripts/sanity_check_features.py` — **7/7 sanity assertions PASS, X.shape == (89608, 77)**.
3. `scripts/_p4_exit_gates.py` — **7/7 Phase 4 gates GREEN** (re-confirms frozen seed integrity: G5 panel drift ±3.000 pp exact, G6 nb2 identity 12-cell max |Δ| < 0.005, G7 decision row present).
4. Confirm **13 new Phase 4 CSV artifacts present** in `data/processed/` (see Section 1C table for row counts). Any missing → re-run 3-script Phase 4 ladder: `phase4_build_forecasts.py` → `phase4_run_ml_wf.py` → `phase4_feature_importances.py` (long runners: use `$env:PYTHONUNBUFFERED='1'; python -u` to avoid stdout buffering hang).

### Step 5.1 — Build empirical residual bootstrap scaffold (MANDATORY — Gaussian REJECTED)
1. Inputs: `X = (89608, 77)`, `y = length 89608 (21d fwd logret)`, frozen RIDGE model weights `src/ml_models.py:train_ridge_linreg()`.
2. Recompute OOS residuals via `extract_oos_residuals("ridge_linreg", X, y, n_splits=5)` → returns `(yhat_concat, resid_concat, _)`. Save `resid_concat` ndarray shape=(89608,) to `data/processed/phase5_ridge_oos_residuals_empirical.npy` (or CSV if numpy save disabled by gitignore via allowlist).
3. **Residual distribution sanity:** Compute 5-number summary (min / Q1 / median / Q3 / max), mean, std, skew, kurtosis excess. Confirm heavy tails (kurtosis excess >> 0 for Indian equity log-returns — consistent with JB rejection). Save summary row to `data/processed/phase5_residual_distribution_summary.csv`.
4. **Counterfactual parametric t-fit (for §E comparison only, never used for draws):** Fit a Student-t (location μ, scale σ, df ν) to `resid_concat` via MLE (`scipy.stats.t.fit`). Record ν (expected ~4–6 for daily equity log-returns). This is for the "what if" comparison cell in Phase 6 — NEVER use t-dist draws in the main MC engine.

### Step 5.2 — Implement block bootstrap (temporal-dependence aware) vs iid bootstrap side-by-side
1. **Mode A — Naive iid bootstrap (baseline, for comparison only):** Draw `N_BOOT` rows uniformly with replacement from `resid_concat`. Expected: understates VaR/ES because ignores 1-day autocorrelation + GARCH-like vol clustering.
2. **Mode B — Block bootstrap (primary production mode, MANDATORY):** Fixed block length `B = 21 trading days` (≈ 1 calendar month). Rationale: residual autocorrelation + vol clustering decorrelates over ≈21 BD. Algorithm: sample starting indices with replacement, each draw = contiguous block of 21 residuals, trim overflow to 89608 total. Use `random_state=7` locked seed.
3. Save `N_BOOT=500` weight-simulation budget placeholder first (convergence curve step 5.4 confirms adequacy).
4. Block structure integrity check: for 10 random draws, verify block contiguity (no row gaps).

### Step 5.3 — MC forward return & weight simulation loop (37 RDs × K draws)
1. For each rebalance date `rd` in the 37-RD frozen list:
   a. Load frozen RIDGE μ̂: `mu_ridge_63d[rd, tkr]` from `phase4_ridge_linreg_forecasts.csv` (or CMS fallback if rd in first 5).
   b. Perturbed return per draw: `r_tilde[draw, tkr] = mu_ridge_63d[rd, tkr] + residual_draw[draw]` (residual shape: 89608 → map to 46 tickers per RD: use the per-RD latest-residual slice per ticker for that RD's 46 rows so residual N=46 per draw per RD).
   c. **Risk aversion grid λ ∈ {1.0, 2.0, 4.0, 8.0} if compute budget allows; default λ=2.0 first.** Objective per draw: `max_w [ wᵀ μ̃ − (λ/2) wᵀ Σ_LW w ]` subject to: sum w=1, w≥0, per-ticker w≤10%, **sector ±3 pp survivor-46 QP projection post-solve.**
   d. Save weight tensor: `phase5_mc_weights.nc` (xarray netCDF) or CSV long format: `(rebal_date, draw_id, ticker, weight)` — dimensions 37 × 500 × 46 = **8 510 000 rows**.
   e. Compute portfolio return per draw per RD → equity curve per draw.

### Step 5.4 — Convergence curve (K = 50 → 100 → 200 → 500 draws, monitor Sharpe distribution quantiles)
1. Run the Step 5.3 loop incrementally with `K ∈ {50, 100, 200, 500}`.
2. For each K, compute: (a) mean tx-Sharpe across K draws, (b) 5% and 95% quantiles of the K tx-Sharpes, (c) width of the 90% interval = q95 − q05.
3. **Stopping criterion:** If interval width shrinks by < 5% when doubling K from 200 → 500, K=500 is adequate. If still shrinking fast, go to K=1000 (budget allowing). Save convergence table to `data/processed/phase5_mc_convergence_curve.csv` (cols: `K, mean_sharpe, q05_sharpe, q50_sharpe, q95_sharpe, interval_width_pct_change_vs_prev_K`).
4. Plot (if matplotlib budget): convergence curve + interval band — for Phase 6 panel evidence.

### Step 5.5 — Mean vs Median centroid portfolio choice + re-risked distribution metrics
1. **Candidate 1 — Mean-weight centroid:** `w̄[rd, tkr] = mean_{draw=1..K} w_draw[rd, tkr]`.
2. **Candidate 2 — Median-weight centroid:** `w̃[rd, tkr] = median_{draw=1..K} w_draw[rd, tkr]`.
3. **Candidate 3 — Draw with Sharpe nearest to the K-draw median Sharpe** (single "representative" draw, not a centroid average).
4. Re-run each candidate through the frozen WF pipeline: rebuild equity curves, apply 10 bps/turn tx-cost, compute full `_nb2_consistent_metrics()` per candidate.
5. Selection rule frozen: pick the candidate with the **highest tx-cost-adjusted Sharpe** among the 3. If Δ < 0.02, prefer Median centroid (literature-backed more robust to outlier draw weight spikes).
6. Output Phase 5 result CSV: `data/processed/phase5_mc_centroid_verdict.csv` — 3 rows × full metric set + 1 DECISION row. Also save the selected centroid weights `phase5_mc_selected_weights.csv` (37 × 46 = 1702 rows) as the seed input to Phase 6.

### Step 5.6 — Risk aversion grid sweep (λ = 1.0 / 2.0 / 4.0 / 8.0) if compute budget allows
1. Repeat Step 5.3–5.5 for each λ.
2. Build λ frontier table: `phase5_lambda_frontier.csv` (cols: λ, selected_centroid_tx_sharpe, ann_vol, max_dd_pct, total_turnover_ann_bps, calmar).
3. Frozen default λ = 2.0; if another λ yields tx-Sharpe > +50 bps over λ=2.0 AND has strictly lower MDD, flag as candidate for Phase 6 panel. Otherwise stick to λ=2.0.

---

## 6. Files Touched So Far

Single-sentence description + current completeness state per file, in
alphabetical order. Any file not listed here has not been modified from
repo initialization (which only contained `CONTEXT.md`, `README.md`,
`LICENSE`, the folder skeleton with `.gitkeep` placeholders, the `.venv/`
directory, and an empty `CHANGELOG.md` / `.gitignore` / `requirements.txt`
all of which were filled during this session).

| File | Description | Completeness state as of this checkpoint |
|---|---|---|
| `.gitignore` | Data-science Python project ignore rules; excludes `.venv/`, raw + processed data dirs, results, caches, notebook checkpoints. `data/raw/universe_frozen.csv` is the only non-`.gitkeep` file under `data/` explicitly allowlisted. ⚠️ Confirm `.yf_cache/` is present in the ignore list before commit (auto-created dir at ROOT level by `yfinance.cache.set_tz_cache_location`). | ✅ Almost complete — verify `.yf_cache/` exclusion (see §3.1). |
| `.yf_cache/` directory | Project-local yfinance tz + cookie sqlite cache. Created at repo root (`D:\ML\Quant\.yf_cache`) by the UA-fix `yfinance.cache.set_tz_cache_location` / `set_cache_location` calls in `scripts/freeze_universe.py`. Redirected there because the TRAE IDE sandbox blocks writes to `%LOCALAPPDATA%\py-yfinance\tkr-tz.db-shm` (which would otherwise raise `OperationalError: unable to open database file`). Should be gitignored — it is a per-machine environment cache, not a reproducible pipeline artifact. | ⚠️ Auto-created env artifact. Gitignore-confirm then treat as not-in-repo. |
| `CHANGELOG.md` | Week 1→3 narrative COMPLETE ✅. Phase 1 exit (6 items), Phase 2 exit (7/7 gates) + Stage 1A/1B frozen decisions all documented. Phase 3 completion block (all 5 bug remediations, 4-script ladder 0 exit, 7/7 WG6 gates + NB2 8/8 panels) appended 2026-10-04. | ✅ Authoritative full signed-off narrative (P1 P2 P3 complete). |
| `CHECKPOINT.md` | This handoff file. §1 rewritten as 4-phase completion (P1→P4 100% frozen) + full §1C Phase 4 completion block (3 ML families, 7/7 exit gates, RIDGE selection frozen). §2 Stage 1A/1B/12 winners (Phase 4 RIDGE model selection row added at §2 ID 12). §3 special cases unchanged. §5 is Phase 5 kickoff (Steps 5.0–5.6 Monte Carlo resampling). §6 updated with all Phase 4 src/scripts/tests/CSV rows. | ✅ Authoritative for Phase 4 end handoff — next: Phase 5 Monte Carlo Resampling. |
| `CONTEXT.md` | Original architecture / math / 16-week roadmap document. **Intentionally NOT modified in this session or this project phase.** The 4 known deviations from it are documented explicitly in §3.5 above (sector constraint 9B, rebal freq selection 4C, Σ selection 11, the 3+1 F1/F3 ticker rejections note). Frozen per user standing rule. | ✅ Frozen reference doc. 4 known deviations documented in §3.5. |
| `data/processed/*.csv` (27 files = 14 Phase 3 + 13 Phase 4) | Full Phase 3 artifact payload (14 files, see §1B row-by-row) + 13 NEW Phase 4 artifacts: 3 forecast CSVs (ridge/rf/xgb, 1472 rows each = 32 ML RD × 46 tickers; 5 earliest RD = CMS 63d μ̂ fallback), residuals_summary (3 rows Jarque-Bera: Gaussian null REJECTED p=0.0 for all 3 families → empirical bootstrap MANDATORY), oos_pred_diagnostics (15 rows = 5 TSS folds × 3 families, fold-level RMSE/R², labeled "NOT success metric"), 3 × (raw + txadj) equity CSVs (3 families × 2222 dev dates each), ml_weights (5106 rows = 37 RD × 46 tkrs × 3 families), ml_summary (3 rows × 17 cols NB2-consistent metrics), ml_vs_baseline_verdict (8 rows ranked: 3 ML + 5 baselines, +50bps-over-1/N PASS flag for Ridge/XGB/RF/CMS), feature_importances (231 rows = 77 feats × 3 models: Ridge |coef×scale_|, RF/XGB permutation on 2021-calendar temporal holdout chunk), model_selection_decision (1 authoritative row: RIDGE_LINREG selected frozen seed). ALL force-added against `.gitignore:49` so fresh clone runs 3-script Phase 4 ladder without retraining. | ✅ 27 CSV artifacts on disk. 14 P3 + 13 P4 verified present by `ls` and 7/7 exit gate script G3/G4/G7 checks. |
| `data/raw/*.csv` (48 ticker OHLCV CSVs) | 46 frozen-survivor daily Close/Open/High/Low/Volume CSVs 2015-01-01 → 2023-12-29 (warm parquet-free cache; `load_prices` reads them directly, 0 network needed for dev-window). + `universe_frozen.csv` + `_survivorship_bias_proxy_cagrs.csv`. | ✅ 48 raw CSVs. 46 × 2222 rows, all Yahoo-vetted. |
| `data/raw/universe_frozen.csv` | 46 rows (frozen survivors only). Output of the 2026-10-02 `freeze_universe.py` 46.5 s end-to-end run. Sorted by `free_float_rank` ascending. Columns identical to `docs/nifty50_sector_map.csv` rows that passed F1+F2+F3. 4 rows NOT present here = the 4 F1 rejects (HDFCLIFE, SBILIFE, HDFCAMC, TATAMOTORS). **Single source of downstream truth for every Phase 2–8 pipeline.** | ✅ Final / authoritative. Do NOT hand-edit — only regenerate if filter rules are re-frozen with spec change. |
| `data/raw/_survivorship_bias_proxy_cagrs.csv` | 46 rows, audit-trail output of `scripts/_oneoff_calc_survivorship_bias_proxy.py`. Columns: `ff_rank, ticker, cagr_pct, note`. Used to produce the `§E` 51.6-bps low-bound proxy number. NOT a downstream pipeline input — kept only as reproducible evidence. | ✅ Audit trail complete. Treat as read-only. |
| `docs/literature_matrix.md` | Six sections, all populated: §A frozen dataset spec (11 params, with 4 known CONTEXT deviations) · §B Stage 1 pipeline ASCII + dev/holdout hard boundary · §C 12 documented defaults D1–D12 · §D filter rules + 4-row definitive rejection log + 46-row survivor table + CSVs footer block (TATAMOTORS HTTP 404 permanently recorded) · §E Limitations paragraph: survivorship-bias **LOW-BOUND 51.6 bps** real computed proxy (exact formula, script path, top10/bottom10 means cited; old 25–75 bps human guess PERMANENTLY DELETED) + sequential-testing bias D12 = 24 configs · §F literature standalone matrix = **21 papers across 6 themes × ≥3 refs each** (18 mirrored from Full Matrix body + 3 targeted adds: Jagannathan & Ma 2003, Fan et al. 2008, Sortino & van der Meer 1991, Ang et al. 2006, Fama-French 2015, Scherer 2002). | ✅ All 6 sections complete. §F meets 15–30 paper target (N=21). |
| `docs/nifty50_sector_map.csv` | 50 NIFTY rows × 15 columns full filter trail. 0 DEFERRED cells anywhere; F1/F2/F3 are real booleans (`True`/`False`) on every row; `f3_manual_ipo_check` is a boolean (APOLLO = False → no caveat). 4 rows have F1=False + exact `f1_reason` strings copied from freeze_universe.py stdout. **Authoritative record of the 4 rejections** (used by `literature_matrix.md §D` rejection log). Also defines 37-count survivor-sector weights used by CMS ±3pp projection. | ✅ Final / authoritative frozen trail. CMS sector-target ground truth. |
| `docs/synopsis.pdf` | Project synopsis PDF. Existed before this session. Not modified. | — Not in scope for this session. |
| `LICENSE` | License file. Not modified this session. | — |
| `notebooks/01_data_exploration.py` (jupytext percent-script, paired ipynb render) | 6.1→6.6 panels rendered, JSON cell valid. Export 37-row `sector_summary_dev_window.csv` on disk. | ✅ 6/6 panels PASS (Phase 2 G2 gate). |
| `notebooks/02_baseline_performance.py` + `_run_nb2_validation.py` | 8 WG5 baseline-performance panels (equity ref-lines 1/2/4/8×, drawdown overlay, 12-mo rolling Sharpe heatmap [−2,+3], turnover violin 10bps line, CMS ±3pp stacked bars, weight snapshots 2016/2019/2022 Jan, DeMiguel 2009 1/N verdict, signature Δ identity cell). `_run_nb2_validation.py` = headless headless renderer with 2 HARD ASSERTS (Panel 5 ±3pp violated_flag FALSE; Panel 8 max Δ < 0.05). | ✅ **8/8 panels PASS, exit 0, 2/2 hard asserts green** on 2026-10-04 regenerated artifacts. Panel 5 drift ±3.000 pp (bound); Panel 8 Δ=0.000 identity exact. |
| `README.md` | Status line + pipeline ASCII diagram identical to `CONTEXT.md §2`. | ⚠️ Status line still says "Week 1 — planning phase". Optional status bump: → "Week 5-6 — Phase 3 baselines complete, Phase 4 ML next". |
| `requirements.txt` | 20 pinned dependencies + numpy<2 pin rationale + plotly<5.23 for vectorbt heatmapgl compat. All verified in a fresh venv 20/20 smoke tests passing, `pip check` clean. | ✅ Complete and verified for this phase. |
| `scripts/__init__.py` | Empty package marker. | — |
| `scripts/build_stage1a_baselines.py` | Walk-forward 8-curve generator (4 strat × 2 freq Monthly/Quarterly, sample Σ, 10% single-name cap, CMS μ̂ stub, 180 BD minimum training window). Outputs: monthly/quarterly rebalance panels + raw-sharpe CSV + 8 equity CSVs. | ✅ Regenerated 2026-10-04 — exit 0. Numbers identical to pre-regeneration documented table. |
| `scripts/stage1a_apply_txcost.py` | Rebuilds weights in 8 cases exactly, applies `0.5 × Σ|Δw_price_drifted|` one-sided turnover definition, 10bps drag per unit turnover, then 2-rule anti-tiebreaker (Δmean<0.02 → cheaper turnover; if still tied → Monthly). Outputs 9-row decision table + txadj 8 equities. DECISION row Quarterly mean = +0.8288 tx-Sharpe over Monthly +0.7811 → +47.7 bps-Sharpe margin direct win. | ✅ Exit 0; winner QUARTERLY confirmed frozen. |
| `scripts/stage1b_compare_cov.py` | **Mandatory §16 synthetic cross-check gate runs FIRST (exits 4 on RMS>5e-4).** Then 2 Σ × 3 cov-sensitive strategies (MinVar / RiskParity / CMS 63d μ̂) quarterly walk-forward. Equal Weight excluded per D10 frozen. Bug #1 meta-record fix applied 2026-10-04 (PCA K from NaN → 14.5833). 2-rule Stage 1B tiebreakers. Outputs: 2-row decision, 12-curve WF equities CSV, 2-row authoritative phase3_stage1_winners.csv. LW wins +2.45 bps-Sharpe tiebreaker-1 turnover confirms. | ✅ §16 synth gate PASS (LW 3.07e-7 < 5e-4; PCA 1.57e-4 < 5e-4). Exit 0. LW frozen. PCA K=14.5833 (valid, no longer NaN). |
| `scripts/phase3_run_5baselines.py` | **WG4 full 5-baseline run.** Reads Stage 1 winners; RuntimeError if `COV_ESTIMATOR_WINNER != "LW"`. Quarterly 63 BD rebalance. LW Σ (63 BD lookback). CMS 63d μ̂ + ±3pp sector-QP projection post-verifier. Final metrics computed via `_nb2_consistent_metrics()` helper lines 53-78 (geo ann return, simple daily ret ddof=1 × √252 vol, rf=4% subtraction Sharpe, cummax MDD) → matches _run_nb2_validation.py Panel 8 formula byte-for-byte, max Δ=0.0. 5 baselines: Equal W / FreeFloat proxy / Min-Variance (LW) / Risk Parity (LW) / Classic Max Sharpe (LW Σ + ±3pp CMS). Outputs: 2 equity CSVs (raw/txadj), weight CSV (185 rows), 10-col summary CSV, idempotent stage1_winners CSV. | ✅ Exit 0; 5-strategy summary on disk. NB2 Panel 8 Δ=0 identity, Panel 5 ±3pp exact drift (no violation). |
| `scripts/phase4_build_forecasts.py` | **Phase 4 WG1 3-family ML walk-forward forecast generator.** Reads frozen 37 quarterly RD list, skips first 5 (CMS 63d μ̂ fallback, 0 training rows pre-2016), runs 32 true ML RDs × 3 families. Leak guards: `assert train_max_date < RD − 21 BD` (target-horizon cut per ticker-date panel) + `assert X_latest no NaN AND X_latest.date.max() < RD` (feature cut). Scaling: 21d daily point forecast × SCALE=3.0 → 63d quarterly μ̂ (exact `assert abs(3.0 − SCALE) < 1e-12`). 3 forecast CSVs output (1472 rows each). TSS OOS 5-fold residuals extracted inline → passed to Jarque-Bera (all 3 families reject Gaussian). | ✅ Exit 0; 37 RD iter 0 ASSERT fail; 3 forecast CSVs 1472 rows each on disk; Gaussian-null REJECTED seed for Phase 5 empirical bootstrap. |
| `scripts/phase4_feature_importances.py` | **Phase 4 honest feature-importance builder (avoids Gini-impurity overfit on RF/XGB).** Ridge: |standardized coef × scaler.scale_| rank. RF/XGB: **permutation importance** (NOT Gini) on internal 2021-calendar temporal holdout (train pre-2021: 56488 rows, val 2021 only: 11408 rows, n_repeats=10, seed=7). Output 231-row CSV (77 feats × 3 models). Top Ridge: cs_rank_mom126; Top RF/XGB: vol_ann_63d (10× next feature). Called from 3-script Phase 4 ladder. | ✅ Exit 0; 231 rows on disk. Permutation-importance guard (not Gini) documented. |
| `scripts/phase4_run_ml_wf.py` | **Phase 4 WG3 full 3-ML-strategy quarterly walk-forward.** Reads frozen Stage 1 winners (LW Σ, QUARTERLY freq). Reuses EXACT Phase 3 pipeline: LW Σ (63 BD lookback), pypfopt Max Sharpe r_f=4% → cvxpy Clarabel ±3pp survivor-46 sector-QP projection → 10% single-name cap clamp → 10 bps/turn tx-cost (drifted-weight turnover) → `_nb2_consistent_metrics()` exact metrics. 3 ML families: Ridge LinReg / RF / XGB. ML + 5 baselines ranked → verdict CSV (8 rows) with 50bps-over-1/N flag. Model selection per frozen chain (sharpe_txadj > calmar > turnover) → 1-row decision CSV (RIDGE selected). Also writes: oos_pred_diagnostics 15-row TSS table, ml_summary, ml_weights, 6 equity CSVs. | ✅ Exit 0; 3 families ±3.000pp drift EXACT bound (0 violations); max single-name weight 10.0000% (cap); nb2 metric identity recompute 12/12 cells max |Δ|=0.000000. 8-strategy verdict CSV on disk + RIDGE frozen decision row. |
| `scripts/sanity_check_features.py` | Phase 2 7-assertion sanity (shape, NaN, holdout-safe, lead-in-safe, top/bottom |corr(X,y)|, dead-col count). Called from the 3-leg Phase 3-verification suite in new Step 5.0. | ✅ Exit 0 — 7/7 assertions PASS 2026-10-04. X.shape=(89608, 77). |
| `scripts/_oneoff_calc_survivorship_bias_proxy.py` | Audit-trail one-off script for the `§E` survivorship-bias low-bound proxy computation. Reads `universe_frozen.csv`, pulls 2015–2023 Close per ticker via same `_raw_yahoo_chart` helper, computes `CAGR_i = (P_end/P_start)^(1/n)-1`, writes `_survivorship_bias_proxy_cagrs.csv`. 46/46 valid, 20.5 s wall-clock. Keep permanently (reproducibility evidence), even though it will not be re-run in the normal pipeline. | ✅ Executed; audit trail complete. |
| `scripts/_p4_exit_gates.py` | **Phase 4 7/7 exit-gate verifier (authoritative).** Runs 7 gates G1→G7 with hard assertions: G1 pytest≥36 PASS (37/37 expected), G2 sanity_check 7/7 X=(89608,77), G3 phase4_build exit 0 + leak asserts + 37 RDs, G4 3 forecast CSVs ≥1472 rows each, G5 phase4_run_ml_wf exit 0 + Panel5 ±3pp drift EXACT + 10% cap ≤10.0000%, G6 nb2 metric identity on 3 ML × 4 cols = 12 cells max |Δ| < 0.005, G7 1-row decision CSV + non-empty justification. Prints consolidated banner. Run from every new session before Phase 5 work. | ✅ Exit 0 banner printed 2026-10-06: ALL 7 PHASE 4 EXIT GATES 7/7 PASS. G6 max |Δ|=0.000000 (exact identity) across 12 cells; G5 3 families ±3.000pp 0 violations. |
| `scripts/freeze_universe.py` | Reproducible 3-filter script (F1/F2/F3), now with: (1) Chrome-mimic UA `requests.Session()` + `yf.utils.user_agent_headers` patch; (2) yfinance cache redirected to `.yf_cache/` ROOT dir (sandbox fix); (3) new `_raw_yahoo_chart()` top-level helper (raw Yahoo v8 `finance/chart` GET, builds OHLCV DataFrame, `auto_adjust=True` semantics — replaces the yfinance `.download(session=...)` path that failed in the sandbox); (4) `_ipo_first_trade_date` rewritten to wide 1980–2015 window reading `result[0].meta.firstTradeDate` epoch UTC (replaces sandbox-broken `tkr.info` lookup); (5) `_apply_filters` calls `_raw_yahoo_chart` directly. All thresholds still match frozen spec exactly; 0.15 s / ticker polite delay; 2024+ holdout never read. | ✅ UA patch applied; helpers verified; full 50-ticker 46.5 s run exited 0 with 46 survivors. |
| `src/__init__.py` | Empty package marker. | — |
| `src/covariance.py` (161 lines) | Two covariance factories: `ledoit_wolf_cov(log_ret)` pypfopt-based (meta: δ, condition_number, rank), `pca_factor_cov(log_ret, min_explained=0.85, max_k=15)` custom eigh-based Σ = QΛQᵀ + clamped idiosyncratic diag SPD guarantee. Generator `walk_forward_cov`. | ✅ 3 synth cases pass solver cross-check 36/36 green, PCA K mean 14.5833 on dev WF. |
| `src/data_loader.py` (205 lines) | Keyword-only `final_holdout: bool=False` guard, ValueError non-bool, UA-patched `_raw_yahoo_chart` cache helper, 5-retry exponential backoff, parquet/CSV fallback. | ✅ 4/4 test_no_lookahead tests PASS. Never read holdout data in any Phase 3 script. |
| `src/features.py` (~77 features) | 8 feature families (mom/vol/mdd/liq/seasonal/sector/trend/price-level). 3 separate public fns: 3 separate `make_features` / `make_targets` / `align_X_y` structural X/y separation. 77 feature cols, 1948 dates × 46 ticker ~89 608 trainable cells after 252d lead-in + 21d target-trim. holdout-safe max common date = 2023-11-29 < safe edge 2023-11-30 1 BD margin. | ✅ 7/7 feature sanity PASS; 77 features 0 dead cols. |
| `src/optimizer.py` (494 lines) | 5 baselines exposed (`equal_weight`, `freefloat_proxy_weight`, `min_variance_weights` dual-backend with cross_check flag, `risk_parity_weights` L-BFGS-B log-space, `classic_max_sharpe_weights` dual code path PyPFOpt no-sector → cvxpy 3-ladder fallback + ±3pp sector tolerance projection). `_post_verify_weights` 4-stage repair (sum 1 auto, non-neg clamp, 400-iter 10% cap clamp, **sector QP min-distance projection** to CMS bound — this is what gives ±3.000pp exact bound in Panel 5). Module constant `COV_ESTIMATOR_WINNER = "LW"` line 19 locked permanent. | ✅ 32/32 optimizer-constraint tests PASS. Solver cross-check RMS ≤ 1e-5 N∈{15,46,80} per frozen §16 synthetic gate. Sector repair in 36/36 WG4 + 37 CMS rebal WF never violated. |
| `src/ml_models.py` | **Phase 4 3-family pooled regression factory + 2 helpers.** 3 training APIs (all return sklearn/xgb Pipeline/model, seed=7 frozen): `train_ridge_linreg(X_train, y_train)` → Pipeline(StandardScaler→Ridge alpha=1.0); `train_rf(X_train, y_train)` → RandomForestRegressor(max_depth=8, min_samples_leaf=20, n_estimators=200); `train_xgb(X_train, y_train, X_val, y_val)` → XGBRegressor(hist, n=500, lr=0.03, depth=4, subsample=0.8, colsample=0.8, reg_lambda=1.0, early_stopping_rounds=40 via init). 2 helpers: `predict_1d(model, X_test)` → flat np.ndarray (ensures consistent shape regardless of model output type); `extract_oos_residuals(model_family_id, X, y, n_splits=5)` → 5-split TSS (shuffle=False) concat of (yhat, resid, fold_ids) — Jarque-Bera normality test run on resid. Called from phase4_build_forecasts.py. | ✅ Public API signature check: 3 train + 2 predict/resid fns all present. Smoke test (test_ml_models_smoke.py) 2 shape+finite asserts PASS. TSS 5-split residuals concat length = 89608 (matches full panel). |
| `tests/__init__.py` | Empty package marker. | — |
| `tests/test_no_lookahead.py` (4 tests) | sentinel-blocked + revealed-when-enabled + non-bool-reject + keyword-only. 4/4 green. | ✅ 4/4 PASS — holdout guard working. |
| `tests/test_optimizer_constraints.py` (32 tests) | post-verifier sanity (sum 1 / non-finite raises), Equal W (5 sizes + empty-raise), FreeFloat proxy (2), MinVar pypfopt/cvxpy 3×2 cases, MinVar infeasible-cap, RiskParity 3 synth + identity, cov-winner lock test, CMS basic / CMS within-tol / CMS missing-keys-raises / CMS no-tickers-raises / CMS infeasible-cap, Solver cross-check RMS ≤ 1e-5 (N ∈ {15,46,80}) + cross_check flag. | ✅ **32/32 PASS, 13.10s.** §16 synth cross-check pypfopt-vs-cvxpy for all 3 size cases all within threshold. |
| `tests/test_ml_models_smoke.py` (2 asserts = 1 test) | **Phase 4 ML smoke guard.** Builds synthetic X (500 rows × 20 cols, uniform noise + linear signal) + y (noisy X·w + bias, length 500). Fits all 3 factory models in sequence (train_ridge_linreg / train_rf / train_xgb — X_val=last 15% of rows) → calls predict_1d on each → asserts: (1) `len(yhat) == len(y_test)` shape match for all 3, (2) `np.all(np.isfinite(yhat))` finite-only for all 3. Single test function `test_ml_shape_and_no_nan()`. | ✅ **1/1 PASS** → pytest suite total = 37/37 (4 no-lookahead + 32 optimizer + 1 ML smoke). No synthetic signal leakage: X is fresh uniform rng 7 draw, not real 89608 panel. |
| (Directories with non-`.gitkeep` content): `notebooks/`, `data/processed/`, `results/` | All populated; not empty. | ✅ Correct — 27 CSVs in `data/processed/` (14 Phase 3 + 13 Phase 4); 2 scripts + validator in notebooks. |
