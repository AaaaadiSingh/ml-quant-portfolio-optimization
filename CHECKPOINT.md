# CHECKPOINT.md

Project handoff file. Updated at the end of every work session. A reader of
this file is assumed to have read `CONTEXT.md` (architecture / math / 16-week
roadmap) but to know **nothing** about the work-session history, prior
diagnoses, or decisions documented in chat rather than in the repo.

---

## 1. Current Status

The project has **completed Week 1 of 16 — Planning & Foundations / Phase 1** (see 16-week
roadmap in `CONTEXT.md §7`) and is ready for the Week 1 → Week 2 (Phase 1 → Phase 2)
handoff. What has actually been delivered versus what remains pending:

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

- **PENDING / BELONGS TO PHASE 2 (no work started yet)**
  - Phase 2 code: `src/` contains only the placeholder `__init__.py`; no
    `data_loader.py`, feature engine, or data pipeline backend has been
    written. A soft recommendation (CHECKPOINT §4.5 item 5) says the very
    first `src/*.py` file should be `src/data_loader.py` with a
    `final_holdout=False` (or `embargo=True`) guard that drops rows ≥
    2024-01-01 before returning any DataFrame, to prevent accidental
    final-holdout ingestion in Phase 2 exploratory notebooks.
  - No *test bodies* have been written yet, only the two `__init__.py`
    markers. The explicit `test_no_lookahead.py` and
    `test_optimizer_constraints.py` skeletons in CONTEXT.md §4.4 Phase 2
    deliverables do not yet exist.
  - Notebooks `01_data_exploration.ipynb` through `07_final_evaluation.ipynb`
    remain empty scaffolds (not created yet; `notebooks/` directory contains
    only `.gitkeep`).

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
| Rebalance frequency selection methodology (4C) | Decided **empirically in Stage 1A before ML enters the pipeline**: compare Monthly (21 d) vs. Quarterly (63 d) across the 4 classical benchmarks only on **transaction-cost-adjusted Sharpe**. Weekly is skipped initially. The same winner frequency is used for all strategies thereafter. | `docs/literature_matrix.md §A, §B` |
| Final holdout / out-of-sample period (5B) | **2024-01-01 → 2025-06-30** (18 months, never inspected before Phase 8). Terminology note: the older word "embargo" is deliberately not used in this repo — the correct, final label is "final holdout" or "out-of-sample". | `docs/literature_matrix.md §A, §B` |
| Data source (6A) | `yfinance` library via Yahoo Finance API, `.NS` suffix. The implementation-level User-Agent patch is §3 business (a workaround), not a spec change — the spec is still "yfinance". If yfinance ever becomes unusable, spec change requires re-freeze in `CHANGELOG.md`. | `docs/literature_matrix.md §A` |
| Feature / forecast cadence (7B) | Daily features engineered, **monthly-retrained** model (≈108 retrains in the dev window, laptop-friendly on Ryzen U), **daily point forecasts** inside each window, forecasts aggregated to the rebalance horizon by **summing daily log-return point forecasts** (log-additivity is mathematically correct). ML target is therefore implicit, not a direct single-monthly return column. | `docs/literature_matrix.md §A, §C` (D4, D5) |
| Universe filters (8') | **Three filters, run in order F1 → F2 → F3**: F1 = data availability (first Close ≤ study-start deadline, NaN frac ≤ 5 % in dev window); F2 = liquidity (illiquid-day frac ≤ 2 % where illiquid = vol=0 OR vol<10k shares); F3 = IPO date (first trade date < 2015-01-01). Market-cap filter explicitly dropped because NIFTY 50 selection already encodes large-cap. | `docs/literature_matrix.md §A`; formal rules + deadlines hard-coded in `scripts/freeze_universe.py` top constants |
| Sector constraint (9B) | Sector-**relative** cap on portfolio weight: **± 3 percentage points** from NIFTY 50's own sector weights at each rebalance date. Applied to the set {Classic Max Sharpe baseline, ML point-estimate strategy, ML + Monte Carlo resampled strategy}. Equal Weight and Min Variance / Risk Parity are excluded because their sector positions are already definitionally balanced or risk-driven. | `docs/literature_matrix.md §A, §C` (D6, D7, D8) |
| Bias handling (10A) | Two biases are **explicitly, publicly acknowledged** in the limitations paragraph §E of the literature matrix: **(a) survivorship bias** because today's NIFTY 50 list is used, not a historical constituent series; **(b) sequential / multiple-selection testing bias** because the project tests (2 freq) × (2 Σ) × (3 ML) × (MC on/off) = **24 implicit configurations** — this is the reason the Phase 6 statistical-correction machinery (Jobson-Korkie pairwise test, Deflated Sharpe Ratio, PBO) is required, not optional. | `docs/literature_matrix.md §A, §E` |
| Covariance estimator selection methodology (11) | Decided **empirically in Stage 1B AFTER frequency is locked, BEFORE ML enters the pipeline**: compare **Ledoit-Wolf shrinkage** vs. **PCA / factor-model covariance** (K factors chosen by ≥ 85 % cross-sectional return variance explained). Winner selected on highest mean annualized Sharpe of {Min Variance, Classic Max Sharpe, Risk Parity} baselines. Same frozen winner is used globally for every strategy thereafter. | `docs/literature_matrix.md §A, §B, §C` (D9, D10, D11) |
| Transaction cost assumption | 10 basis points per unit of turnover (both buy and sell legs), applied uniformly across the 4 baselines in Stage 1A and every later strategy. | `docs/literature_matrix.md §C` (D3) |
| NSE sector classification | Hand-mapped once, frozen, in `docs/nifty50_sector_map.csv` (column `sector_provisional` today; the authoritative D8 mapping is this same file, to be promoted from provisional in Phase 2 once the 47 survivors are settled). `yfinance.info["sector"]` is not used because of known gaps and inconsistencies for `.NS` tickers. | `docs/literature_matrix.md §C` (D8) |
| Sequential-testing configuration tally | 24 implicit configurations = (2 rebal freq) × (2 Σ estimators) × (3 ML families: LinReg / RF / XGBoost) × (2 MC modes: off / on). | `docs/literature_matrix.md §C` (D12); §E |

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

## 5. Next Immediate Steps (Phase 2 Kickoff — concrete, in order)

Phase 1 is signed off as complete. These are the **first actions** a fresh
LLM (or the original author) should run upon picking up this handoff for
Week 2. Execute in numbered order — do not skip.

### Step 5.1 — Write the `src/data_loader.py` embargo-aware stub (defense-in-depth against final-holdout leakage)

1. Create `d:\ML\Quant\src\data_loader.py` (first real `.py` module inside
   `src/`). Reuse the same `_raw_yahoo_chart()` pattern from
   `scripts/freeze_universe.py` for first-time pulls, or read from a local
   `data/raw/` parquet/csv cache if already populated.
2. Expose a single entry-point:
   `load_prices(tickers: list[str], start_date: str|Timestamp, end_date: str|Timestamp, *, final_holdout: bool = False) -> pd.DataFrame`.
   Keyword-only `final_holdout` is mandatory (cannot be passed positionally)
   and MUST default to `False`.
3. **Guard logic:** Before any `return`, always apply:
   ```python
   if not final_holdout:
       df = df.loc[df.index < pd.Timestamp("2024-01-01")]
   ```
   If `final_holdout=True` is explicitly set, return the full requested
   range unchanged (used ONLY once in Phase 8 final evaluation script; any
   other caller passing `True` outside Phase 8 is a forbidden look-ahead).
4. Raise a clear `ValueError` if `final_holdout` is not a Python `bool`
   (prevents accidental truthy string `"True"` / `"yes"` silent passes).
5. Expose a parallel helper for IPO / meta lookups only if needed. Do NOT
   write feature code here — `features.py` is step 5.3.

### Step 5.2 — Write `tests/test_no_lookahead.py` synthetic future-value injection test (MUST PASS BEFORE any modeling)

1. Create `d:\ML\Quant\tests\test_no_lookahead.py`. Test target is the
   `load_prices(...)` stub from step 5.1.
2. Synthetic data pattern (no real Yahoo calls): construct an in-memory
   DataFrame with DatetimeIndex spanning 2023-11-01 → 2024-03-01, one row
   per business day, and a single well-known ticker (say `RELIANCE.NS`
   `Close` column). Set the 2024-01-15 `Close` to an unmistakable sentinel
   value (e.g., `999_999.0`) that cannot occur organically. Write this to a
   temp CSV/parquet via pytest `tmp_path`, then monkeypatch the loader's
   cache read so it returns this synthetic frame.
3. **Test body:** Call
   `load_prices(["RELIANCE.NS"], "2023-11-01", "2024-03-01")` with
   `final_holdout` defaulted (i.e., `False` implicit). Assert:
   - Returned index has **no row on or after 2024-01-01**.
   - The sentinel `999_999.0` value is **not present anywhere** in the
     returned DataFrame.
4. **Complementary test:** Call the same with explicit
   `final_holdout=True`; assert the 2024-01-15 row and the sentinel value
   ARE present. This proves the guard works in both directions.
5. Run `pytest tests/ -v`; expected result: **2 passed, collected 2 items,
   exit code 0**. Do NOT proceed to features / notebooks until this test
   file is green.

### Step 5.3 — Implement `src/features.py` + run `notebooks/01_data_exploration.ipynb`

1. `src/features.py` should accept a wide prices DataFrame from
   `load_prices` and return:
   - Daily simple returns and log-returns columns (per-asset).
   - 1-d, 5-d, 21-d rolling vol (annualized sqrt × sqrt(252)).
   - 5-d / 21-d / 63-d SMA cross-over signals.
   - Feature-target splits with `target = t+1 : t+H forward log-return
     aggregated to the rebalance horizon` (H = 21 or 63, not locked yet
     until Stage 1A frequency winner — keep it a parameter).
2. `01_data_exploration.ipynb` notebook deliverables:
   - Per-asset NaN heatmap across 2015–2023 dev window.
   - Top-5 / bottom-5 CAGR bar chart on the 46 survivors.
   - 30-day rolling correlation histograms (pairwise) + N=largest
     correlation cluster dendrogram.
   - Liquidity-by-year bar chart (daily `Volume` deciles).
   - Confirm sector weight distribution matches §2 D8 NIFTY-relative cap.
3. Do **NOT** yet run any optimizer, any ML training, or any Stage 1A
   backtests in this notebook — exploration only.

### Step 5.4 — Stage 1A: Monthly (21 d) vs Quarterly (63 d) rebalance frequency comparison on the 4 classical baselines

1. Four classical baselines (per §2 frozen pipeline):
   - **(1) Equal Weight (1/N)**.
   - **(2) Global Min Variance** (long-only, 0 ≤ wᵢ ≤ w_max, Σwᵢ = 1).
   - **(3) Classic Max Sharpe (naive historical μ̂)** (long-only + ± 3 %
     NIFTY-sector-relative cap).
   - **(4) Risk Parity** (iterative equal-risk-contribution, no sector cap).
2. Single comparison metric: **transaction-cost-adjusted annualized Sharpe
   Ratio** of the 4-strategy mean (or per-strategy individually if spread
   is wide). 10 bps per turnover leg applied uniformly from this point
   forward (never later).
3. Walk-forward on dev window only (2015-01-01 → 2023-12-31), expanding or
   rolling training window per §C D2 default. The winner frequency is
   **locked globally** for all 6 strategies (4 baselines + 2 ML) for the
   rest of the project — no re-test.
4. Document the winner in `CHANGELOG.md` as a frozen decision and update
   `CHECKPOINT.md §2` row "Rebalance frequency selection methodology (4C)"
   from "decided empirically Stage 1A" to the actual numeric winner
   (Monthly or Quarterly) with exact tx-cost-Sharpe values.

(PHASE-1 ACTIONS 5.1 UA-PATCH / 5.2 FREEZE-RUN / 5.3 CAGR-PROXY: ALL
EXECUTED 2026-10-02. See §1 current-status block for verification.)

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
| `CHANGELOG.md` | Week 1 narrative COMPLETE ✅. All pending / DEFERRED items moved to Completed section. Phase 1 exit checklist (6 items) = 100 % checked. New completed-items entries added for: UA-fix + raw-chart helper smoke-pass; full 50-ticker freeze run 46 survivors; APOLLO 2002-07-01 resolved; §E 51.6-bps real proxy; lit matrix §F 21 papers 6 buckets ≥ 3; pytest exit-5 sanity OK. | ✅ Authoritative Phase-1 signed-off state. |
| `CHECKPOINT.md` | This file — session handoff. §1 Current Status rewritten to Week-1-Complete. §3.1/3.2/3.3/3.4 all RESOLVED (no unresolved gotchas remain). §5 rewritten as Phase-2 Kickoff ordered action list 5.1→5.4. §6 below = updated file-table. | ✅ Authoritative for Phase-1 end handoff. |
| `CONTEXT.md` | Original architecture / math / 16-week roadmap document. **Intentionally NOT modified in this session or this project phase.** The 4 known deviations from it are documented explicitly in §3.5 above (sector constraint 9B, rebal freq selection 4C, Σ selection 11, the 3+1 F1/F3 ticker rejections note). Frozen per user standing rule. | ✅ Frozen reference doc. 4 known deviations documented in §3.5. |
| `data/raw/universe_frozen.csv` | 46 rows (frozen survivors only). Output of the 2026-10-02 `freeze_universe.py` 46.5 s end-to-end run. Sorted by `free_float_rank` ascending. Columns identical to `docs/nifty50_sector_map.csv` rows that passed F1+F2+F3. 4 rows NOT present here = the 4 F1 rejects (HDFCLIFE, SBILIFE, HDFCAMC, TATAMOTORS). **Single source of downstream truth for every Phase 2–8 pipeline.** | ✅ Final / authoritative. Do NOT hand-edit — only regenerate if filter rules are re-frozen with spec change. |
| `data/raw/_survivorship_bias_proxy_cagrs.csv` | 46 rows, audit-trail output of `scripts/_oneoff_calc_survivorship_bias_proxy.py`. Columns: `ff_rank, ticker, cagr_pct, note`. Used to produce the `§E` 51.6-bps low-bound proxy number. NOT a downstream pipeline input — kept only as reproducible evidence. | ✅ Audit trail complete. Treat as read-only. |
| `docs/literature_matrix.md` | Six sections, all populated: §A frozen dataset spec (11 params, with 4 known CONTEXT deviations) · §B Stage 1 pipeline ASCII + dev/holdout hard boundary · §C 12 documented defaults D1–D12 · §D filter rules + 4-row definitive rejection log + 46-row survivor table + CSVs footer block (TATAMOTORS HTTP 404 permanently recorded) · §E Limitations paragraph: survivorship-bias **LOW-BOUND 51.6 bps** real computed proxy (exact formula, script path, top10/bottom10 means cited; old 25–75 bps human guess PERMANENTLY DELETED) + sequential-testing bias D12 = 24 configs · §F literature standalone matrix = **21 papers across 6 themes × ≥3 refs each** (18 mirrored from Full Matrix body + 3 targeted adds: Jagannathan & Ma 2003, Fan et al. 2008, Sortino & van der Meer 1991, Ang et al. 2006, Fama-French 2015, Scherer 2002). | ✅ All 6 sections complete. §F meets 15–30 paper target (N=21). |
| `docs/nifty50_sector_map.csv` | 50 NIFTY rows × 15 columns full filter trail. 0 DEFERRED cells anywhere; F1/F2/F3 are real booleans (`True`/`False`) on every row; `f3_manual_ipo_check` is a boolean (APOLLO = False → no caveat). 4 rows have F1=False + exact `f1_reason` strings copied from freeze_universe.py stdout. **Authoritative record of the 4 rejections** (used by `literature_matrix.md §D` rejection log). | ✅ Final / authoritative frozen trail. |
| `docs/synopsis.pdf` | Project synopsis PDF. Existed before this session. Not modified. | — Not in scope for this session. |
| `LICENSE` | License file. Not modified this session. | — |
| `README.md` | Honest `Status: Week 1 — planning phase`. Created pre-session. Pipeline ASCII diagram identical to `CONTEXT.md §2`. Links to `CONTEXT.md` and `docs/synopsis.pdf`. Status line is intentionally NOT bumped yet because Phase-2 (Week 2) work has not begun — bump it when the first `src/*.py` file beyond `__init__.py` is actually written (data_loader.py stub). | ⚠️ Accurate for current repo state. Optional status bump at Phase 2 kickoff. |
| `requirements.txt` | 20 pinned dependencies + numpy<2 pin rationale + plotly<5.23 for vectorbt heatmapgl compat. All verified in a fresh venv 20/20 smoke tests passing, `pip check` clean. | ✅ Complete and verified for this phase. |
| `scripts/__init__.py` | Empty package marker. | — |
| `scripts/_oneoff_calc_survivorship_bias_proxy.py` | NEW (≈ 200 lines). Audit-trail one-off script for the `§E` survivorship-bias low-bound proxy computation. Reads `universe_frozen.csv`, pulls 2015–2023 Close per ticker via same `_raw_yahoo_chart` helper, computes `CAGR_i = (P_end/P_start)^(1/n)-1`, writes `_survivorship_bias_proxy_cagrs.csv`. 46/46 valid, 20.5 s wall-clock. Keep permanently (reproducibility evidence), even though it will not be re-run in the normal pipeline. | ✅ Executed; audit trail complete. |
| `scripts/freeze_universe.py` | Reproducible 3-filter script (F1/F2/F3), now with: (1) Chrome-mimic UA `requests.Session()` + `yf.utils.user_agent_headers` patch; (2) yfinance cache redirected to `.yf_cache/` ROOT dir (sandbox fix); (3) new `_raw_yahoo_chart()` top-level helper (raw Yahoo v8 `finance/chart` GET, builds OHLCV DataFrame, `auto_adjust=True` semantics — replaces the yfinance `.download(session=...)` path that failed in the sandbox); (4) `_ipo_first_trade_date` rewritten to wide 1980–2015 window reading `result[0].meta.firstTradeDate` epoch UTC (replaces sandbox-broken `tkr.info` lookup); (5) `_apply_filters` calls `_raw_yahoo_chart` directly. All thresholds still match frozen spec exactly; 0.15 s / ticker polite delay; 2024+ holdout never read. | ✅ UA patch applied; helpers verified; full 50-ticker 46.5 s run exited 0 with 46 survivors. |
| `src/__init__.py` | Empty package marker; no `src/*.py` module code written yet. Next file to write here (Phase 2 Kickoff step 5.1): `src/data_loader.py` embargo-aware loader with keyword-only `final_holdout=False` guard. | Empty scaffold. |
| `tests/__init__.py` | Empty package marker; no test bodies written yet. Pytest discovery sanity-check (collected 0 items / exit 5) was run 2026-10-02 and passed (§1 above). Next test to write: `tests/test_no_lookahead.py` synthetic future-value injection (Phase 2 step 5.2). | Empty scaffold + framework wiring verified (pytest exit 5). |
| (Directories with `.gitkeep` only): `notebooks/`, `data/processed/`, `results/`, `results/figures/` | Not modified. | — Empty placeholders, correct. |
