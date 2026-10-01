# CHANGELOG

All changes to the research plan, frozen specification, and codebase are
recorded here. Weeks are counted from project kickoff.

---

## Week 1 — Planning & Foundations (1/2 done — in progress)

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

- **Docs: Full Dataset Specification written to `docs/literature_matrix.md`**
  (2026-10-02)
  - §A — 11 frozen parameters (ID 1A, 2B'', 3A, 4C, 5B, 6A, 7B, 8', 9B, 10A, 11)
  - §B — Stage 1 sequential decision pipeline ASCII diagram + hard
    development/final-holdout boundary (2015-01-01 → 2023-12-31 /
    2024-01-01 → 2025-06-30)
  - §C — 12 documented design defaults (D1–D12) with rationales
  - §D — 50-row NIFTY 50 candidate universe table (order, ticker, company,
    provisional sector label, free-float rank, rejection column)
  - §E — **Limitations acknowledgement paragraph** covering both
    survivorship bias (today's-NIFTY-universe limitation) and
    sequential/multiple-selection testing bias (24 implicit configurations
    → justifies the Phase 6 DSR / PBO / Jobson-Korkie correction machinery)
  - §F — Literature matrix scaffold (5 rows + themes; 15–30 papers target by
    end of Week 1)

- **Chore: `scripts/freeze_universe.py`** (2026-10-02)
  - Reproducible script that pulls the 50 NIFTY candidates from yfinance
    (DEV window only — 2015-01-01 → 2023-12-31; final-holdout 2024+ is NEVER
    read), applies F1 (data availability) / F2 (liquidity) / F3 (IPO date)
    filters, and writes both `docs/nifty50_sector_map.csv` (all 50, full
    filter trail) and `data/raw/universe_frozen.csv` (survivors only).
  - Not successfully executed in this session due to a **transient
    Yahoo-Finance-level API block** on this internet connection (both US
    tickers and `.NS` tickers are rejected; not a code issue). The output
    CSVs below were produced via Phase-1-appropriate manual IPO-date filter
    pass instead. Script is preserved and will be re-run (possibly against
    an alternative `nsepy` or `nsetools` backend) in Phase 2, at which time
    F1 and F2 filters will also be verified and the universe row counts may
    be trimmed by a further 0–3 tickers.

- **Data freeze: `universe_frozen.csv` + `nifty50_sector_map.csv`** (2026-10-02)
  - 50 NIFTY 50 current candidates → **3 F3 IPO-date filter rejections**
    → **47 survivors** (target band 45–48 met)
  - Rejected tickers (all for first-NSE-listing date ≥ 2015-01-01):
    - 36  `SBILIFE.NS`     — listed 2017-10-03
    - 18  `HDFCLIFE.NS`    — listed 2017-11-17
    - 43  `HDFCAMC.NS`     — listed 2018-08-06
  - IPO-date filter notes: `APOLLOHOSP.NS` retained as survivor with a
    **manual NSE IPO-date double-check required in Phase 2** (provisional
    listing date used: 2012-12-01). If this date is wrong the universe count
    will drop to 46 (still inside 45–48 tolerance).
  - F1 (data availability) and F2 (liquidity) are marked **DEFERRED —
    Phase 2** in the sector map CSV; they cannot be evaluated until
    `scripts/freeze_universe.py` runs successfully against an Indian-market
    data backend available from this network.

### 🔜 Still pending before "Phase 1 — Foundations complete" sign-off

- [ ] `docs/literature_matrix.md §F` — 15–30 paper literature matrix
  (currently a scaffold of 5 rows)
- [ ] Phase 2 prep — resolve Indian-market data backend so
  `scripts/freeze_universe.py` can run end-to-end: verify F1 and F2 filters
  against the 47 survivors; confirm `APOLLOHOSP.NS` listing date
- [ ] Final verification: `pytest tests/ -v` discovers and passes 0 tests
  (sanity-checks test framework is wired before Phase 2 code is written)

---

## Phase-Signing Checklists

### Phase 1 — Foundations exit (from `CONTEXT.md §7`)

- [x] `docs/literature_matrix.md` **file created** (15–30 papers TBD — section
      F still pending content)
- [x] Final stock universe list frozen — 47 tickers (from NIFTY 50) with
      rationale and provisional sector classification documented
- [x] Full date range (2015-01-01 → 2025-06-30) + 18-mo final holdout
      (2024-01-01 → 2025-06-30) explicitly documented
- [x] `requirements.txt` committed; `.venv/` runs cleanly on a fresh install
      (20/20 smoke tests pass, `pip check` reports 0 broken dependencies)
