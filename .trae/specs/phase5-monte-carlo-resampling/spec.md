# Phase 5 — Monte Carlo Resampling — Product Requirements Document

## Overview
- **Summary**: Implement the Monte Carlo resampling estimation-error mitigation layer on top of the Phase 4 frozen RIDGE μ̂ point-estimate pipeline. Generate 500–1000 perturbed μ̂ scenarios per quarterly rebalance via empirical (Gaussian-rejected) multivariate-row residual bootstrap, re-solve the LW Σ Max-Sharpe QP (±3pp sector QP, 10% cap) in parallel for every scenario, aggregate weights via mean/median/medoid centroids, and converge to a single centroid portfolio whose inter-RD turnover and max concentration strictly improve vs. the Phase 4 RIDGE single-point-estimate strategy.
- **Purpose**: This is the project's core methodological contribution per CONTEXT.md §6 Phase 5 and CHECKPOINT.md §1. It converts the error-maximizing single μ̂-point input to Markowitz MVO into the center of a distribution, forcing the solver to commit across hundreds of plausible inputs and therefore producing weights that are less sensitive to RIDGE forecast noise.
- **Target Users**: The project handoff LLM (next sessions), the project supervisor (Dr. Ruchika Sehgal), and the Phase 6 backtest/Phase 7 stress-test evaluator.

## Goals
1. Build `src/monte_carlo.py` with the two public APIs mandated by CONTEXT.md §6 line 468: `simulate_scenarios()` and `resample_weights()`, plus multivariate row residual cross-asset correlation preservation (CONTEXT.md §6 line 459 pitfall).
2. Run a full Phase 5 walk-forward on the 37 frozen quarterly rebalance dates using Mode C (multivariate-row bootstrap, block-size aware, scaled residuals), with joblib parallelization across per-draw QP solves.
3. Prove (not assume) Monte Carlo *helps* stability: selected centroid strictly lowers inter-RD annualized turnover and equal-or-lowers single-name max concentration vs. Phase 4 RIDGE single-point-estimate, on the identical 32 true ML RD window.
4. Produce a frozen centroid weights CSV (37 × 46 = 1702 rows) + 7/7 authoritative exit-gate banner verifier (analogous to Phase 4 `_p4_exit_gates.py`).
5. Satisfy CONTEXT.md §6 Phase 5 deliverables: convergence curve, MC-vs-point A/B notebook comparison, and parallel runtime benchmark.

## Non-Goals
- Do NOT retrain Ridge/RF/XGB models. μ̂ source is the frozen `phase4_ridge_linreg_forecasts.csv` on disk only.
- Do NOT change covariance estimator, sector tolerance, weight cap, transaction cost, metric convention, or rebalance schedule. Every single numeric protocol from Phase 3/4 is frozen per CHECKPOINT.md §2 rows 1–12.
- Do NOT touch the 2024+ final holdout data.
- Do NOT use any parametric (normal / t-distributed) draws for production MC weights. Gaussian was rejected (JB p ≈ 0, 3/3 families); only empirical residual draws are allowed.
- Do NOT hand-roll a new QP for the per-scenario solve. Reuse `classic_max_sharpe_weights` and `_post_verify_weights` from `src/optimizer.py` byte-for-byte including the Clarabel sector-QP min-distance post-projection.

## Background & Context
### Frozen input seeds (single source of truth, MUST reuse)
| Item | Source file / location | Value |
|---|---|---|
| μ̂ (forecast vector per RD × ticker, 63d quarterly horizon, log-ret) | `data/processed/phase4_ridge_linreg_forecasts.csv` | 1472 rows. cols: `rebalance_date, ticker, model_family, mu_hat_logret_63d, yhat_21d_point, train_start_date, train_end_date, n_train_rows`. First 5 RDs (2015 CMS fallback) preserved exactly. |
| Frozen RIDGE OOS residual extraction contract | `src/ml_models.py:extract_oos_residuals("ridge_linreg", X, y, n_splits=5)` | 5-split TSS, `shuffle=False`, returns tuple `(yhat_concat, resid_concat, fold_ids)`. Expected concat length 89608 = len(X) = len(y). |
| Covariance estimator | `src/covariance.py:ledoit_wolf_cov()` + `src/optimizer.py:COV_ESTIMATOR_WINNER == "LW"` RuntimeError gate if not LW | LW only. 63 BD lookback. |
| Per-scenario solver + post repair | `src/optimizer.py:classic_max_sharpe_weights()` → `src/optimizer.py:_post_verify_weights(..., w_upper=0.10, sector_info={...})` | 4-stage: sum-1 auto / non-neg clamp / 400-iter 10% single-name cap clamp / Clarabel min-distance ±3pp survivor-46 sector QP projection. |
| Sector target convention (survivor-46 renormalized NOT NIFTY 50-50) | `src/optimizer.py:classic_max_sharpe_weights()` + Phase 3 Bug #3 fix documented | sector_targets computed on N_survivors=46 counts; same function used by CMS baseline that produced ±3.000pp EXACT bound in Phase 4 G5. |
| Transaction cost model | 10 bps per unit one-sided turnover; drift definition: `0.5 Σ|w_i,post_drift − w_i,new_post_turnover|` | Used in Phase 3/4 txadj equity builder verbatim. |
| Metrics formula (byte-for-byte with NB2 Panel 8) | `_nb2_consistent_metrics()` inside Phase 3 scripts/phase3_run_5baselines.py | geo CAGR, simple daily pct_change mean − rf_daily over (simple std ddof=1 × √252) = excess Sharpe rf=4% annual, cummax-based MDD, Calmar = CAGR / abs(MDD), annual turnover bps/yr from RD turnover × (252 / avg_bd_per_rd_gap). |
| RD list frozen | 37 quarterly rebalance dates generated in `scripts/build_stage1a_baselines.py` | Same 37 RDs used in Phase 3/4 equities; MUST be identical list (no new RD generation allowed). |
| RNG seed lock | 7 | `numpy.random.default_rng(seed=7)` everywhere RNG invoked; never bare `np.random` calls. |
| Gaussian null verdict | `phase4_residuals_summary.csv` all 3 rows `gaussian_null_rejected_at_1pct == True` | Empirical bootstrap MANDATORY; parametric draws FORBIDDEN in production per-draw μ̂ perturbation. |

### Residual scaling contract (SILENT BUG PITFALL)
- `mu_hat_logret_63d` in forecast CSV = 21d RIDGE yhat × SCALE=3.0 (used in Phase 4).
- `residuals = y_21d_true - y_21d_hat` = 21-day horizon residuals (because target is `y_fwd_21` log-ret).
- To perturb a 63D μ̂ with the correct noise magnitude you must scale: `residual_63d = residual_21d × sqrt(3.0)`. Volatility scales with √ of time when returns are additive. Scaling linearly (×3) would inject 3× too much variance → downstream centroids would be uselessly noisy.
- HARD ASSERTION REQUIRED in code: compute `std(resid_concat * np.sqrt(3))` and compare to the expected residual volatility used by scenario generator; the two must match to within `< 1e-9`.

### Dependencies verified installed in .venv (2026-10-06 pip show)
arch 7.0.0, joblib 1.4.2, xgboost 2.1.1, scikit-learn 1.5.2, cvxpy 1.5.3 (Clarabel), PyPortfolioOpt 1.5.5, pandas 2.2.x, numpy<2.

## Functional Requirements
- **FR-1**: `src/monte_carlo.py` exports `simulate_scenarios(mu_frozen_df, ridge_oos_resid_long_df, mode, n_scenarios, block_size=21, seed=7, rebal_dates_order, tickers_order, scale_resid_sqrt3=True) -> np.ndarray shape (n_rebal, n_tkr, n_scenarios)` supporting modes: `iid`, `block_21`, and `multivariate_row` (PRIMARY). Mode `multivariate_row` jointly resamples residual dates so all 46 tickers' errors for the same historical date enter the perturbation vector together — preserving cross-asset forecast-error correlation structure explicitly per CONTEXT §6 line 459.
- **FR-2**: `src/monte_carlo.py` exports `resample_weights(rd_list, ticker_list, mu_scenario_tensor, log_ret_panel, sector_targets_per_rd, *, aggregation, risk_aversion_lambda=2.0, n_jobs=-1, warm_start=True, seed=7)` returning tuple `(centroid_weights_df[rd x tkr], raw_weights_long_df[rd,draw_id,tkr,weight], per_draw_sharpe_series)`. Every single scenario solve reuses `optimizer.classic_max_sharpe_weights` followed by `optimizer._post_verify_weights(..., w_upper=0.10, sector_info=sector_info_rd)` — no hand-rolled constraints allowed.
- **FR-3**: `scripts/phase5_step1_build_residuals.py` produces: (a) `data/processed/phase5_ridge_oos_residuals_empirical.csv` rows == 89608 long-format with cols `date, ticker, y_true, yhat_oos, residual`, (b) `phase5_residual_distribution_summary.csv` single row containing 5-number summary + mean/std/skew/kurtosis_excess/JB stat+P/gauss_rej_1pct + counterfactual `t_df/t_loc/t_scale` MLE fit (for report only, never used for draws). kurtosis_excess HARD ASSERT > 0 (rejects degenerate case).
- **FR-4**: `scripts/phase5_step2_bootstrap_modes.py` runs 3 modes for K=500 placeholder, prints integrity checks: (i) block contiguity (10 random draws pass), (ii) multivariate same_date_mask.sum() == 46 for all draws, (iii) residual sqrt(3) scaling assert <1e-9, (iv) perturb_std comparison mode_C >= mode_B >= mode_A. Writes `phase5_bootstrap_mode_integrity.csv`.
- **FR-5**: `scripts/phase5_step3_mc_weight_loop.py` runs the FULL 37 RD × K scenarios using PRIMARY mode `multivariate_row`, joblib parallel per draw, warm start. Outputs: (a) `phase5_mc_weights_long.csv` rows >= 37*500*46 = 8,510,000 cols `rebal_date,draw_id,ticker,weight`; (b) `phase5_mc_per_draw_equities.csv` 2222 dev dates × K draws; (c) `phase5_runtime_benchmark.csv` (total s, s/RD, s/solve avg, n_jobs, mem peak GB); (d) prints per RD progress with unbuffered stdout (`PYTHONUNBUFFERED=1`). CRITICAL CORRECTNESS ASSERT: for every RD with index >= 5 (true ML, not CMS fallback), scenario k=0 (mu FROZEN unperturbed + LW sigma frozen + post verify chain) produces a weight vector L2-distance vs. the frozen Phase4 RIDGE point weight of that same RD that is < 1e-6. If this fails the MC pipeline is numerically different from Phase 4 (bug).
- **FR-6**: `scripts/phase5_step4_convergence_curve.py` evaluates K' subset sweep 50/100/200/500 (no rerun, slices from the long K=500 CSV). For each K' and for each agg method (mean/median/medoid): compute 4 centroid WF metrics plus draw-distribution quantiles. Writes `phase5_mc_convergence_curve.csv` cols: `K, agg_method, sharpe_txadj, q05, q50, q95, interval_90_width, interval_width_pct_change_vs_prev, centroid_l2_diff_vs_k500`. Stores stopping rule result.
- **FR-7**: `scripts/phase5_step5_centroid_selection.py` runs C1 mean / C2 median / C3 medoid_draw centroids through post_verify_weights (sum/±3pp/10% clamp on the aggregated centroids themselves — because mean/median aggregations DO NOT PRESERVE constraints exactly), runs 3 WF equities+10bps+nb2 metrics, computes Phase 4 vs. MC inter-RD turnover and max concentration, then writes: (a) `phase5_mc_centroid_verdict.csv` rows = C1/C2/C3/DECISION with justification per frozen rule (top sharpe, Δ<0.02 → prefer median C2), (b) `phase5_mc_selected_weights.csv` rows EXACTLY = 37 * 46 = 1702 cols `rebal_date, ticker, weight`, (c) `phase5_stability_ab_vs_phase4_pointestimate.csv` cols `strategy, avg_inter_rd_turnover_bps_ann, max_single_name_concentration_pp, sharpe_txadj, calmar, turnover_rank_asc`.
- **FR-8**: `(Budget optional) scripts/phase5_step6_lambda_frontier.py` sweeps λ ∈ {1,2,4,8}, reuses chosen K from convergence, writes `phase5_lambda_frontier.csv`. Decision flag: stick with λ=2.0 unless (sharpe >= λ2+50 bps) AND (strictly lower MDD).
- **FR-9**: Create `notebooks/03_monte_carlo_resampling.py` (jupytext percent, paired ipynb) 8 panels (QQ+5num, bootstrap-mode box, convergence curve, 3-centroid equity overlay, stability A/B bar, 2016/2019/2022 snapshot, winner drawdown vs. RIDGE+CMS+1N, K-draw Sharpe histogram + quantile vert + winner star).
- **FR-10**: Create `notebooks/_run_nb3_validation.py` headless renderer with TWO HARD ASSERTS: (1) stopping rule met (interval width change <5% from K=200→500 OR K>=1000 documented); (2) selected centroid `avg_inter_rd_turnover_bps_ann < phase4_ridge_point.avg_inter_rd_turnover_bps_ann AND selected_centroid.max_concentration <= phase4_ridge_point.max_concentration` (MC MUST IMPROVE stability not hurt it — methodological void if violated).
- **FR-11**: Create `tests/test_monte_carlo_smoke.py` 1 test, 3 asserts: (1) simulate_scenarios shape check (K=10 synthetic x N_RD_small=3 x N_TKR=5), (2) mode multivariate preserves N_tkr=5 per same-draw joint slice, (3) resample_weights sum-to-one and 0<=w<=w_upper for all draws. pytest total goes from 37 → 38/38.
- **FR-12**: Create `scripts/_p5_exit_gates.py` 7/7 exit gate banner verifier analogous to Phase4: G1 Step0 tripwire chained green 3/3; G2 resid 89608 rows + kurtosis>0 + JB rej; G3 mode C 10/10 contig + 46/46 joint; G4 sqrt3 scale assert PASS + drw_000 vs phase4 w L2 <1e-6 for all RD>=5; G5 long weights on disk >=8.51M rows every row 0<=w<=10.001% every (rd,drw) sum|w-1|<1e-8 every (rd,drw) sector drift +/-3.001pp max; G6 convergence rule met (interval width change 200-500 <5% or K>=1000 rerun documented); G7 centroid: 3 on disk + DECISION row + stability A/B flags both TRUE (turnover lower, conc <= point). Script MUST print `>>> 7 / 7  P A S S  <<<` on success.

## Non-Functional Requirements
- **NFR-1**: Reproducibility: any fresh clone running Phase 5 scripts in the same environment (pinned reqs) produces numerically identical raw_weight_long.csv float values up to cvxpy Clarabel solver tolerance (< 1e-8 relative diff at 99th percentile across all 8.51M cells). Seed lock 7 everywhere.
- **NFR-2**: Parallelism from Day 1. Inner per-draw loop MUST use `joblib.Parallel(n_jobs=-1, prefer="processes")(delayed(...))` — NOT a single-threaded Python for-loop across 37 RD × 500 = 18,500 QP solves. Runtime: ≤ 30 minutes wall-clock at K=500 on a Ryzen 5 U (expected ~5-15 min once warm because cvxpy Clarabel on 46-variable QPs is microsecond-scale / 10s per RD).
- **NFR-3**: No look-ahead. (a) Scenario draws use ONLY `resid_concat` from `extract_oos_residuals()` — that function itself uses TSS `shuffle=False`, so NO future residuals can appear before their chronological fold boundary is strictly correct by construction. (b) LW Σ per RD only looks back 63 BD (same frozen function). (c) No cell in code reads 2024+ dates — `final_holdout=False` guard must be passed to any data loaders.
- **NFR-4**: Memory safe. Do NOT materialize an 8.51M row wide array float64 (≈280 MB) at once if a 16 GB Windows machine would page; write per-RD chunked CSV with `mode='a'` append header only once.
- **NFR-5**: Defensive asserts fail LOUD. Every frozen protocol (solver chain, scaling, weight constraints, sector bounds, RNG seed) must be encoded as an explicit `assert COND, "human-readable error with file+line+reason"` — no silent "do the right thing" fallbacks, because silent wrong results are the #1 killer of quant ML pipelines.
- **NFR-6**: All data artifacts go to `data/processed/*.csv` (force-allowlisted from gitignore line 49 same pattern as Phase 3/4). No `.npy` as single source of truth; CSV long-format authoritative even if .npy is written as an optimization.
- **NFR-7**: All 4 frozen-step env sanity (step0) + p5 exit gate runs green before any CHECKPOINT edits — exactly same chain as Phase 4 procedure.

## Constraints
- **Technical**: Only LW covariance. Only empirically-bootstrapped residual draws. Only Clarabel-based 4-stage post-verify sector-QP. Seed=7. All 37 RDs reused verbatim.
- **Business**: All 12 frozen CHECKPOINT.md §2 parameters enforced — no re-derivation allowed of Stage 1A freq, Stage 1B cov, universe, holdout, sector tolerance, txcost, target horizon, 21d→63d scaling constant=3.0.
- **Dependencies**: No new pip packages unless blocked (arch already installed); if a pinned version has a bug document but don't upgrade — pin to existing versions in requirements.txt.
- **Data embargo 2024-01-01+**: forbidden from read/ingest at any point in Phase 5 (must still pass `test_no_lookahead.py` 4/4 green after our edits — G1 gate chained).

## Assumptions
1. The 37 RD list stored in the quarterly_rebalance_prices.csv columns and phase4_ridge_linreg_forecasts.csv `rebalance_date.unique()` are sorted identical sets. No mismatches.
2. joblib Parallel(n_jobs=-1) works under TRAE sandboxed Windows .venv (not restricted to spawn / thread count caps).
3. 8.51 M row long-format CSV is writable under NTFS filesystem block limit on D:\ (~several GB free expected).
4. Phase 4 RIDGE first-5-RD CMS fallback forecasts are present in the forecast CSV (rows for those 5 RD×46 = 230 lines in CSV). If missing, we reuse Phase3 CMS μ̂ calculation exactly (not recomputed).
5. `src/ml_models.py:extract_oos_residuals` when invoked with model_family="ridge_linreg" will instantiate `Pipeline(StandardScaler(), Ridge(alpha=1.0))` identically to the training contract. (Self-check: G1 runs `_p4_exit_gates.py` green which already asserts this function works.)

## Acceptance Criteria

### AC-1: Environment tripwire chained green
- **Type**: `rule`
- **Given**: fresh terminal D:\ML\Quant .venv\Scripts\python.exe active
- **When**: (a) pytest tests/ -v exits with 38/38 PASS (one new MC smoke test); (b) sanity_check_features.py 7/7; (c) _p4_exit_gates.py 7/7
- **Then**: all three exit code 0 and no stdout red FAIL text
- **Pass Condition**: G1=Green in _p5_exit_gates.py (prints PASS)
- **Evidence**: stdout text copied into tasks.md completion evidence for each

### AC-2: src/monte_carlo.py public API contract + seed 7 + joblib parallel correct
- **Type**: `rule`
- **Given**: import src.monte_carlo as mc
- **When**: inspect dir(mc), call signatures, and execute smoke test (test_monte_carlo_smoke.py)
- **Then**: simulate_scenarios exists, resample_weights exists; smoke test 3/3 asserts PASS; pytest 38/38; multivariate draw same-draw joint=46; scenario output dtype float64 no NaN no INF
- **Pass Condition**: both function names in module, pytest exit 0 with 38/38 count, AND `(pd.read_csv('phase5_mc_weights_long.csv').isna().sum().sum() == 0)`
- **Evidence**: pytest line output + NaN assert line

### AC-3: Residual scaling (sqrt3 NOT linear ×3) + multivariate cross-asset correlation preservation
- **Type**: `rule`
- **Given**: residuals CSV, bootstrap_integrity CSV, raw_weight_long.csv AND residual std computed two ways
- **When**: compute `std(resid * sqrt(3))` vs. std used inside simulate_scenarios and compare
- **Then**: assert abs(diff) < 1e-9; multivariate joint slice (single draw 46 ticker residual perturbations for same RD) must correspond to same 46 historical dates (not 46 unrelated random draws) — verified 10/10
- **Pass Condition**: G4 (from _p5_exit_gates.py) prints [PASS] for both subconditions and G3 integrity 10/10 prints PASS
- **Evidence**: exact numeric assert values printed into both script stdout + gate banner

### AC-4: Per-scenario solve uses EXACT frozen CMS pipeline — drw_000 equals Phase 4 point estimate within L2 < 1e-6 for RD index 5+
- **Type**: `rule`
- **Given**: phase4_ml_weights.csv rows filtered model_family=ridge_linreg, and phase5_mc_weights_long.csv draw_id=0, RD 5+
- **When**: for each RD 5+, pivot both weight vectors into same ticker order, compute L2 norm of the difference
- **Then**: every RD 5+ L2 diff < 1e-6
- **Pass Condition**: max L2 across all RD>=5 < 1e-6, as checked in _p5_exit_gates.py G4
- **Evidence**: gate banner G4 PASS line with explicit max L2 value

### AC-5: All 37×K scenario weight vectors strictly satisfy post-verify constraints after every solve (0 breach across entire long CSV)
- **Type**: `rule`
- **Given**: full phase5_mc_weights_long.csv
- **When**: groupby (rebal_date, draw_id) then compute: sum(w) closeness to 1, max(w) <= 10.001%, min(w) >= -1e-9 (allow 1e-9 float noise on zero lower), sector weight aggregate vs. survivor-46 targets within + 3.001 / - 3.001 pp
- **Then**: all groups pass all 4 checks, fraction of breach rows == 0.0
- **Pass Condition**: _p5_exit_gates.py G5 prints PASS with breach-fraction = 0.000%
- **Evidence**: gate banner G5 table PASS rows

### AC-6: Stopping criterion met (K=500 adequate or K>=1000 rerun documented)
- **Type**: `rule`
- **Given**: phase5_mc_convergence_curve.csv interval_width_pct_change_vs_prev for K=200→500 row, agg method = selected centroid's method
- **When**: compare the percent change value to 5.0% threshold OR check K>=1000
- **Then**: condition holds AND `_run_nb3_validation.py` assert 1 passes (exit 0)
- **Pass Condition**: _p5_exit_gates.py G6 prints PASS; nb3_validation exit code 0 both hard asserts
- **Evidence**: convergence CSV percent change value copied + nb3 exit code 0

### AC-7: MC centroid stability A/B vs Phase 4 RIDGE point is IMPROVED (methodologically mandatory)
- **Type**: `rule`
- **Given**: phase5_stability_ab_vs_phase4_pointestimate.csv row `selected_centroid` and row `phase4_ridge_pointestimate`
- **When**: compare (a) avg_inter_rd_turnover_bps_ann centroid < point estimate; (b) max_single_name_concentration_pp centroid <= point estimate
- **Then**: BOTH (a)(b) TRUE across same non-overlap RD window (RDs 5+ true ML, same as Phase 4 ML-operating window)
- **Pass Condition**: _p5_exit_gates.py G7 prints PASS with both stability booleans True; nb3_validation assert 2 passes
- **Evidence**: stability CSV numeric values + gate banner + nb3 exit 0

### AC-8: Frozen centroid seed output exists and is correctly shaped
- **Type**: `rule`
- **Given**: phase5_mc_selected_weights.csv on disk + phase5_mc_centroid_verdict.csv DECISION row + justification non-empty
- **When**: rows count == 37*46 = 1702; columns rebalance_date/ticker/weight all non-null; decision CSV row count == 4 (C1/C2/C3/DECISION); justification string len > 20 chars
- **Then**: all pass
- **Pass Condition**: 1702 exact rows + 4 row verdict + non-empty justification
- **Evidence**: wc -l values + head -3 of CSVs pasted into gate evidence

### AC-9: Runtime benchmark recorded (CONTEXT §6 deliverable) + parallel loop real joblib usage
- **Type**: `rubric`
- **Dimension**: Computational-efficiency and evidence-quality of Phase 5 MC benchmark documentation
- **Scale**: 1-5
- **Anchors**: 1 = no runtime benchmark, serial-only loop (forbidden per NFR-2); 3 = benchmark present with 2 cols but n_jobs unclear; 5 = benchmark file cols exactly total_s / seconds_per_rd / seconds_per_qp_solve_avg / n_jobs / memory_peak_gb + stdout confirms Parallel backend spawned workers
- **Pass Threshold**: >= 4
- **Evidence**: runtime_benchmark CSV head + stdout joblib worker count line

### AC-10: Notebook comparison evidence (CONTEXT §6 deliverable) exists and nb3 validator runs green
- **Type**: `rubric`
- **Dimension**: Faithfulness to CONTEXT Phase 5 deliverables (convergence curve figure, point-vs-MC A/B, panels 1..8 renderer headless with valid assertions)
- **Scale**: 1-5
- **Anchors**: 1 = no notebook, no validator; 3 = 4 panels, no validator; 5 = 8 panels renderer, paired ipynb valid JSON, headless validator exit 0 with 2 hard assertions passing, stability metrics in panel 5 exactly matching disk CSVs to 6 decimals (no recompute drift)
- **Pass Threshold**: >= 4
- **Evidence**: nb3_validation exit 0 output + panel 5 Δ cell compare vs disk CSV value

### AC-11: Phase 5 7/7 exit gate banner prints PASS
- **Type**: `rule`
- **Given**: `.venv\Scripts\python.exe scripts/_p5_exit_gates.py` run in D:\ML\Quant
- **When**: command completes
- **Then**: exit code is 0; last banner line contains `ALL 7 PHASE 5 EXIT GATES  >>>  7 / 7  P A S S  <<<`
- **Pass Condition**: banner line substring match AND exit 0
- **Evidence**: final stdout lines pasted verbatim

## Open Questions
- [ ] None. Frozen parameters / frozen contract from Phase 3/4 leave no material ambiguity for the scope of this spec.
