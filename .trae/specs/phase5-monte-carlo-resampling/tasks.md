# Phase 5 — Monte Carlo Resampling — Implementation Plan

## Dependencies map (sequential)
T0 -> T1 -> (T2) -> T3 -> T4 -> (T5 optional) -> T6 -> T7 -> T8
(where T0 = spec.md approval is EXTERNAL PREREQUISITE that must be signed off BEFORE T1 starts)

---

## Task 1: Environment sanity — Step-0 tripwire & spec/tasks handoff
- **Status**: `complete`
- **Priority**: high
- **Depends On**: External (spec.md APPROVED by user)
- **Description**:
  - Verify fresh 3-step Step0 tripwire BEFORE writing any new code (pytest ≥37/37, sanity 7/7 X=(89608,77), _p4_exit_gates 7/7 banner). This greenlights all subsequent tasks; if any fail STOP immediately and remediate (e.g., missing CSV artifact from clone → regenerate 4-script ladder as documented in Project Memory).
  - Confirm arch / joblib / cvxpy 1.5.3 / pypfopt 1.5.5 / numpy<2 pinned requirements version match (skip install if already present; never upgrade pinned).
  - Create the output stub placeholder files so that downstream T3+ fail EARLY with clear "stub not populated" messages rather than silent NaN results.
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `rule` TR-1.1: pytest tests/ -v → terminal last line `37 passed` (before new MC test) and exit 0. Evidence: `pytest` command stdout captured.
  - `rule` TR-1.2: sanity_check_features.py → last line `7 / 7 assertions PASSED` and X.shape printed equals `(89608, 77)`. Evidence: script stdout last 3 lines.
  - `rule` TR-1.3: `_p4_exit_gates.py` → banner contains substring `7 / 7 PASS` and exit 0. Evidence: banner final 2 lines copied.
  - `rule` TR-1.4: all 6 Phase 4 core CSV artifacts non-empty on disk (count, not content): phase4_ridge_linreg_forecasts, phase4_ml_weights, phase4_ml_summary_txadj, phase4_ml_vs_baseline_verdict, phase4_model_selection_decision, phase4_residuals_summary → all 6 `wc -l` ≥ header + at least 3 rows (min of 3 true rows — e.g., residual_summary=3 model rows, verdict≥8 strategy rows). Evidence: powershell ls file length or wc -l.

---

## Task 2: Build src/monte_carlo.py (2 public APIs + 2 internal helpers + all defensive asserts)
- **Status**: `complete`
- **Priority**: high
- **Depends On**: T1 (green)
- **Description**:
  - Write `simulate_scenarios(mu_frozen_df, ridge_oos_resid_long_df, mode, n_scenarios, block_size=21, seed=7, rebal_dates_order=None, tickers_order=None, scale_resid_sqrt3=True)` → returns `np.ndarray shape (n_rebal, n_tkr, n_scenarios) float64`. Modes:
    - mode=`iid`: per-RD-per-tkr draw (baseline, ignores cross-asset correlation)
    - mode=`block_21`: arch.bootstrap CircularBlockBootstrap(block_size=21) per-RD-per-tkr (time-aware but univariate)
    - mode=`multivariate_row` (PRODUCTION PRIMARY): draw `date_ids` from `unique(resid_long_df.date)` with replacement, then for each RD and each scenario k, slice the 46-row vector `[resid_long_df[date == d] for all d ∈ draw_ids_draw_k]`, sum across draw repeats (or mean? → contract says *sample* w/ replacement, so keep repeats as additive or use occurrence count? → to keep std matching √3 scaled exactly, use *select without collapsing repeats so sample size == n_sampled_dates_per_scaling* → use a fixed `n_resid_draws_per_scen=63` or use single-draw-per? → simplest: 1 random date = 1 residual 21d slice per scenario → scaled to 63d via √3 multiply. This preserves CROSS-ASSET 46-variate correlation exactly, and for K=500 you get 500 distinct 46-vectors of perturbation. Document this choice in code comment.
  - Write `_multivariate_row_draw(...)` internal helper that draws dates and returns joint 46-vectors. ASSERT for 10 random draws that same-date slice sum tickers count ==46.
  - Write `_one_qp_solve_one_draw` internal helper that: takes `mu_perturbed: np.array[46]`, `sigma_lw: pd.DataFrame[46×46]`, `w_warmstart`, `sector_info: dict[same contract as used by optimizer._post_verify_weights]`, `risk_aversion_lambda=2.0`, `rf=0.04/252 daily` → runs `pypfopt expected_returns.mean_historical_return dummy bypass? Actually: use PyPFOpt `objective_functions.negative_sharpe` custom OR write custom μᵀw − (λ/2) wᵀΣw max with explicit r_f subtraction. Simplest: reuse `classic_max_sharpe_weights` but PASS μ̃_perturbed as the expected returns vector in place of the frozen μ̂. Read the signature of `classic_max_sharpe_weights` BEFORE coding this task (do not guess args). After solve chain result: feed output through `_post_verify_weights(w, w_upper=0.10, label=f"drw_{k:04d}", sector_info=sector_info)`. ASSERT post solve: sum w within 1e-8 of 1.0 AND 0≤w≤0.10001 AND sector drift ±3.001 pp max.
  - Write `resample_weights(rd_list, tickers_list, mu_scenario_tensor: np.array[n_rd,46,K], log_ret_panel: pd.DataFrame, sector_targets_per_rd: dict[rd_str -> Series sector_to_wt], *, aggregation: str in {mean, median, medoid_draw}, risk_aversion_lambda=2.0, n_jobs=-1, warm_start=True, seed=7) -> (centroid_weights_df[rd x tkr], raw_weights_long_df[rebal_date, draw_id, ticker, weight], per_draw_sharpe_approx_series length K×n_RD)`:
    - First: for each RD: call `ledoit_wolf_cov(log_ret_panel, tickers, lookback=63 bd)` EXACT same frozen call used by Phase 3/4 (do NOT change window).
    - Second: call `joblib.Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one_qp_solve_one_draw)(...) for each scenario)`.
    - Warm start within RD: solver call with previous draw's w as starting value via pypfopt `solver_kws=dict(initvals=w_prev)`. cvxpy Clarabel has warm start.
    - Collect raw weights per draw. Compute per_draw_sharpe via fast proxy `(mu_draw_vect @ w − rf)/sqrt(clip(w @ Σ @ w, 1e-12))` (approx, real sharpe via equity later).
    - Aggregation: `mean` = simple mean across K dimension then re-run `_post_verify_weights` AGAIN on centroid because linear means don't preserve sum=1 or sector ±3 bounds exactly; `median` = row-wise median then same post repair (this is literature-recommended more robust vs outliers); `medoid_draw` = find k* with minimal |approx_sharpe(k) − median(approx sharpes all K)| and return raw k* vector (no repair needed because already post-repaired individual draw).
- **Acceptance Criteria Addressed**: AC-2, AC-3
- **Test Requirements**:
  - `rule` TR-2.1: pytest `tests/test_monte_carlo_smoke.py` 3 asserts PASS (pytest 38/38 total). Evidence: pytest line output 38 passed, file path, assert descriptions.
  - `rule` TR-2.2: simulate_scenarios with mode multivariate on synthetic resid_long_df[cols date,tkr,resid,10 dates,5 tkrs,K=10,scale_sqrt3=False] → for draw 0 check all 5 perturbations in tkr dimension correspond to exactly one shared date value (sum across boolean masks = 5). Evidence: inline assert and printed slice.
  - `rule` TR-2.3: residual sqrt3 scaling hard assert present and PASS inside simulate_scenarios (printed "scale√3 assert PASS"). Evidence: copy line of stdout with assert pass text.

---

## Task 3: scripts/phase5_step1_build_residuals.py + step2_bootstrap_modes.py
- **Status**: `complete`
- **Priority**: high
- **Depends On**: T2 (src/monte_carlo.py API signatures stable)
- **Description**:
  - step1_build_residuals: load data using existing pipeline: `load_prices(final_holdout=False)` → `make_features` → `make_targets(horizon=21 BD)` → `align_X_y` → X,y shapes match (89608,77) / len(y)=89608. Call `extract_oos_residuals("ridge_linreg", X, y, n_splits=5)` from ml_models.py. Save 3 return values merged into long CSV: cols `date, ticker, y_true, yhat_oos, residual` (89608 rows). Save residual summary: 5-number summary, skew, kurtosis_excess, JB stat+P via `scipy.stats.jarque_bera(resid_concat[np.isfinite(resid_concat)])`. Counterfactual t-fit via `scipy.stats.t.fit(resid_concat[np.isfinite(resid_concat)])`. HARD ASSERT: kurtosis_excess > 0.
  - step2_bootstrap_modes: for placeholder K=500 (no convergence yet — run same for modes `iid`, `block_21`, `multivariate_row` on SMALL n_RD_subset=6 slice first to debug, then full 37 RDs). Run integrity assertions 10 random times for block and multivariate. Print mode_C >= mode_B >= mode_A perturb_std comparison table. Write `phase5_bootstrap_mode_integrity.csv` with 3 rows × 3 integrity cols PASS/FAIL.
- **Acceptance Criteria Addressed**: AC-3, AC-1 (step1 builds the resid source needed by G2)
- **Test Requirements**:
  - `rule` TR-3.1: resid CSV row count == 89608, columns match exactly [date, ticker, y_true, yhat_oos, residual]. Evidence: `df.shape` assert and column list.
  - `rule` TR-3.2: summary CSV kurtosis_excess > 0 (line assert inside script). Evidence: printed kurtosis value.
  - `rule` TR-3.3: integrity CSV 9 cells total == "PASS" string (3 modes × 3 checks). Evidence: head of CSV 9 cells PASS.
  - `rule` TR-3.4: residual √3 scaling assert printed PASS. Evidence: exact line in step2 stdout.

---

## Task 4: The Bottleneck — scripts/phase5_step3_mc_weight_loop.py (full 37 RD × K=500 Mode C)
- **Status**: `complete`
- **Priority**: high
- **Depends On**: T3 (residuals + mode integrity green)
- **Description**:
  - Read: phase4_ridge_linreg_forecasts.csv, quarterly_log_ret_panel (frozen RD list 37 dates from build_stage1a rebalance_dates_order.txt if exists OR from phase4_ml_weights.rebalance_date.unique() sorted). Verify the 37 RD sets are equal sets via assert.
  - Per RD 37 loop (sorted ASCENDING):
    - if RD is in first 5 (CMS fallback): mu_frozen = CMS_63d_μ̂ (get from phase3_run's 5-baseline raw inputs — easiest: query phase3_baseline_weights on disk if available OR calculate via same rolling mean exactly as Phase 3). ASSERT first 5 rows have mu_hat_logret_63d matching forecast CSV (no mismatch).
    - else RD >=6: mu_frozen_rd[46] from forecast CSV pivot (46 floats logret 63d).
    - Σ_LW_rd[46×46]: call `ledoit_wolf_cov(log_ret_panel_window=RD-63 BD to RD-1 BD)` with exact 63 BD lookback frozen same as P3/P4. DO NOT change window.
    - sector_targets_survivor46_rd: call the same helper CMS used (look at phase3_run / classic_max_sharpe for the correct import path). sector_tolerance=0.03. Pass the resulting dict into _one_qp_solve_one_draw sector_info.
    - mu_tensor_rd: simulate_scenarios(mode="multivariate_row", n_scenarios=500, seed=7, scale_resid_sqrt3=True).
    - joblib parallel solve 500 draws, warm start, warm_start_seed_rd=hash(RD+7) so scenarios are reproducible but different per RD.
    - For EVERY solved weight vector: run `_post_verify_weights` (this is redundant because `_one_qp_solve_one_draw` already does it, but DO IT AGAIN defensively on the collected results) AND hard ASSERT: sum w within 1e-8 of 1 AND max_w ≤ 0.10001 AND min_w ≥ -1e-9 AND sector_drift_abs_max ≤ 3.001 pp.
    - CRITICAL CORRECTNESS DRIFT GUARD: compute L2 norm between Phase4 RIDGE weights (from phase4_ml_weights filtered model==ridge_linreg, same RD) vs draw 0 of mu tensor (which is draw with scenarios that must produce same weights because draw0 scenario seed=7 if we set scenario k=0 => perturbed with EMPTY perturbation? No — simpler: construct a special SENTINEL test run where we pass a mu_tensor=0 perturbation (mu UNPERTURBED) and compare weights against Phase4 RIDGE point — then run full loop and print L2 diff for drw_000 where we ensure mu_scenario[..., 0] = exactly_mu_frozen_direct. This is the critical check. Write this assertion directly in the script. It MUST print `(PIPELINE DRIFT GUARD) max_L2_diff_across_RD5_plus = X.XXXXe-08 < 1e-6 PASS` every run. Exit code 1 if FAIL — loud.
    - Write weights for RD K 500 draws × 46 tkrs to long output CSV as mode='a' append with header only on first RD.
    - Benchmark timing per RD and total, record mem_peak via `resource.getrusage` or psutil if available, fall back to approx if not.
  - Post loop: write per_draw_approx_sharpe, runtime benchmark CSV.
  - Call with: `$env:PYTHONUNBUFFERED='1'; .venv\Scripts\python.exe -u scripts/phase5_step3_mc_weight_loop.py` so per-RD progress is visible, you don't stare at blank screen.
- **Acceptance Criteria Addressed**: AC-4, AC-5, AC-9
- **Test Requirements**:
  - `rule` TR-4.1: Full loop run to completion exit 0, long weight CSV exists ≥ 8,510,000 rows (37 × 500 × 46). Evidence: row count printout.
  - `rule` TR-4.2: NaN count across entire raw weight CSV == 0 (no nonfinite). Evidence: assert line.
  - `rule` TR-4.3: Pipeline drift guard across all RD ≥ 5 → max L2 < 1e-6. Evidence: exact printed max_L2 line.
  - `rule` TR-4.4: Entire long CSV 4 constraints zero breach (sum 1e-8, 10% cap, nonneg, ±3pp sector). Group-by (rebal_date, draw_id), fraction breach = 0.000%. Evidence: breach counter script output.
  - `rubric` TR-4.5: (covers AC-9) runtime_benchmark.csv cols present (total_s, s_per_rd_avg, s_per_qp_avg, n_jobs, mem_peak_gb); stdout confirms N>1 joblib workers spawned. Scale 1-5; anchors 1=no benchmark / serial only, 3=2 cols, 5=all 5 cols + stdout confirms worker count; threshold ≥4; evidence=CSV head + line.

---

## Task 5 (Optional high-priority if K=500 fails convergence): scripts/phase5_step6_lambda_frontier.py
- **Status**: `pending`
- **Priority**: medium (skipped if K=500 converges — runs post-selection for exploration and optional nb3 panel inclusion)
- **Depends On**: T4 (full loop complete with K=500)
- **Description**:
  - Sweep λ ∈ {1.0, 2.0, 4.0, 8.0}. Default λ=2.0 already.
  - Reuse simulate_scenarios tensors and Σ and sector info from T4 pickle cache (don't recompute LW for performance; but if cache absent, recompute LW 4 times — cheap because 37 small 63d windows).
  - Run 500-draw QP solve with λ-adjusted objective.
  - Per λ run 3 aggregations (mean/median/medoid) → full equity + 10bps + nb2 metrics.
  - Write `phase5_lambda_frontier.csv`. Decision flag: λ = argmax sharpe_txadj with strict filter `sharpe >= λ2 + 0.005 (50bps) AND max_dd_abs < λ2_max_dd_abs * 1.0`. If none meets, freeze λ=2.0 as default in step5.
- **Acceptance Criteria Addressed**: NFR-1 (via sweep reproducibility) + partial coverage for FR-8 budget deliverable
- **Test Requirements**:
  - `rule` TR-5.1: 4 λ × 3 centroid candidate metrics rows produced (12 rows output non-empty + final DECISION row appended). Evidence: wc -l == 13 rows.
  - `rule` TR-5.2: If λ != 2.0 decision row exists, verify BOTH conditions flag columns == True (sharpe_improvement_50bps_flag AND strictly_lower_mdd_flag). Evidence: DECISION row printed.

---

## Task 6: Convergence curve + centroid selection (step4_convergence_curve.py + step5_centroid_selection.py)
- **Status**: `complete`
- **Priority**: high
- **Depends On**: T4 (long CSV on disk), optionally T5 (if λ sweep done and λ != 2 → rerun weights before step6? No — just read K subset slices from disk for convergence; for step6 centroid use λ=2 frozen by default unless T5 explicitly decided better λ with user signoff. λ decision is NOT a free choice; it's a flag that requires 2 strict thresholds, so default λ=2 is lock.)
- **Description**:
  - step4_convergence_curve: slice long CSV K=500 draws into subsets draws 0..49, 0..99, 0..199, 0..499 → K=50/100/200/500. For each K' and each of {mean,median,medoid} aggregate centroid → run full WF equity 2222 dev dates + 10bps drag + _nb2_consistent_metrics. Also compute K' approx_draw sharpe quantiles 0.05, 0.50, 0.95 from per_draw_sharpe proxy. Compute: interval_90_width = q95 - q05, and interval_width_pct_change_vs_prev_K = ((width now - width at K previous) / width_K_prev)*100. Store. Also compute centroid_L2_diff_vs_k500 = L2 norm of (centroid_k'_weights - centroid_K=500_weights_flat) across 37×46=1702 cells. Write CSV with columns as AC-6 describes.
  - step5_centroid_selection: for selected K (500 default, unless stopping rule fails → rerun K=1000 via Task 6b if needed), compute C1=mean centroid across K draws per RD, C2=median (robust, literature default), C3=medoid_draw (representative scenario). IMPORTANT: PASS every centroid candidate vector through `_post_verify_weights` AGAIN — the linear aggregations do NOT conserve the sum-to-1 and ±3 sector tolerance exactly (you need to clamp mean again to be exactly 1 and project mean onto sector QP feasible set, same for median; medoid does not need repair since it's a draw, but you MAY re-run defensively anyway). ASSERT post centroid repair, same 4 constraints, zero breach on 37 × 3 × 46 = 5,106 centroid cells.
  - Next: run each centroid C1/C2/C3 full WF: equity, 10bps tx cost drift model EXACT same definition as P3/P4 (drift weights = prior w × (1+r_i_rdgap) / sum, turnover = 0.5 Σ |new - drifted|), nb2 metrics 17-col phase4_summary identical format so they can be concatenated with verdict.
  - Compute stability A/B comparison between Phase4 RIDGE point and each of 3 centroids: (a) avg_inter_rd_turnover_bps_ann — for each strategy, sum of RD-level turnover values (36 inter-RD gaps) multiplied by (252 / avg_bd_between_rd); (b) max_single_name_concentration_pp = max across RDs (max across 46 tkrs w * 100). Write stability_ab CSV with cols per AC-7.
  - Select final centroid:
    - sort C1/C2/C3 desc by sharpe_txadj (column exact match to phase4 naming)
    - if ΔSharpe TOP1 vs TOP2 < 0.02 → tiebreak → prefer MEDIAN (C2, more robust lit-backed)
    - write verdict row C1/C2/C3/DECISION rows to centroid_verdict.csv. DECISION row must contain: agg_method_selected, sharpe_txadj_selected, turnover_rank_selected, justification_string_len>20.
  - Write `phase5_mc_selected_weights.csv` rows EXACT = 37 × 46 = 1702 cols rebal_date, ticker, weight → FROZEN SEED for Phase 6.
- **Acceptance Criteria Addressed**: AC-6, AC-7, AC-8
- **Test Requirements**:
  - `rule` TR-6.1: convergence_csv rows = K(4) × agg_methods(3) = 12 rows. Evidence: row count + head.
  - `rule` TR-6.2: Stopping rule PASS — for SELECTED aggregation method, interval_width_change (200→500) < 5.0% OR K rerun ≥1000 documented. Evidence: gate G6 later checks this row.
  - `rule` TR-6.3: Selected weights row count 1702 = 37×46, nulls=0, cols match exactly. Evidence: shape assert.
  - `rule` TR-6.4: A/B stability. BOTH flags True (turnover_lt_phase4_point AND max_conc ≤ phase4_point). Evidence: stability_ab CSV line values.
  - `rule` TR-6.5: Centroid verdict DECISION row justification len > 20 characters, method non-empty. Evidence: row string.

---

## Task 7: tests/test_monte_carlo_smoke.py + notebooks (03 NB3 renderer, _run_nb3_validation.py)
- **Status**: `complete`
- **Priority**: medium (high for nb3 assertions because they are methodology guards)
- **Depends On**: T6 (stability_ab / convergence CSVs exist)
- **Description**:
  - tests/test_monte_carlo_smoke.py single test fn:
    - Build synthetic X, y (tiny uniform rng 7 draw, not real 89608).
    - Fit Ridge sklearn dummy (NOT pooled real data) → build resid_long_df synthetic.
    - simulate_scenarios on 3 RD × 5 TKR × K=10 tiny → (a) shape (3,5,10) assert, (b) 2 random draws mode multi check N=5 ticker values correspond to same dates, (c) resample_weights on tiny synthetic solve with min feasible Σ (semi-positive definite diag + outer 1e-3) → centroid sum-to-one assert and all weights in [0, w_upper=0.20 synthetic]. 3 asserts → 1 test.
  - notebooks/03_monte_carlo_resampling.py jupytext percent. Panels:
    7.1 Residual QQ plot vs standard normal + 5num + box.
    7.2 3-mode perturb_std boxplot per RD (mode_C ≥ B ≥ A expected visual).
    7.3 Convergence curve x=K, 2 y-axes (tx-Sharpe line + 90% interval fill + text width_pct_change).
    7.4 C1/C2/C3 centroid equity 3 curves overlay 2015-2023; legend + y log scale.
    7.5 Stability A/B bars: 2 sub-bar groups (bar 1 inter_rd_turnover_bps_ann 4 strategies: phase4_ridge/C1/C2/C3; bar2 max_singlename_conc_pp same 4) — HLINE reference at phase4_ridge baseline level so visual show MC centroids BELOW line (lower turnover / lower conc).
    7.6 Weight snapshot stacked bars for 3 rebal dates: 2016-01, 2019-01, 2022-01 (same dates as literature-standard snapshots). Sector colors colorblind-safe. Dashed 10% cap HLINE.
    7.7 Winner-selected centroid drawdown curve overlay vs Phase4 RIDGE point, CMS baseline, Equal W 1/N. MDD labeled in legend.
    7.8 K-draw approx Sharpe histogram, 0.05 and 0.95 quantile vert black lines, selected centroid tx-Sharpe red STAR marker (check star is outside 90% interval → indicates centroid consistently in top half).
  - notebooks/_run_nb3_validation.py headless renderer: two HARD EXIT CODE 1 asserts:
    - Assert 1 convergence: load convergence CSV, SELECTED agg_method row where K=200 and K=500, compute pct change width, assert < 5.0 OR alternative K decision >= 1000 (check K flag from convergence comments field if we ever go to 1000).
    - Assert 2 stability: load stability_ab CSV, get row selected_centroid, get row phase4_ridge_pointestimate. ASSERT: `selected_centroid.turnover_bps_ann < point.turnover_bps_ann AND selected_centroid.max_conc_pp <= point.max_conc_pp`.
- **Acceptance Criteria Addressed**: AC-10
- **Test Requirements**:
  - `rule` TR-7.1: pytest now 38/38 (37 old + 1 MC smoke). Evidence: pytest stdout line 38 passed.
  - `rule` TR-7.2: nb3_validation exit code 0, 2 assertions print "PASSED assert 1/2 PASSED assert 2/2". Evidence: full stdout with 2 pass lines.
  - `rubric` TR-7.3: (covers AC-10 dimension) nb panels (1) count 8/8 rendered; (2) paired ipynb valid JSON or at least jupytext pairing marker present; (3) panel5 values exactly disk CSV values 6 decimals. Scale 1-5; anchors 1=none, 3=4 panels, 5=all 8 + valid pairing + cell value exact match; threshold >=4; evidence=validator output panel count + exact Δ cell check.

---

## Task 8: scripts/_p5_exit_gates.py (7/7 banner verifier) — AUTHORITATIVE SIGN OFF
- **Status**: `complete`
- **Priority**: high (all gates)
- **Depends On**: All previous tasks T1..T7
- **Description**:
  - Mirror Phase 4 gate style from _p4_exit_gates.py. Seven gates:
    G1 Step0 tripwire chained: (a) pytest subprocess check output contains `38 passed` (we added 1 MC test → 37→38) OR at least ≥37 AND test_monte_carlo_smoke.py explicitly listed 1 passed. Actually exact: `assert num_pytest_pass >= 38`, because T7 adds one test. (b) sanity_check_features 7/7. (c) _p4_exit_gates banner contains 7/7 PASS. Print gate table.
    G2 Residuals: resid CSV rows=89608, kurtosis_excess>0, JB_rej_flag=1 (p-value < 0.01).
    G3 Bootstrap integrity: mode_C csv 3 cells PASS → read and assert all 3 strings equal PASS.
    G4 Drift guard + scaling: read (1) max_L2_diff variable from step3_drift_guard.log or recompute L2 from phase4_ml_weights point vs draw_000 weights → assert max < 1e-6; (2) scaling sqrt3 assert confirmed either from step2_integrity csv row or recompute std(21d*sqrt3) vs scenario perturb at zero RD ticker.
    G5 Long CSV constraints: load phase5_mc_weights_long.csv. GroupBy rebal_date+draw_id:
       i. sum w within 1e-8 of 1 → fraction fail = 0.0%
       ii. min w ≥ -1e-9 → 0.0%
       iii. max w ≤ 0.10001 → 0.0%
       iv. sector_drift_abs_max_pp ≤ 3.001 → 0.0%
       Print 0.000% for all 4.
    G6 Convergence: read convergence_csv, selected agg method → interval_width_pct_change_vs_prev_K for K=500 row vs prev K (=200) assert < 5.0%.
    G7 Centroid + stability A/B: (1) 4 centroid_verdict rows present, DECISION justification len>20; (2) selected_weights_1702_rows == 1702 cells non-null; (3) stability_ab selected centroid turnover_lt_point AND max_conc ≤ point BOTH True.
  - Final banner ASCII, exit 0 only if all 7 PASS.
  - Also write CHANGELOG.md Phase 5 completion block. (Bonus / optional — but recommended to document run date.)
- **Acceptance Criteria Addressed**: AC-11
- **Test Requirements**:
  - `rule` TR-8.1: Script exits 0 AND banner final line contains `ALL 7 PHASE 5 EXIT GATES  >>>  7 / 7  P A S S  <<<`. Evidence: full last 20 lines stdout.

---

## Queue prerequisites to start Implementation
- All tasks depend on T0 external (USER SIGNS OFF spec.md + tasks.md APPROVAL) → then T1 (step0 tripwire green) is the first runnable implementation item.
- If blocked on K=500 convergence (G6 FAIL at T6): add T6b (K=1000 rerun) pending with priority high, dependency=T4; unblock G6 after K=1000 produced. For the initial T4 run we start K=500 target.
