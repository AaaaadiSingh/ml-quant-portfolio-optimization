# %% [markdown]
# # Notebook 03 — Monte Carlo Resampling & Risk
#
# **Phase 5 Work Group 5 Deliverable**
#
# This notebook implements and verifies the Michaud (1998) resampled efficient frontier
# on top of the frozen Phase 4 RIDGE conditional return forecast seed.
#
# ### 8 Core Verification Panels:
# 1. Residual OOS Empirical Distribution vs Gaussian & Student-t fit
# 2. Bootstrap Mode Integrity (Block B=21 days & Multivariate joint row check)
# 3. Monte Carlo Forward Weight Simulation Loop (37 RDs x 500 draws)
# 4. Centroid Convergence Curve across Draw Count K (K=50..500)
# 5. Stability A/B Comparison (Centroid vs Phase 4 RIDGE Point Estimate)
# 6. Candidate Centroid Comparison Panel (Mean vs Median vs Medoid)
# 7. Selected Centroid Weight Heatmap (37 Dates x 46 Tickers) & Rebalance Snapshots
# 8. Final Compliance Audit & 2 Hard Asserts

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.monte_carlo import WEIGHT_UPPER_LOCKED, SECTOR_TOLERANCE_LOCKED
from src.optimizer import _sector_group_matrix

print("Phase 5 environment loaded. Root:", ROOT)

# %% [markdown]
# ## Panel 1: Residual Distribution & Heavy-Tail Diagnostic
# Gaussian null was rejected (Jarque-Bera p=0.0). Fat tails confirmed via Student-t MLE fit (nu=4.30).

# %%
resid_df = pd.read_csv(ROOT / "data" / "processed" / "phase5_ridge_oos_residuals_empirical.csv")
summary_df = pd.read_csv(ROOT / "data" / "processed" / "phase5_residual_distribution_summary.csv")

print("Residual count:", len(resid_df))
display_summary = summary_df[["n_obs", "mean", "std", "skew", "kurtosis_excess", "jb_stat", "jb_p_value", "t_df_mle"]]
print(display_summary.to_string(index=False))

# %% [markdown]
# ## Panel 2: Bootstrap Mode Integrity
# Comparison across naive i.i.d., Block bootstrap (B=21 days), and Multivariate-row bootstrap.

# %%
integ_df = pd.read_csv(ROOT / "data" / "processed" / "phase5_bootstrap_mode_integrity.csv")
print(integ_df.to_string(index=False))

# %% [markdown]
# ## Panel 3: MC Forward Weight Simulation Loop
# Full 37 RDs x 500 draws = 18,500 QP solves with Ledoit-Wolf covariance.

# %%
loop_df = pd.read_csv(ROOT / "data" / "processed" / "phase5_mc_loop_summary.csv")
print(f"Total RDs audited: {len(loop_df)}")
print(f"Total cap violations: {loop_df['cap_violated'].sum()}")
print(f"Total drift violations: {loop_df['sector_drift_violated'].sum()}")
print(f"Total constraint violations: {loop_df['violations_total'].sum()}")

# %% [markdown]
# ## Panel 4: Convergence Curve across Draw Count K
# Empirical stabilization of centroid weights and 90% confidence interval width change (< 5.0%).

# %%
conv_df = pd.read_csv(ROOT / "data" / "processed" / "phase5_mc_convergence_curve.csv")
print(conv_df.to_string(index=False))

# %% [markdown]
# ## Panel 5: Stability A/B Comparison (Centroid vs Phase 4 RIDGE Point)
# Quantifying turnover reduction and concentration improvement.

# %%
stab_df = pd.read_csv(ROOT / "data" / "processed" / "phase5_stability_ab_vs_phase4_pointestimate.csv")
print(stab_df.to_string(index=False))

# %% [markdown]
# ## Panel 6: Candidate Centroid Comparison Panel
# Evaluated across Mean, Median, and Medoid draw aggregations.

# %%
verdict_df = pd.read_csv(ROOT / "data" / "processed" / "phase5_mc_centroid_verdict.csv")
print(verdict_df.to_string(index=False))

# %% [markdown]
# ## Panel 7: Selected Centroid Weight Heatmap
# Selected 'mean' centroid weights across 37 rebalance dates and 46 constituents.

# %%
sel_weights = pd.read_csv(ROOT / "data" / "processed" / "phase5_selected_centroid_weights.csv")
print(f"Selected weights shape: {sel_weights.shape}")
print(f"Max single-name weight: {sel_weights['weight'].max()*100:.3f}% (Cap: <= 10.0%)")
print(f"Min weight: {sel_weights['weight'].min()*100:.6f}% (Non-negative)")

# %% [markdown]
# ## Panel 8: Final Compliance Audit & Hard Asserts

# %%
from notebooks._run_nb3_validation import main as run_validation
ret = run_validation()
assert ret == 0, "Notebook 3 validation failed!"
