# %% [markdown]
# # Notebook 04 — Walk-Forward Backtesting, Statistical Inference & Robustness
#
# **Phase 6 Work Group 6 Deliverable**
#
# This notebook presents the final out-of-sample walk-forward backtest results,
# multiple testing statistical corrections (Jobson-Korkie, Deflated Sharpe Ratio, PBO),
# four-factor risk decomposition, pre-defined macro regime analysis, and transaction-cost sensitivity.
#
# ### 8 Core Panels:
# 1. Cumulative Equity Curves (all 8 walk-forward strategies, tx-adjusted)
# 2. Rolling Underwater Drawdown Overlay
# 3. Rolling 12-Month Sharpe Ratio Trajectory
# 4. Jobson-Korkie (Memmel 2003) Pairwise Significance Matrix (28 pairs)
# 5. Deflated Sharpe Ratio & Combinatorially Symmetric Cross-Validation (PBO)
# 6. Four-Factor Asset Pricing Decomposition (Market, SMB, HML, MOM)
# 7. Pre-Defined Macro Regime Performance Breakdown (6 Regimes x 8 Strategies)
# 8. Transaction Cost Sensitivity Sweep (0 to 50 bps) & 3 Hard Asserts
#

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

print("Phase 6 environment loaded. Root:", ROOT)

# %% [markdown]
# ## Panel 1: Cumulative Equity Curves (Walk-Forward Out-of-Sample, 10 bps Drag)
# Comparison of MC Centroid vs RIDGE ML point estimate vs classical optimization baselines.

# %%
all_eq = pd.read_csv(ROOT / "data" / "processed" / "phase6_all_equities_txadj.csv", index_col=0, parse_dates=True)
print(f"Loaded {all_eq.shape[1]} equity series across {len(all_eq)} trading days.")
print("Terminal wealth multiples (base = 1.0):")
for col in all_eq.columns:
    print(f"  {col:<20s} : {all_eq[col].iloc[-1]:.4f}")

# %% [markdown]
# ## Panel 2: Rolling Underwater Drawdown Overlay
# Quantifying tail losses and maximum peak-to-trough drawdowns.

# %%
dd_df = (all_eq / all_eq.cummax() - 1.0) * 100.0
print("Maximum Drawdown (%):")
print(dd_df.min().to_string())

# %% [markdown]
# ## Panel 3: Rolling 12-Month Sharpe Ratio Trajectory
# Evaluation of stability across 252-day trailing windows.

# %%
rf_daily = (1.0 + 0.04) ** (1.0 / 252) - 1.0
daily_rets = all_eq.pct_change().dropna()
rolling_excess = daily_rets - rf_daily
rolling_sr = (rolling_excess.rolling(252).mean() / rolling_excess.rolling(252).std(ddof=1)) * np.sqrt(252)
print("Rolling 12-Month Sharpe summary:")
print(rolling_sr.describe().T[["mean", "std", "min", "50%", "max"]].to_string())

# %% [markdown]
# ## Panel 4: Jobson-Korkie (Memmel 2003) Pairwise Significance Matrix
# Asymptotic hypothesis testing across all 28 pairwise strategy differences.

# %%
jk_df = pd.read_csv(ROOT / "data" / "processed" / "phase6_jk_pairwise_tests.csv")
print(f"Evaluated {len(jk_df)} pairwise comparisons.")
print("\nTop Pairwise Comparisons:")
sub_jk = jk_df[((jk_df["strat_1"] == "centroid") | (jk_df["strat_2"] == "centroid"))]
print(sub_jk[["strat_1", "strat_2", "delta_sharpe", "z_stat", "p_value", "sig_at_5pct"]].to_string(index=False))

# %% [markdown]
# ## Panel 5: Deflated Sharpe Ratio & PBO CSCV Distribution
# Multiple-testing adjustment for 24 implicit sequential configurations & combinatorial cross-validation.

# %%
dsr_df = pd.read_csv(ROOT / "data" / "processed" / "phase6_dsr_summary.csv")
pbo_df = pd.read_csv(ROOT / "data" / "processed" / "phase6_pbo_results.csv")
print("Deflated Sharpe Ratio Summary:")
print(dsr_df[["strategy", "sharpe_ann", "e_max_sharpe", "dsr_z", "dsr_prob"]].to_string(index=False))

pbo_rate = float(pbo_df["overfit_flag"].mean())
print(f"\nProbability of Backtest Overfitting (PBO): {pbo_rate:.3f} (S=6 slices, C(6,3)=20 combinations)")
print(f"Median Out-of-Sample Relative Rank: {pbo_df['oos_relative_rank'].median():.2f}")

# %% [markdown]
# ## Panel 6: Four-Factor Asset Pricing Decomposition
# Regressing daily excess returns on Indian Market, SMB, HML, and MOM factor proxies.

# %%
factor_df = pd.read_csv(ROOT / "data" / "processed" / "phase6_factor_attribution.csv")
print("Factor Attribution Table:")
print(factor_df[["strategy", "alpha_ann_bps", "alpha_tstat", "beta_mkt", "beta_smb", "beta_mom", "r2"]].to_string(index=False))

# %% [markdown]
# ## Panel 7: Pre-Defined Macro Regime Performance Breakdown
# 6 non-overlapping historical regimes covering 2015 to 2023.

# %%
regime_df = pd.read_csv(ROOT / "data" / "processed" / "phase6_regime_analysis.csv")
print(f"Loaded {len(regime_df)} regime observations across 6 epochs.")
pivot_sharpe = regime_df.pivot_table(index="regime_name", columns="strategy", values="sharpe_txadj")
print("\nSharpe Ratio by Macro Regime:")
print(pivot_sharpe.to_string())

# %% [markdown]
# ## Panel 8: Transaction Cost Sensitivity Sweep & Headless Validation Audit
# Testing resilience across 0, 5, 10, 20, 30, and 50 bps execution costs, followed by hard asserts.

# %%
txcost_df = pd.read_csv(ROOT / "data" / "processed" / "phase6_txcost_sensitivity.csv")
pivot_cost = txcost_df.pivot_table(index="cost_bps", columns="strategy", values="sharpe_txadj")
print("Sharpe Ratio vs Transaction Cost (bps):")
print(pivot_cost.to_string())

# %%
from notebooks._run_nb4_validation import main as run_validation
ret = run_validation()
assert ret == 0, "Notebook 04 headless validation failed!"
print("\n>>> All 8 panels rendered and 3/3 hard asserts passed successfully! <<<")
