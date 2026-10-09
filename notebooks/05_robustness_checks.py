# %% [markdown]
# # Notebook 05: Robustness & Synthetic Stress Testing
# 
# **Author:** Quant Portfolio Optimization Research Pipeline
# **Phase:** 7 (Week 14)
# **Focus:** Block Bootstrap Resampling, Empirical Distributions across Synthetic Histories,
# Outperformance Probabilities, and Realized Volatility Regime Analysis.
# 
# ---
# 
# ## Overview
# This notebook validates the statistical robustness of the Monte Carlo Resampled Centroid
# portfolio against 7 competing strategies across $B = 200$ synthetic market histories.
# Synthetic histories are generated via Stationary Block Bootstrap ($L = 19$), preserving
# contemporaneous cross-asset correlation and volatility clustering.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"

# Set visualization style
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120

# %% [markdown]
# ### Panel 5.1: Autocorrelation Function (ACF) of Squared Returns & Block Length Selection
# Demonstrates empirical volatility clustering decay and determines the optimal block length $L$.

# %%
acf_csv = PROC / "phase7_acf_squared_returns.csv"
block_len_csv = PROC / "phase7_block_length.csv"

acf_df = pd.read_csv(acf_csv)
bl_df = pd.read_csv(block_len_csv)
chosen_L = int(bl_df["chosen_L"].iloc[0])
ci_val = float(bl_df["ci_threshold"].iloc[0])

fig, ax = plt.subplots(figsize=(10, 5))
ax.stem(acf_df["lag"][1:], acf_df["acf"][1:], basefmt="k-", linefmt="steelblue", markerfmt="o")
ax.axhline(ci_val, color="crimson", linestyle="--", label=f"95% CI (+1.96/√T = {ci_val:.4f})")
ax.axhline(-ci_val, color="crimson", linestyle="--")
ax.axvline(chosen_L, color="forestgreen", linestyle="-.", label=f"Chosen Block Length L = {chosen_L}")
ax.set_title("Panel 5.1: ACF of Market Squared Returns (Dev Window 2015–2023)")
ax.set_xlabel("Lag (Trading Days)")
ax.set_ylabel("Autocorrelation")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Panel 5.2: Bootstrap Distribution of Annualized Sharpe Ratios (8 Strategies)
# Box plots showing the full empirical distribution of Sharpe ratios across $B = 200$ synthetic histories.

# %%
dist_csv = PROC / "phase7_bootstrap_distributions.csv"
dist_df = pd.read_csv(dist_csv)

# Sort strategies by median Sharpe
order_sharpe = dist_df.groupby("strategy")["sharpe"].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(
    data=dist_df,
    x="strategy",
    y="sharpe",
    order=order_sharpe,
    palette="Blues_r",
    ax=ax,
)
ax.axhline(0.0, color="black", linestyle="--", alpha=0.6)
ax.set_title("Panel 5.2: Annualized Sharpe Ratio Distributions across B=200 Synthetic Histories")
ax.set_xlabel("Strategy")
ax.set_ylabel("Sharpe Ratio (10 bps tx drag)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Panel 5.3: Bootstrap Distribution of Compound Annual Growth Rate (CAGR)

# %%
order_cagr = dist_df.groupby("strategy")["cagr"].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(
    data=dist_df,
    x="strategy",
    y="cagr",
    order=order_cagr,
    palette="Greens_r",
    ax=ax,
)
ax.set_title("Panel 5.3: CAGR (%) Distributions across B=200 Synthetic Histories")
ax.set_xlabel("Strategy")
ax.set_ylabel("CAGR (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Panel 5.4: Bootstrap Distribution of Maximum Drawdown (MDD)

# %%
order_mdd = dist_df.groupby("strategy")["max_drawdown"].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(
    data=dist_df,
    x="strategy",
    y="max_drawdown",
    order=order_mdd,
    palette="Reds",
    ax=ax,
)
ax.set_title("Panel 5.4: Maximum Drawdown (%) Distributions across B=200 Synthetic Histories")
ax.set_xlabel("Strategy")
ax.set_ylabel("Max Drawdown (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Panel 5.5: Outperformance Win-Rate Bar Chart
# Fraction of synthetic paths where Centroid beats each competitor strategy.

# %%
prob_csv = PROC / "phase7_outperformance_probabilities.csv"
prob_df = pd.read_csv(prob_csv)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    prob_df["competitor"],
    prob_df["p_centroid_beats_sharpe"] * 100.0,
    color="steelblue",
    edgecolor="black",
)
ax.axhline(50.0, color="crimson", linestyle="--", label="Neutral Win-Rate (50%)")
ax.set_title("Panel 5.5: Centroid Sharpe Win-Rate P(Centroid > Competitor) across B=200 Paths")
ax.set_ylabel("Win-Rate (%)")
ax.set_ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

plt.xticks(rotation=20)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Panel 5.6: Volatility-Regime Median Sharpe Heatmap
# Performance across Low, Mid, and High realized volatility terciles.

# %%
regime_csv = PROC / "phase7_regime_vol_tercile.csv"
regime_df = pd.read_csv(regime_csv)

heatmap_data = regime_df.pivot(index="regime", columns="strategy", values="sharpe_p50")
heatmap_data = heatmap_data.reindex(index=["low_vol", "mid_vol", "high_vol"])

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    cbar_kws={"label": "Median Sharpe"},
    ax=ax,
)
ax.set_title("Panel 5.6: Median Sharpe Ratio across Synthetic Paths by Realized Volatility Tercile")
ax.set_xlabel("Strategy")
ax.set_ylabel("Volatility Regime")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Panel 5.7: Win-Rate Estimate Convergence vs Bootstrap Sample Size B
# Demonstrates statistical stability of the Centroid vs Equal Weight win-rate estimate.

# %%
piv_sh = dist_df.pivot(index="path_id", columns="strategy", values="sharpe")
cent_beats_ew = (piv_sh["centroid"] > piv_sh["equal_weight"]).astype(float)
cum_win_rate = cent_beats_ew.expanding().mean() * 100.0

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(cum_win_rate.index + 1, cum_win_rate.values, color="navy", lw=2, label="Cumulative Win-Rate P(Centroid > EW)")
ax.axhline(50.0, color="crimson", linestyle="--", label="50% Benchmark")
ax.axhline(cum_win_rate.iloc[-1], color="darkgreen", linestyle=":", label=f"Final Win-Rate ({cum_win_rate.iloc[-1]:.1f}%)")
ax.set_title("Panel 5.7: Convergence of Win-Rate Estimate vs Bootstrap Paths B")
ax.set_xlabel("Number of Synthetic Paths (B)")
ax.set_ylabel("Cumulative Win-Rate (%)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Panel 5.8: Summary Compliance Table & Hard Asserts Audit

# %%
cent_sharpe_p5 = float(dist_df[dist_df["strategy"] == "centroid"]["sharpe"].quantile(0.05))
ew_win_rate = float(prob_df[prob_df["competitor"] == "equal_weight"]["p_centroid_beats_sharpe"].iloc[0])
strat_count = dist_df["strategy"].nunique()

compliance_data = [
    {
        "Assertion": "Hard Assert 1: P(Centroid Sharpe > EW Sharpe) > 50%",
        "Observed": f"{ew_win_rate * 100:.1f}%",
        "Threshold": "> 50.0%",
        "Status": "PASS" if ew_win_rate > 0.50 else "FAIL",
    },
    {
        "Assertion": "Hard Assert 2: Centroid P5-Sharpe (worst 5%) > 0.0",
        "Observed": f"{cent_sharpe_p5:.4f}",
        "Threshold": "> 0.0000",
        "Status": "PASS" if cent_sharpe_p5 > 0.0 else "FAIL",
    },
    {
        "Assertion": "Hard Assert 3: Exactly 8 strategies in distribution table",
        "Observed": f"{strat_count} strategies",
        "Threshold": "== 8 strategies",
        "Status": "PASS" if strat_count == 8 else "FAIL",
    },
]

compliance_df = pd.DataFrame(compliance_data)
print("\n" + "=" * 72)
print("P A N E L   5 . 8   C O M P L I A N C E   S U M M A R Y")
print("=" * 72)
print(compliance_df.to_string(index=False))
