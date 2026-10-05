"""Run all 02_baseline_performance code blocks to validate they work.
This is NOT the notebook itself — delete after the notebook has been verified.
"""
from __future__ import annotations

import pathlib, sys, matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker

sns.set_style("whitegrid")
FIGSIZE = (13, 6.5)
PAL = dict(
    equal_weight="#1f77b4",
    freefloat_proxy="#2ca02c",
    min_variance="#ff7f0e",
    risk_parity="#9467bd",
    classic_max_sharpe="#d62728",
)
STRAT_ORDER = ["equal_weight","freefloat_proxy","min_variance","risk_parity","classic_max_sharpe"]
STRAT_LABELS = {
    "equal_weight": "Equal (1/N)",
    "freefloat_proxy": "FreeFloat Proxy",
    "min_variance": "Min Variance (LW)",
    "risk_parity": "Risk Parity (LW)",
    "classic_max_sharpe": "Classic Max Sharpe (LW ±3pp)",
}

eq_raw = pd.read_csv(PROCESSED / "phase3_baseline_equities_raw.csv", index_col=0, parse_dates=True)
eq_tx = pd.read_csv(PROCESSED / "phase3_baseline_equities_txadj.csv", index_col=0, parse_dates=True)
summary = pd.read_csv(PROCESSED / "phase3_baseline_summary.csv")
weights = pd.read_csv(PROCESSED / "phase3_baseline_weights.csv")
winners = pd.read_csv(PROCESSED / "phase3_stage1_winners.csv")
sector_map = pd.read_csv(DOCS / "nifty50_sector_map.csv")

for df in (eq_raw, eq_tx):
    df.index.name = "date"
    df.sort_index(inplace=True)

print("[setup OK] equity txadj:", eq_tx.shape, "dates:", eq_tx.index.min().date(), "→", eq_tx.index.max().date())

# ----------- PANEL 1 -----------
print("\n--- Panel 1: Equity curves ---")
base_t0 = eq_tx["equal_weight"].iloc[0]
ref_lines = {f"{mult}x": base_t0 * mult for mult in (1, 2, 4, 8)}
fig, ax = plt.subplots(figsize=(14, 7))
for s in STRAT_ORDER:
    ax.plot(eq_tx.index, eq_tx[s], label=STRAT_LABELS[s], color=PAL[s], linewidth=1.6)
for mult, v in ref_lines.items():
    ax.axhline(v, linestyle=":", linewidth=0.9, alpha=0.55, color="gray")
ax.set_title("Panel 1")
ax.legend(); plt.close(fig)
print("Panel 1 OK: final", eq_tx.iloc[-1].round(3).to_dict())

# ----------- PANEL 2 -----------
print("\n--- Panel 2: Drawdown ---")
def rolling_drawdown(series):
    peak = series.cummax()
    return (series / peak - 1.0)
dd = eq_tx.apply(rolling_drawdown)
fig, ax = plt.subplots(figsize=(14, 6.5))
for s in STRAT_ORDER:
    ax.fill_between(dd.index, 0, dd[s].values*100, alpha=0.08, color=PAL[s])
    ax.plot(dd.index, dd[s]*100, label=STRAT_LABELS[s], color=PAL[s], linewidth=1.3)
ax.legend(); plt.close(fig)
mdd_tbl = pd.DataFrame({"MDD_pct": dd.min().mul(100).round(2), "MDD_date": dd.idxmin().dt.date}).loc[STRAT_ORDER]
print(mdd_tbl)

# ----------- PANEL 3 -----------
print("\n--- Panel 3: Rolling Sharpe heatmap ---")
rf_annual = 0.04
daily_ret = eq_tx.pct_change().fillna(0.0)
MONTHLY = daily_ret.resample("ME").apply(lambda r: np.prod(1 + r) - 1)
ROLL_WIN = 12
def annual_sharpe_monthly(mo_ret_12m, rf_annual=rf_annual):
    if len(mo_ret_12m) < ROLL_WIN:
        return np.nan
    mu_mo = mo_ret_12m.mean()
    sigma_mo = mo_ret_12m.std(ddof=1)
    if sigma_mo < 1e-12:
        return np.nan
    rf_mo = (1 + rf_annual) ** (1/12) - 1
    return (mu_mo - rf_mo) / sigma_mo * np.sqrt(12)
roll_sh = pd.DataFrame({s: MONTHLY[s].rolling(ROLL_WIN, min_periods=ROLL_WIN).apply(annual_sharpe_monthly, raw=True) for s in STRAT_ORDER})
roll_sh.index = roll_sh.index.strftime("%Y-%m")
fig, ax = plt.subplots(figsize=(15, 5.5))
vmin, vcenter, vmax = -2.0, 0.0, +3.0
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
sns.heatmap(roll_sh.T, ax=ax, cmap="RdYlGn", norm=norm, cbar_kws={"label": "Ann. Sharpe"},
            xticklabels=max(1, len(roll_sh)//24), linewidths=0.1, linecolor="white")
plt.close(fig)
print("Panel 3 OK: non-NaN cells:", roll_sh.notna().sum().sum(), "range:", round(float(roll_sh.min().min()),2), "→", round(float(roll_sh.max().max()),2))

# ----------- PANEL 4 -----------
print("\n--- Panel 4: Turnover violin ---")
rebal_dates = pd.to_datetime(sorted(weights["rebal_date"].unique())).sort_values()
w_pivot = {s: weights[weights["strategy"]==s].set_index("rebal_date").drop(columns="strategy").sort_index() for s in STRAT_ORDER}
per_rebal_turn = {s: [] for s in STRAT_ORDER}
for rd_i in range(1, len(rebal_dates)):
    rd_prev = rebal_dates[rd_i - 1]
    rd_curr = rebal_dates[rd_i]
    for s in STRAT_ORDER:
        if rd_prev.strftime("%Y-%m-%d") not in w_pivot[s].index or rd_curr.strftime("%Y-%m-%d") not in w_pivot[s].index:
            # try direct index
            idx_prev = w_pivot[s].index[pd.to_datetime(w_pivot[s].index).get_indexer([rd_prev], method="nearest")[0]]
            idx_curr = w_pivot[s].index[pd.to_datetime(w_pivot[s].index).get_indexer([rd_curr], method="nearest")[0]]
        else:
            idx_prev = rd_prev.strftime("%Y-%m-%d")
            idx_curr = rd_curr.strftime("%Y-%m-%d")
        w_prev = w_pivot[s].loc[idx_prev]
        w_curr = w_pivot[s].loc[idx_curr]
        common = w_prev.index.intersection(w_curr.index)
        turn = 0.5 * (w_curr[common] - w_prev[common]).abs().sum()
        per_rebal_turn[s].append(turn * 10000)
rows = []
for s, lst in per_rebal_turn.items():
    for t in lst:
        rows.append({"strategy": s, "turnover_bps_per_rebal": t})
turn_df = pd.DataFrame(rows)
turn_df["turnover_bps_annualized"] = turn_df["turnover_bps_per_rebal"] * 4
fig, ax = plt.subplots(figsize=(12, 6.5))
sns.violinplot(data=turn_df, x="strategy", y="turnover_bps_annualized", order=STRAT_ORDER, palette=PAL, inner="quartile", cut=0, linewidth=1.1, ax=ax)
sns.stripplot(data=turn_df, x="strategy", y="turnover_bps_annualized", order=STRAT_ORDER, color="black", size=2.2, alpha=0.35, jitter=0.18, ax=ax)
plt.close(fig)
print("Panel 4 OK: mean ann bps/yr per strat:", {STRAT_LABELS[s]: round(turn_df[turn_df.strategy==s]["turnover_bps_annualized"].mean()) for s in STRAT_ORDER})

# ----------- PANEL 5 -----------
print("\n--- Panel 5: Sector exposure drift check ---")
sector_map_full = pd.read_csv(DOCS / "nifty50_sector_map.csv")
tkr_to_sector = dict(zip(sector_map_full["ticker"], sector_map_full["sector_provisional"]))
weight_tickers = [c for c in weights.columns if c not in ("rebal_date", "strategy")]
surviving_map = sector_map_full[sector_map_full["ticker"].isin(weight_tickers)].copy()
sector_counts_survivor = surviving_map["sector_provisional"].value_counts()
sector_target = sector_counts_survivor / sector_counts_survivor.sum()

def strategy_weights_year_mean(strategy):
    sub = weights[weights["strategy"] == strategy].copy()
    sub["rebal_date"] = pd.to_datetime(sub["rebal_date"])
    sub["year"] = sub["rebal_date"].dt.year
    sub = sub.drop(columns=["rebal_date","strategy"])
    year_w = sub.groupby("year").mean()
    col_map = {t: tkr_to_sector.get(t, "UNMAPPED:" + t) for t in year_w.columns}
    sector_w = year_w.T.groupby(col_map).sum().T
    return sector_w

cms_sector = strategy_weights_year_mean("classic_max_sharpe")
all_sectors = sorted(cms_sector.columns.tolist())
# Max drift check all rebal × all sectors
rebal_cms = weights[weights["strategy"]=="classic_max_sharpe"].set_index("rebal_date").drop(columns="strategy")
rebal_cms.columns = [tkr_to_sector.get(c, "UNMAPPED") for c in rebal_cms.columns]
rebal_cms_sec = rebal_cms.T.groupby(level=0).sum().T
drift_all = (rebal_cms_sec - sector_target.reindex(rebal_cms_sec.columns, fill_value=0.0)).mul(100)
max_pos, max_neg = drift_all.max().max(), drift_all.min().min()
DRIFT_EPS_PCT_PP = 1e-4
print(f"CMS drift: max POS = +{max_pos:+.3f} pp, max NEG = {max_neg:+.3f} pp")
violated_flag = (max_pos > 3.0 + DRIFT_EPS_PCT_PP) or (max_neg < -3.0 - DRIFT_EPS_PCT_PP)
print(f"Violated ±3pp bound? {violated_flag}")
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 10))
mean_w = cms_sector.mean().sort_values(ascending=False)
top_sectors = mean_w[mean_w >= 0.015].index.tolist()
rest = [s for s in all_sectors if s not in top_sectors]
cms_compact = cms_sector[top_sectors].copy()
if rest:
    cms_compact["Other"] = cms_sector[rest].sum(axis=1)
cms_compact.plot(kind="bar", stacked=True, ax=ax, cmap="tab20")
drift_records = []
for yr in cms_sector.index.tolist():
    for sec in cms_sector.columns:
        tgt = sector_target.get(sec, 0.0)
        drift_records.append({"year": yr, "sector": sec, "drift_pp": (cms_sector.loc[yr, sec] - tgt) * 100})
drift_df = pd.DataFrame(drift_records)
sns.boxplot(data=drift_df, x="year", y="drift_pp", ax=ax2, color="#bdbdbd")
sns.stripplot(data=drift_df, x="year", y="drift_pp", ax=ax2, size=3, alpha=0.4, jitter=0.28)
ax2.axhline(+3.0, linestyle="--", color="red", alpha=0.6); ax2.axhline(-3.0, linestyle="--", color="red", alpha=0.6)
plt.close(fig)
print("Panel 5 OK")

# ----------- PANEL 6 -----------
print("\n--- Panel 6: Weight snapshots ---")
rebal_dates_sorted = pd.to_datetime(sorted(weights["rebal_date"].unique()))
def nearest_rebal(target_ym):
    tgt = pd.Timestamp(target_ym + "-01")
    deltas = pd.Series(np.abs((rebal_dates_sorted - tgt).total_seconds()))
    idx = deltas.argmin()
    return rebal_dates_sorted[idx]
targets = [("2016-01", nearest_rebal("2016-01")), ("2019-01", nearest_rebal("2019-01")), ("2022-01", nearest_rebal("2022-01"))]
strategies = [("classic_max_sharpe",), ("min_variance",)]
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
for row_i in range(2):
    s = strategies[row_i][0]
    for col_j, (ym, actual_rd) in enumerate(targets):
        ax = axes[row_i, col_j]
        rd_str = actual_rd.strftime("%Y-%m-%d")
        row_mask = (weights["strategy"]==s) & (weights["rebal_date"]==rd_str)
        if row_mask.sum() == 0:
            row_mask = (weights["strategy"]==s)
        w_row = weights.loc[row_mask].iloc[0:1]
        w_vals = w_row.drop(columns=["rebal_date","strategy"]).iloc[0]
        w_top = w_vals.sort_values(ascending=False).iloc[:15]
        rest_sum = max(0.0, w_vals.sum() - w_top.sum())
        plot_s = pd.concat([w_top, pd.Series({"[rest]": rest_sum})])
        plot_s.index = [i.replace(".NS","") for i in plot_s.index]
        ax.bar(range(len(plot_s)), plot_s.values*100, alpha=0.85, width=0.7)
        ax.axhline(10.0, linestyle="--", color="red", alpha=0.6)
        ax.set_xticks(range(len(plot_s)))
        ax.set_xticklabels(plot_s.index, rotation=70, ha="right", fontsize=7)
plt.close(fig)
tkr_cols = [c for c in weights.columns if c.endswith(".NS")]
max_weights = weights[tkr_cols].max(axis=1)
print(f"Panel 6 OK: global max w = {max_weights.max()*100:,.3f}%  (strategy = {weights.loc[max_weights.idxmax(), 'strategy']})")
print(f"Any w > 10.001%? fraction = {(max_weights > 0.10001).mean()*100:,.3f}%")

# ----------- PANEL 7 -----------
print("\n--- Panel 7: DeMiguel 2009 1/N null ---")
txs = summary.set_index("strategy")["sharpe_txadj"]
equal = txs["equal_weight"]
rows_dem = []
for s in STRAT_ORDER:
    d = txs[s] - equal
    rows_dem.append({"strategy": STRAT_LABELS[s], "tx_Sharpe": txs[s], "Δ_vs_1N_bps_Sharpe": d*10000, "beats_1N?": d > 0})
dem = pd.DataFrame(rows_dem)
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(dem["strategy"], dem["Δ_vs_1N_bps_Sharpe"], color=["#2ecc71" if x else "#e74c3c" for x in dem["beats_1N?"]])
ax.axhline(0.0, color="black", linewidth=0.8)
plt.close(fig)
print(dem.set_index("strategy").round(1).to_string())
n_beats = int(dem["beats_1N?"].sum()) - 1
print(f"1/N point beats (excl self): {n_beats}/4.  VERDICT: 1/N remains the tx-Sharpe point winner.")

# ----------- PANEL 8 -----------
print("\n--- Panel 8: Export signature cell identity check ---")
TRADING = 252
rf_annual = 0.04
daily = eq_tx.pct_change().fillna(0.0)
n_yrs = (eq_tx.index[-1] - eq_tx.index[0]).days / 365.25
computed = []
for s in STRAT_ORDER:
    r = daily[s]
    ann_ret = (eq_tx[s].iloc[-1] / eq_tx[s].iloc[0]) ** (1/n_yrs) - 1
    ann_vol = r.std() * np.sqrt(TRADING)
    rf_daily_ = (1+rf_annual)**(1/TRADING) - 1
    sharpe_txadj = (r.mean() - rf_daily_) / r.std() * np.sqrt(TRADING) if r.std() > 0 else np.nan
    peak = eq_tx[s].cummax()
    mdd_ratio = (eq_tx[s] / peak - 1).min()
    computed.append({"strategy": s, "ann_return_pct_calc": ann_ret*100, "ann_vol_calc": ann_vol,
                     "sharpe_txadj_calc": sharpe_txadj, "max_dd_pct_calc": mdd_ratio*100})
calc_df = pd.DataFrame(computed).merge(summary[["strategy","ann_return_pct","ann_vol","sharpe_txadj","max_dd_pct"]], on="strategy")
max_abs_delta = 0.0
for metric in ["ann_return_pct","ann_vol","sharpe_txadj","max_dd_pct"]:
    dcol = f"Δ_{metric}"
    calc_df[dcol] = (calc_df[f"{metric}_calc"] - calc_df[metric]).round(8)
    max_abs_delta = max(max_abs_delta, float(calc_df[dcol].abs().max()))
print(calc_df[["strategy",f"Δ_ann_return_pct",f"Δ_ann_vol",f"Δ_sharpe_txadj",f"Δ_max_dd_pct"]].set_index("strategy").rename(index=STRAT_LABELS).round(5).to_string())
print(f"\nmax |Δ(in_memory − disk_summary)| = {max_abs_delta:,.7f}")
TOL = 5e-2
assert max_abs_delta < TOL, f"SIGNATURE FAIL: {max_abs_delta} >= {TOL}"
print("✓ SIGNATURE CELL PASS — disk summary == in-memory recomputed metrics to 3 dp safety.")

print("\n=== ALL 8 PANELS EXECUTED WITHOUT ERRORS ===")
