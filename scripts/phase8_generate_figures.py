"""
Phase 8 — Step 8.2: Publication-Quality Figures
================================================
Generates 8 publication-quality PNG figures from Phase 6 & 7 artifacts.

Figures:
  F1  fig1_equity_curves.png       — cumulative equity curves (8 strategies)
  F2  fig2_drawdown_overlay.png    — rolling max-drawdown overlay
  F3  fig3_performance_bar.png     — Sharpe bar chart with DSR annotation
  F4  fig4_jk_heatmap.png          — Jobson-Korkie 8×8 p-value heatmap
  F5  fig5_bootstrap_boxplot.png   — Sharpe distribution across B=200 paths
  F6  fig6_regime_heatmap.png      — 6 regimes × 8 strategies Sharpe heatmap
  F7  fig7_txcost_sensitivity.png  — Sharpe vs txcost (0–50 bps)
  F8  fig8_factor_attribution.png  — 4-factor decomposition stacked bar

Design rules (per roadmap):
  - matplotlib Agg backend (headless, no plt.show())
  - dpi=150, tight_layout()
  - All axes labeled with units
  - Legend inside figure
  - Consistent sans-serif font

Hard assert: all 8 PNGs exist and > 5 KB after completion.
Exit code 0 on success.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────
STRAT_COLORS = {
    "centroid":          "#E63946",   # bold red — primary strategy
    "ridge_linreg":      "#457B9D",   # steel blue
    "rf":                "#2A9D8F",   # teal
    "xgb":               "#E9C46A",   # amber
    "equal_weight":      "#264653",   # dark slate
    "min_variance":      "#A8DADC",   # powder blue
    "risk_parity":       "#F4A261",   # sandy orange
    "classic_max_sharpe":"#6D6875",   # mauve
}
STRAT_LABELS = {
    "centroid":          "MC Centroid ★",
    "ridge_linreg":      "Ridge Linreg",
    "rf":                "Random Forest",
    "xgb":               "XGBoost",
    "equal_weight":      "Equal Weight (1/N)",
    "min_variance":      "Min Variance",
    "risk_parity":       "Risk Parity",
    "classic_max_sharpe":"Classic Max Sharpe",
}
STRATEGIES = list(STRAT_COLORS.keys())

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


def save(fig: plt.Figure, fname: str) -> Path:
    path = FIGS / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"  [SAVED] {fname}  ({size_kb:.1f} KB)")
    return path


# ── F1: Cumulative equity curves ─────────────────────────────────────────────
def fig1_equity_curves() -> None:
    eq = pd.read_csv(DATA / "phase6_all_equities_txadj.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    for strat in STRATEGIES:
        if strat not in eq.columns:
            continue
        lw = 2.5 if strat == "centroid" else 1.2
        alpha = 1.0 if strat == "centroid" else 0.75
        ax.plot(eq.index, eq[strat], color=STRAT_COLORS[strat],
                label=STRAT_LABELS[strat], linewidth=lw, alpha=alpha)

    # COVID crash shading
    ax.axvspan(pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"),
               alpha=0.12, color="gray", label="_COVID crash")

    ax.set_title("Walk-Forward Equity Curves — All 8 Strategies (Dev Window 2015–2023)\n"
                 "10 bps/turn transaction cost applied · Risk-free rate 4% p.a.")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (normalised to 1.0)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "fig1_equity_curves.png")


# ── F2: Rolling max-drawdown overlay ─────────────────────────────────────────
def fig2_drawdown_overlay() -> None:
    eq = pd.read_csv(DATA / "phase6_all_equities_txadj.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    for strat in STRATEGIES:
        if strat not in eq.columns:
            continue
        dd = (eq[strat] / eq[strat].cummax() - 1.0) * 100.0
        lw = 2.5 if strat == "centroid" else 1.0
        alpha = 1.0 if strat == "centroid" else 0.7
        ax.plot(dd.index, dd.values, color=STRAT_COLORS[strat],
                label=STRAT_LABELS[strat], linewidth=lw, alpha=alpha)

    ax.axvspan(pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"),
               alpha=0.15, color="gray", label="_COVID crash")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title("Rolling Maximum Drawdown — All 8 Strategies (Dev Window 2015–2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown from Peak (%)")
    ax.legend(loc="lower left", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "fig2_drawdown_overlay.png")


# ── F3: Sharpe bar chart with DSR annotation ─────────────────────────────────
def fig3_performance_bar() -> None:
    ms = pd.read_csv(RESULTS / "metrics_summary.csv")
    ms = ms.sort_values("sharpe_txadj", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [STRAT_COLORS.get(s, "#888") for s in ms["strategy"]]
    bars = ax.barh(ms["strategy"].map(STRAT_LABELS).fillna(ms["strategy"]),
                   ms["sharpe_txadj"], color=colors, edgecolor="white", height=0.65)

    # Annotate DSR prob
    for bar, (_, row) in zip(bars, ms.iterrows()):
        x = bar.get_width()
        dsr = row["dsr_prob"]
        label = f"DSR={dsr:.3f}"
        offset = 0.01 if x >= 0 else -0.05
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left" if x >= 0 else "right",
                fontsize=7.5, color="black")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Walk-Forward Sharpe Ratio (tx-adj, rf=4%) — Ranked by Strategy\n"
                 "Annotated with Deflated Sharpe Ratio probability (N=24 implicit configs)")
    ax.set_xlabel("Sharpe Ratio (annualised, tx-cost-adjusted)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    save(fig, "fig3_performance_bar.png")


# ── F4: Jobson-Korkie 8×8 heatmap ────────────────────────────────────────────
def fig4_jk_heatmap() -> None:
    jk = pd.read_csv(DATA / "phase6_jk_pairwise_tests.csv")
    strats = STRATEGIES

    # Build symmetric 8×8 p-value matrix
    pmat = pd.DataFrame(np.nan, index=strats, columns=strats)
    for _, row in jk.iterrows():
        s1, s2 = row["strat_1"], row["strat_2"]
        if s1 in strats and s2 in strats:
            pmat.loc[s1, s2] = row["p_value"]
            pmat.loc[s2, s1] = row["p_value"]
    np.fill_diagonal(pmat.values, 1.0)

    # Mask upper triangle
    mask = np.triu(np.ones_like(pmat.values, dtype=bool), k=1)
    pmat_plot = pmat.copy()
    pmat_plot.values[mask] = np.nan

    labels = [STRAT_LABELS.get(s, s) for s in strats]

    fig, ax = plt.subplots(figsize=(9, 7))
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(pmat_plot.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="p-value (Jobson-Korkie)")
    ax.set_xticks(range(len(strats)))
    ax.set_yticks(range(len(strats)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells with p-values; bold significant ones
    for i in range(len(strats)):
        for j in range(len(strats)):
            v = pmat_plot.values[i, j]
            if np.isnan(v):
                continue
            text = f"{v:.3f}"
            weight = "bold" if v < 0.05 else "normal"
            color = "white" if v < 0.3 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7, fontweight=weight, color=color)

    ax.set_title("Jobson-Korkie (Memmel 2003) Pairwise p-Values\n"
                 "Bold = significant at 5%; green = high p (not significant)")
    plt.tight_layout()
    save(fig, "fig4_jk_heatmap.png")


# ── F5: Bootstrap Sharpe box plot ─────────────────────────────────────────────
def fig5_bootstrap_boxplot() -> None:
    bs = pd.read_csv(DATA / "phase7_bootstrap_distributions.csv")
    # Expected columns: strategy, sharpe, cagr, etc. (long format)
    # Pivot to wide: index=path_id, columns=strategy, values=sharpe
    sharpe_col = "sharpe" if "sharpe" in bs.columns else [c for c in bs.columns if "sharpe" in c.lower()][0]
    strat_col = "strategy" if "strategy" in bs.columns else bs.columns[0]

    # Realized Sharpes from metrics summary
    ms = pd.read_csv(RESULTS / "metrics_summary.csv").set_index("strategy")

    fig, ax = plt.subplots(figsize=(11, 5))
    positions = range(len(STRATEGIES))
    data_list = []
    valid_strats = []
    for s in STRATEGIES:
        sub = bs.loc[bs[strat_col] == s, sharpe_col].dropna()
        if len(sub) > 0:
            data_list.append(sub.values)
            valid_strats.append(s)

    bp = ax.boxplot(data_list, positions=list(range(len(valid_strats))),
                    patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))

    for patch, strat in zip(bp["boxes"], valid_strats):
        patch.set_facecolor(STRAT_COLORS.get(strat, "#888"))
        patch.set_alpha(0.85)

    # Overlay realized Sharpe as scatter
    for i, strat in enumerate(valid_strats):
        if strat in ms.index:
            realized = ms.loc[strat, "sharpe_txadj"]
            ax.scatter(i, realized, color="black", zorder=5, s=60, marker="D",
                       label="_realized" if i > 0 else "Realized Sharpe (dev window)")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(valid_strats)))
    ax.set_xticklabels([STRAT_LABELS.get(s, s) for s in valid_strats],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe Ratio (annualised)")
    ax.set_title("Sharpe Ratio Distribution — B=200 Stationary Block Bootstrap Paths\n"
                 "L=19 block length · ◆ = realized Sharpe on dev-window equity curve")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "fig5_bootstrap_boxplot.png")


# ── F6: Regime heatmap ────────────────────────────────────────────────────────
def fig6_regime_heatmap() -> None:
    reg = pd.read_csv(DATA / "phase6_regime_analysis.csv")
    # Pivot: rows=regime_name, cols=strategy, values=sharpe_txadj
    pivot = reg.pivot_table(index="regime_name", columns="strategy",
                            values="sharpe_txadj", aggfunc="first")
    # Reorder columns
    ordered_cols = [s for s in STRATEGIES if s in pivot.columns]
    pivot = pivot[ordered_cols]

    # Reorder rows by regime_id
    regime_order = reg.drop_duplicates("regime_id").sort_values("regime_id")["regime_name"].tolist()
    pivot = pivot.reindex([r for r in regime_order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = pivot.values[np.isfinite(pivot.values)].max()
    vmin = pivot.values[np.isfinite(pivot.values)].min()
    vext = max(abs(vmin), abs(vmax))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vext, vmax=vext, aspect="auto")
    plt.colorbar(im, ax=ax, label="Sharpe Ratio (tx-adj)")

    ax.set_xticks(range(len(ordered_cols)))
    ax.set_xticklabels([STRAT_LABELS.get(s, s) for s in ordered_cols],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(ordered_cols)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) > vext * 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color)

    ax.set_title("Macro-Regime × Strategy Sharpe Heatmap (tx-adj)\n"
                 "6 pre-defined regimes · Green=positive · Red=negative")
    plt.tight_layout()
    save(fig, "fig6_regime_heatmap.png")


# ── F7: Transaction-cost sensitivity ─────────────────────────────────────────
def fig7_txcost_sensitivity() -> None:
    tc = pd.read_csv(DATA / "phase6_txcost_sensitivity.csv")
    cost_levels = sorted(tc["cost_bps"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    for strat in STRATEGIES:
        sub = tc.loc[tc["strategy"] == strat].sort_values("cost_bps")
        if sub.empty:
            continue
        lw = 2.5 if strat == "centroid" else 1.2
        alpha = 1.0 if strat == "centroid" else 0.75
        ax.plot(sub["cost_bps"], sub["sharpe_txadj"],
                color=STRAT_COLORS[strat], label=STRAT_LABELS[strat],
                linewidth=lw, alpha=alpha, marker="o", markersize=4)

    # Mark operating point
    ax.axvline(10, color="black", linestyle="--", linewidth=1.0, label="10 bps operating point")
    ax.set_xlabel("Transaction Cost (bps per unit one-sided turnover)")
    ax.set_ylabel("Sharpe Ratio (tx-adjusted, annualised)")
    ax.set_title("Transaction-Cost Sensitivity — Sharpe vs. Cost Assumption\n"
                 "Cost range: 0–50 bps · Operating assumption: 10 bps (dashed)")
    ax.legend(loc="lower left", framealpha=0.9, ncol=2, fontsize=7.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "fig7_txcost_sensitivity.png")


# ── F8: 4-factor attribution stacked bar ─────────────────────────────────────
def fig8_factor_attribution() -> None:
    fa = pd.read_csv(DATA / "phase6_factor_attribution.csv")
    # Use beta × market return proxy (bps) as contribution proxy
    # We'll show alpha + 4 beta columns in a stacked horizontal bar
    # Normalise to show proportional contribution signs
    strats_present = [s for s in STRATEGIES if s in fa["strategy"].values]
    fa = fa.set_index("strategy").reindex(strats_present)

    factor_cols = ["alpha_ann_bps", "beta_mkt", "beta_smb", "beta_hml", "beta_mom"]
    factor_labels = ["Alpha (bps)", "β MKT", "β SMB", "β HML", "β MOM"]
    factor_colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]

    # For a meaningful stacked bar we show betas (unitless) and alpha in bps on same plot.
    # We split into two subplots: alpha_ann_bps and the 4 betas.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # -- Alpha panel --
    alpha_vals = fa["alpha_ann_bps"].values
    y_pos = np.arange(len(strats_present))
    colors_alpha = [STRAT_COLORS.get(s, "#888") for s in strats_present]
    bars = ax1.barh(y_pos, alpha_vals, color=colors_alpha, edgecolor="white", height=0.65)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([STRAT_LABELS.get(s, s) for s in strats_present], fontsize=8)
    ax1.set_xlabel("4-Factor Alpha (bps p.a.)")
    ax1.set_title("4-Factor Alpha (Annualised bps)")
    ax1.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, alpha_vals):
        x = bar.get_width()
        ax1.text(x + (5 if x >= 0 else -5), bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}", va="center",
                 ha="left" if x >= 0 else "right", fontsize=7.5)

    # -- Beta panel (grouped bars) --
    beta_cols = ["beta_mkt", "beta_smb", "beta_hml", "beta_mom"]
    beta_labels_short = ["β_MKT", "β_SMB", "β_HML", "β_MOM"]
    beta_colors_list = ["#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]
    n_factors = len(beta_cols)
    bar_width = 0.18
    for fi, (bcol, blabel, bcolor) in enumerate(zip(beta_cols, beta_labels_short, beta_colors_list)):
        offsets = y_pos + (fi - n_factors / 2 + 0.5) * bar_width
        ax2.barh(offsets, fa[bcol].values, height=bar_width,
                 color=bcolor, label=blabel, alpha=0.9, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([STRAT_LABELS.get(s, s) for s in strats_present], fontsize=8)
    ax2.set_xlabel("Factor Loading (β)")
    ax2.set_title("4-Factor Betas (MKT, SMB, HML, MOM)")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Fama-French 4-Factor Risk Attribution — Walk-Forward Returns (2015–2023)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save(fig, "fig8_factor_attribution.png")


def main() -> None:
    print("=" * 68)
    print("PHASE 8 — STEP 8.2: Publication-Quality Figures")
    print("=" * 68)

    fig1_equity_curves()
    fig2_drawdown_overlay()
    fig3_performance_bar()
    fig4_jk_heatmap()
    fig5_bootstrap_boxplot()
    fig6_regime_heatmap()
    fig7_txcost_sensitivity()
    fig8_factor_attribution()

    # Hard assert: all 8 PNGs > 5 KB
    expected = [
        "fig1_equity_curves.png", "fig2_drawdown_overlay.png",
        "fig3_performance_bar.png", "fig4_jk_heatmap.png",
        "fig5_bootstrap_boxplot.png", "fig6_regime_heatmap.png",
        "fig7_txcost_sensitivity.png", "fig8_factor_attribution.png",
    ]
    print("\n--- Hard Assert Checks ---")
    for fname in expected:
        p = FIGS / fname
        assert p.exists(), f"[FAIL] Missing: {fname}"
        sz = p.stat().st_size
        assert sz > 5_000, f"[FAIL] {fname} too small: {sz} bytes"
        print(f"  [PASS] {fname}  ({sz/1024:.1f} KB)")

    print("\n" + "=" * 68)
    print(f"STEP 8.2 COMPLETE — {len(expected)} figures written to results/figures/")
    print("=" * 68)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[ASSERTION FAILED] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
