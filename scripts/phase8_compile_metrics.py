"""
Phase 8 — Step 8.1: Consolidated Metrics Summary Table
=======================================================
Reads all Phase 6 and Phase 7 CSV artifacts and assembles a single
authoritative 8-strategy × 16-column metrics table.

Output: results/metrics_summary.csv

Hard asserts:
  1. Exactly 8 rows.
  2. dsr_prob for centroid > 0 (expect ~0.9967).
  3. pbo < 0.50 (expect 0.300).
  4. p7_win_rate_vs_ew for centroid > 0.50 (expect 0.64).

Exit code 0 on success, non-zero on any failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 68)
    print("PHASE 8 — STEP 8.1: Consolidated Metrics Summary Table")
    print("=" * 68)

    # ------------------------------------------------------------------
    # 1. Load Phase 6 performance summary (core metrics)
    # ------------------------------------------------------------------
    perf = pd.read_csv(DATA / "phase6_performance_summary.csv")
    # Rename for final table clarity
    perf_cols = {
        "strategy": "strategy",
        "display_name": "display_name",
        "cagr_pct": "ann_return_pct",
        "ann_vol": "ann_vol",
        "sharpe_txadj": "sharpe_txadj",
        "sortino": "sortino",
        "max_dd_pct": "max_dd_pct",
        "calmar": "calmar",
        "turnover_bps_ann": "ann_turnover_bps",
        "var_95_daily_pct": "var_95_daily_pct",
        "cvar_95_daily_pct": "cvar_95_daily_pct",
    }
    perf = perf.rename(columns={v: k for k, v in {
        "cagr_pct": "ann_return_pct",
        "turnover_bps_ann": "ann_turnover_bps",
    }.items()}).rename(columns={
        "cagr_pct": "ann_return_pct",
        "turnover_bps_ann": "ann_turnover_bps",
    })
    # Simpler rename approach
    perf = perf.rename(columns={
        "cagr_pct": "ann_return_pct",
        "turnover_bps_ann": "ann_turnover_bps",
    })
    print(f"  Phase 6 performance summary: {len(perf)} rows, cols={list(perf.columns)}")

    # ------------------------------------------------------------------
    # 2. DSR probabilities (join on strategy)
    # ------------------------------------------------------------------
    dsr = pd.read_csv(DATA / "phase6_dsr_summary.csv")[["strategy", "dsr_prob"]]
    print(f"  DSR summary: {len(dsr)} rows")

    # ------------------------------------------------------------------
    # 3. Jobson-Korkie: p-value vs EW for each strategy
    # ------------------------------------------------------------------
    jk = pd.read_csv(DATA / "phase6_jk_pairwise_tests.csv")
    # For each strategy, find its p-value when paired against equal_weight
    def get_jk_pval_vs_ew(strategy: str) -> float:
        mask1 = (jk["strat_1"] == strategy) & (jk["strat_2"] == "equal_weight")
        mask2 = (jk["strat_1"] == "equal_weight") & (jk["strat_2"] == strategy)
        rows = jk[mask1 | mask2]
        if rows.empty:
            return float("nan")
        return float(rows.iloc[0]["p_value"])

    # ------------------------------------------------------------------
    # 4. PBO — single aggregate pbo_fraction across all combinations
    # ------------------------------------------------------------------
    pbo_raw = pd.read_csv(DATA / "phase6_pbo_results.csv")
    pbo_val = float(pbo_raw["overfit_flag"].mean())   # fraction of combos marked overfit
    print(f"  PBO fraction (overfit_flag mean): {pbo_val:.4f}")

    # ------------------------------------------------------------------
    # 5. Factor attribution (alpha, beta_mkt)
    # ------------------------------------------------------------------
    fa = pd.read_csv(DATA / "phase6_factor_attribution.csv")[
        ["strategy", "alpha_ann_bps", "beta_mkt"]
    ].rename(columns={"alpha_ann_bps": "alpha_4f_ann_bps"})
    print(f"  Factor attribution: {len(fa)} rows")

    # ------------------------------------------------------------------
    # 6. Phase 7 metric distributions (sharpe_mean, sharpe_p5)
    # ------------------------------------------------------------------
    p7_dist = pd.read_csv(DATA / "phase7_metric_distributions.csv")[
        ["strategy", "sharpe_mean", "sharpe_p5"]
    ].rename(columns={"sharpe_mean": "p7_sharpe_mean", "sharpe_p5": "p7_sharpe_p5"})
    print(f"  P7 metric distributions: {len(p7_dist)} rows")

    # ------------------------------------------------------------------
    # 7. Phase 7 outperformance probabilities vs EW
    # ------------------------------------------------------------------
    p7_prob = pd.read_csv(DATA / "phase7_outperformance_probabilities.csv")
    # Columns: competitor, p_centroid_beats_sharpe, ...
    # We need p_centroid_beats_sharpe FROM centroid's perspective vs each competitor
    # Build a lookup: for each strategy (competitor), what is the probability Centroid beats it?
    # For centroid itself, we need a different approach: P(centroid beats EW)
    centroid_vs_ew_prob = float(
        p7_prob.loc[p7_prob["competitor"] == "equal_weight", "p_centroid_beats_sharpe"].iloc[0]
    )
    print(f"  P(Centroid > EW) Sharpe: {centroid_vs_ew_prob:.3f}")

    # For non-centroid strategies, we can record NaN (probability is centroid-centric)
    # We'll store p7_win_rate_vs_ew = p_centroid_beats_sharpe only for centroid row;
    # for others store NaN since the metric is defined only for centroid.
    # Actually: let's store the outperformance prob of centroid vs each competitor.
    p7_win_lookup = dict(zip(p7_prob["competitor"], p7_prob["p_centroid_beats_sharpe"]))

    # ------------------------------------------------------------------
    # 8. Assemble final table
    # ------------------------------------------------------------------
    strategies = list(perf["strategy"])
    rows = []
    for _, row in perf.iterrows():
        strat = row["strategy"]
        dsr_row = dsr.loc[dsr["strategy"] == strat]
        fa_row = fa.loc[fa["strategy"] == strat]
        p7_row = p7_dist.loc[p7_dist["strategy"] == strat]

        d = {
            "strategy": strat,
            "display_name": row.get("display_name", strat),
            "ann_return_pct": row.get("ann_return_pct", row.get("cagr_pct", float("nan"))),
            "ann_vol": row["ann_vol"],
            "sharpe_txadj": row["sharpe_txadj"],
            "sortino": row["sortino"],
            "max_dd_pct": row["max_dd_pct"],
            "calmar": row["calmar"],
            "ann_turnover_bps": row.get("ann_turnover_bps", row.get("turnover_bps_ann", float("nan"))),
            "var_95_daily_pct": row.get("var_95_daily_pct", float("nan")),
            "cvar_95_daily_pct": row.get("cvar_95_daily_pct", float("nan")),
            "dsr_prob": float(dsr_row["dsr_prob"].iloc[0]) if not dsr_row.empty else float("nan"),
            "jk_pval_vs_ew": get_jk_pval_vs_ew(strat),
            "pbo": pbo_val,  # single aggregate for the backtest suite
            "alpha_4f_ann_bps": float(fa_row["alpha_4f_ann_bps"].iloc[0]) if not fa_row.empty else float("nan"),
            "beta_mkt": float(fa_row["beta_mkt"].iloc[0]) if not fa_row.empty else float("nan"),
            "p7_sharpe_mean": float(p7_row["p7_sharpe_mean"].iloc[0]) if not p7_row.empty else float("nan"),
            "p7_sharpe_p5": float(p7_row["p7_sharpe_p5"].iloc[0]) if not p7_row.empty else float("nan"),
            # P(centroid > this competitor): only populated for strategies that appear in p7_prob
            "p7_win_rate_centroid_vs": p7_win_lookup.get(strat, float("nan")),
        }
        rows.append(d)

    summary = pd.DataFrame(rows)
    # Add explicit p7_win_rate_vs_ew for centroid row (the key gate metric)
    summary["p7_win_rate_vs_ew"] = summary["strategy"].map(
        lambda s: centroid_vs_ew_prob if s == "centroid" else p7_win_lookup.get(s, float("nan"))
    )

    # ------------------------------------------------------------------
    # 9. Hard asserts
    # ------------------------------------------------------------------
    print("\n--- Hard Assert Checks ---")

    n_rows = len(summary)
    assert n_rows == 8, f"[FAIL] Expected 8 rows, got {n_rows}"
    print(f"  [PASS] Row count = {n_rows}")

    centroid_row = summary.loc[summary["strategy"] == "centroid"]
    assert not centroid_row.empty, "[FAIL] Centroid row missing from summary"
    dsr_centroid = float(centroid_row["dsr_prob"].iloc[0])
    assert dsr_centroid > 0, f"[FAIL] Centroid DSR prob = {dsr_centroid} <= 0"
    print(f"  [PASS] Centroid DSR prob = {dsr_centroid:.6f} > 0")

    assert pbo_val < 0.50, f"[FAIL] PBO = {pbo_val:.4f} >= 0.50"
    print(f"  [PASS] PBO = {pbo_val:.4f} < 0.50")

    win_rate = float(centroid_row["p7_win_rate_vs_ew"].iloc[0])
    assert win_rate > 0.50, f"[FAIL] P7 win-rate vs EW = {win_rate:.3f} <= 0.50"
    print(f"  [PASS] P7 win-rate (Centroid > EW Sharpe) = {win_rate:.3f} > 0.50")

    # Check no NaN in core metric columns
    core_cols = ["ann_return_pct", "ann_vol", "sharpe_txadj", "max_dd_pct", "dsr_prob"]
    for col in core_cols:
        n_nan = summary[col].isna().sum()
        assert n_nan == 0, f"[FAIL] NaN in column '{col}': {n_nan} rows"
    print(f"  [PASS] No NaN in core metric columns: {core_cols}")

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    out_path = RESULTS / "metrics_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  [SAVED] {out_path}  ({len(summary)} rows × {len(summary.columns)} cols)")

    # Print the headline table
    print("\n  === HEADLINE PERFORMANCE TABLE ===")
    display_cols = ["strategy", "ann_return_pct", "sharpe_txadj", "max_dd_pct", "dsr_prob", "pbo", "p7_win_rate_vs_ew"]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 68)
    print("STEP 8.1 COMPLETE — results/metrics_summary.csv written")
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
