from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DISTRIBUTIONS_CSV = ROOT / "data" / "processed" / "phase7_bootstrap_distributions.csv"
OUT_METRIC_DIST_CSV = ROOT / "data" / "processed" / "phase7_metric_distributions.csv"
OUT_OUTPERF_PROB_CSV = ROOT / "data" / "processed" / "phase7_outperformance_probabilities.csv"

METRICS = ["sharpe", "cagr", "annualized_vol", "sortino", "max_drawdown", "calmar"]


def compute_distributional_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    assert DISTRIBUTIONS_CSV.exists(), f"Missing {DISTRIBUTIONS_CSV}. Run Step 7.2 first."
    df = pd.read_csv(DISTRIBUTIONS_CSV)
    
    strategies = sorted(df["strategy"].unique().tolist())
    assert len(strategies) == 8, f"Expected 8 strategies, got {len(strategies)}"
    
    # 1. Wide-format metric distributions summary (8 strategies x 30 stats)
    rows_summary: list[dict] = []
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        strat_dict: dict[str, object] = {"strategy": strat}
        for m in METRICS:
            vals = sub[m].values.astype(float)
            strat_dict[f"{m}_mean"] = float(np.mean(vals))
            strat_dict[f"{m}_std"] = float(np.std(vals, ddof=1))
            strat_dict[f"{m}_p5"] = float(np.percentile(vals, 5))
            strat_dict[f"{m}_p50"] = float(np.percentile(vals, 50))
            strat_dict[f"{m}_p95"] = float(np.percentile(vals, 95))
        rows_summary.append(strat_dict)
        
    summary_df = pd.DataFrame(rows_summary)
    
    # 2. Outperformance probabilities (Centroid vs each competitor across B paths)
    piv_sharpe = df.pivot(index="path_id", columns="strategy", values="sharpe")
    piv_cagr = df.pivot(index="path_id", columns="strategy", values="cagr")
    piv_sortino = df.pivot(index="path_id", columns="strategy", values="sortino")
    piv_calmar = df.pivot(index="path_id", columns="strategy", values="calmar")
    
    assert "centroid" in piv_sharpe.columns, "Centroid strategy missing from pivot table!"
    cent_sharpe = piv_sharpe["centroid"]
    cent_cagr = piv_cagr["centroid"]
    cent_sortino = piv_sortino["centroid"]
    cent_calmar = piv_calmar["centroid"]
    
    competitors = [s for s in strategies if s != "centroid"]
    prob_rows: list[dict] = []
    for comp in competitors:
        p_sharpe = float((cent_sharpe > piv_sharpe[comp]).mean())
        p_cagr = float((cent_cagr > piv_cagr[comp]).mean())
        p_sortino = float((cent_sortino > piv_sortino[comp]).mean())
        p_calmar = float((cent_calmar > piv_calmar[comp]).mean())
        prob_rows.append(
            {
                "competitor": comp,
                "p_centroid_beats_sharpe": p_sharpe,
                "p_centroid_beats_cagr": p_cagr,
                "p_centroid_beats_sortino": p_sortino,
                "p_centroid_beats_calmar": p_calmar,
            }
        )
        
    prob_df = pd.DataFrame(prob_rows)
    return summary_df, prob_df


def main() -> None:
    print("=" * 72)
    print("P H A S E  7  --  S T E P  7 . 3   D I S T R I B U T I O N   A N A L Y S I S")
    print("=" * 72)
    
    summary_df, prob_df = compute_distributional_summaries()
    
    summary_df.to_csv(OUT_METRIC_DIST_CSV, index=False)
    print(f"[7.3] Saved metric distributions -> {OUT_METRIC_DIST_CSV}")
    print(f"      Shape: {summary_df.shape} (8 strategies x {summary_df.shape[1] - 1} stat cols)")
    
    prob_df.to_csv(OUT_OUTPERF_PROB_CSV, index=False)
    print(f"[7.3] Saved outperformance probabilities -> {OUT_OUTPERF_PROB_CSV}")
    print("\n[7.3] Centroid Win-Rate vs Competitors across B Synthetic Paths:")
    print(prob_df.to_string(index=False))
    
    # Exit criteria assertions
    ew_row = prob_df[prob_df["competitor"] == "equal_weight"]
    assert len(ew_row) == 1, "Equal Weight missing from probability table!"
    p_vs_ew = ew_row["p_centroid_beats_sharpe"].iloc[0]
    
    cent_row = summary_df[summary_df["strategy"] == "centroid"]
    cent_p5_sharpe = cent_row["sharpe_p5"].iloc[0]
    
    print(f"\n[7.3] Hard Assert 1: P(Centroid Sharpe > EW Sharpe) = {p_vs_ew:.4f} (> 0.50)")
    assert p_vs_ew > 0.50, f"Failed Hard Assert 1: P(Centroid > EW) = {p_vs_ew} <= 0.50"
    print("      >>> HARD ASSERT 1 PASS <<<")
    
    print(f"[7.3] Hard Assert 2: Centroid P5-Sharpe = {cent_p5_sharpe:.4f} (> 0.0)")
    assert cent_p5_sharpe > 0.0, f"Failed Hard Assert 2: Centroid P5-Sharpe = {cent_p5_sharpe} <= 0.0"
    print("      >>> HARD ASSERT 2 PASS <<<")
    
    print("\n[PASS] Step 7.3 distribution analysis verified.")


if __name__ == "__main__":
    main()
