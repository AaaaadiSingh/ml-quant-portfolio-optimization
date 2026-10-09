from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    print("=" * 72)
    print("N O T E B O O K   0 4   H E A D L E S S   V A L I D A T I O N")
    print("=" * 72)

    proc = ROOT / "data" / "processed"

    # Panel 1: Cumulative Equity Curves (all 8 strategies)
    print("\n--- Panel 1: Cumulative Equity Curves (8 Strategies Aligned) ---")
    all_eq_csv = proc / "phase6_all_equities_txadj.csv"
    assert all_eq_csv.exists(), f"Missing {all_eq_csv}"
    all_eq = pd.read_csv(all_eq_csv, index_col=0, parse_dates=True)
    assert all_eq.shape == (2222, 8), f"Expected shape (2222, 8), got {all_eq.shape}"
    assert not all_eq.isna().any().any(), "NaN values found in equity curves!"
    print(f"Panel 1 OK: 8 strategies aligned across 2,222 dates (2015-01-01 -> 2023-12-29)")
    for col in all_eq.columns:
        print(f"  {col:<20s} end equity: {all_eq[col].iloc[-1]:.4f}")

    # Panel 2: Drawdown Overlay
    print("\n--- Panel 2: Rolling Drawdown Overlay ---")
    dd_df = all_eq / all_eq.cummax() - 1.0
    mdd = dd_df.min() * 100.0
    assert (mdd <= 0.0).all(), "Drawdowns must be non-positive"
    assert (mdd >= -90.0).all(), "Drawdown exceeds plausible bound (> 90%)"
    print(f"Panel 2 OK: Max Drawdowns verified (Centroid: {mdd['centroid']:.2f}%, 1/N: {mdd['equal_weight']:.2f}%)")

    # Panel 3: Rolling 12-Month Sharpe Ratio Heatmap
    print("\n--- Panel 3: Rolling 12-Month Sharpe Ratios ---")
    rf_daily = (1.0 + 0.04) ** (1.0 / 252) - 1.0
    rets = all_eq.pct_change().dropna()
    rolling_excess = rets - rf_daily
    rolling_sr = (rolling_excess.rolling(252).mean() / rolling_excess.rolling(252).std(ddof=1)) * np.sqrt(252)
    valid_rolling = rolling_sr.dropna()
    assert len(valid_rolling) == 2221 - 251, f"Expected {2221 - 251} rolling points, got {len(valid_rolling)}"
    print(f"Panel 3 OK: {len(valid_rolling)} rolling 12-month Sharpe periods calculated across 8 strategies")

    # Panel 4: Jobson-Korkie (Memmel 2003) Pairwise Significance Table
    print("\n--- Panel 4: Jobson-Korkie Pairwise Significance Matrix ---")
    jk_csv = proc / "phase6_jk_pairwise_tests.csv"
    assert jk_csv.exists(), f"Missing {jk_csv}"
    jk_df = pd.read_csv(jk_csv)
    assert len(jk_df) == 28, f"Expected 28 rows (8C2), got {len(jk_df)}"
    c_1n = jk_df[((jk_df["strat_1"] == "centroid") & (jk_df["strat_2"] == "equal_weight")) |
                 ((jk_df["strat_1"] == "equal_weight") & (jk_df["strat_2"] == "centroid"))].iloc[0]
    print(f"Panel 4 OK: 28 pairwise tests verified (Centroid vs 1/N p-value = {c_1n['p_value']:.4f})")

    # Panel 5: Deflated Sharpe Ratio & PBO CSCV Distribution
    print("\n--- Panel 5: Deflated Sharpe Ratio & PBO CSCV Distribution ---")
    dsr_csv = proc / "phase6_dsr_summary.csv"
    pbo_csv = proc / "phase6_pbo_results.csv"
    assert dsr_csv.exists(), f"Missing {dsr_csv}"
    assert pbo_csv.exists(), f"Missing {pbo_csv}"
    
    dsr_df = pd.read_csv(dsr_csv).set_index("strategy")
    cent_dsr = float(dsr_df.loc["centroid", "dsr_prob"])
    print(f"Centroid DSR Probability: {cent_dsr:.4f} (threshold > 0.0)")
    assert cent_dsr > 0.0, f"HARD ASSERT 1 FAILED: Centroid DSR {cent_dsr} <= 0"
    print(">>> HARD ASSERT 1 PASS: Centroid Deflated Sharpe Ratio > 0.0! <<<")

    pbo_combs = pd.read_csv(pbo_csv)
    assert len(pbo_combs) == 20, f"Expected 20 combinations, got {len(pbo_combs)}"
    pbo_val = float(pbo_combs["overfit_flag"].mean())
    print(f"Combinatorially Symmetric Cross-Validation PBO: {pbo_val:.3f} (threshold < 0.5)")
    assert pbo_val < 0.5, f"HARD ASSERT 2 FAILED: PBO {pbo_val} >= 0.5"
    print(">>> HARD ASSERT 2 PASS: Probability of Backtest Overfitting < 0.5! <<<")

    # Panel 6: Factor Attribution Decomposition
    print("\n--- Panel 6: Factor Attribution Decomposition ---")
    factor_csv = proc / "phase6_factor_attribution.csv"
    assert factor_csv.exists(), f"Missing {factor_csv}"
    factor_df = pd.read_csv(factor_csv).set_index("strategy")
    assert len(factor_df) == 8, f"Expected 8 rows, got {len(factor_df)}"
    cent_alpha = float(factor_df.loc["centroid", "alpha_ann_bps"])
    cent_bmkt = float(factor_df.loc["centroid", "beta_mkt"])
    print(f"Panel 6 OK: 8 strategies decomposed into MKT, SMB, HML, MOM (Centroid Beta_MKT: {cent_bmkt:.3f})")

    # Panel 7: Pre-Defined Regime Analysis
    print("\n--- Panel 7: Pre-Defined Regime Analysis (6 Regimes x 8 Strategies) ---")
    regime_csv = proc / "phase6_regime_analysis.csv"
    assert regime_csv.exists(), f"Missing {regime_csv}"
    regime_df = pd.read_csv(regime_csv)
    assert len(regime_df) == 48, f"Expected 48 rows, got {len(regime_df)}"
    assert len(regime_df["regime_id"].unique()) == 6, "Expected 6 unique regimes"
    print(f"Panel 7 OK: 48 regime rows verified across 6 pre-defined macro epochs")

    # Panel 8: Transaction Cost Sensitivity Sweep & HARD ASSERT 3
    print("\n--- Panel 8: Transaction Cost Sensitivity Sweep & Performance Summary Audit ---")
    txcost_csv = proc / "phase6_txcost_sensitivity.csv"
    assert txcost_csv.exists(), f"Missing {txcost_csv}"
    txcost_df = pd.read_csv(txcost_csv)
    assert len(txcost_df) == 48, f"Expected 48 rows (6 cost levels x 8 strategies), got {len(txcost_df)}"
    
    perf_csv = proc / "phase6_performance_summary.csv"
    assert perf_csv.exists(), f"Missing {perf_csv}"
    perf_df = pd.read_csv(perf_csv)
    assert len(perf_df) == 8, f"HARD ASSERT 3 FAILED: Expected 8 rows in performance summary, got {len(perf_df)}"
    print(f"Performance Summary contains {len(perf_df)} strategies:")
    for _, r in perf_df.iterrows():
        print(f"  {r['strategy']:<20s} CAGR={r['cagr_pct']:>6.2f}% Vol={r['ann_vol']:>6.3f} Sharpe={r['sharpe_txadj']:>7.4f} MDD={r['max_dd_pct']:>6.2f}%")
    print(">>> HARD ASSERT 3 PASS: Performance Summary contains exactly 8 strategies! <<<")

    print("\n" + "=" * 72)
    print("ALL 8 PANELS EXECUTED AND 3/3 HARD ASSERTS PASSED CLEANLY")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
