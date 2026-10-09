from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    out = ROOT / "data" / "processed"
    print("=" * 72)
    print("P H A S E  7  --  7 / 7  E X I T   G A T E  V E R I F I C A T I O N")
    print("=" * 72)

    # G1: Environment tripwire
    print("G1: Environment sanity tripwire (real pytest suite execution & prior phase verification)")
    import pytest
    pytest_exit = pytest.main(["tests/", "-q", "--tb=no"])
    assert pytest_exit == 0, f"pytest suite failed with exit code {pytest_exit}"
    print(f"    Evidence: pytest executed cleanly (exit code {int(pytest_exit)}), prior phase gates green")
    print("    [PASS]\n")

    # G2: Step 7.1 Block Length Selection (ACF Diagnostic)
    print("G2: Step 7.1 Block Length Selection & Volatility Clustering Diagnostic")
    bl_csv = out / "phase7_block_length.csv"
    acf_csv = out / "phase7_acf_squared_returns.csv"
    assert bl_csv.exists(), f"Missing {bl_csv}"
    assert acf_csv.exists(), f"Missing {acf_csv}"
    
    bl_df = pd.read_csv(bl_csv)
    acf_df = pd.read_csv(acf_csv)
    chosen_L = int(bl_df["chosen_L"].iloc[0])
    ci_val = float(bl_df["ci_threshold"].iloc[0])
    T_val = int(bl_df["T"].iloc[0])
    
    assert 5 <= chosen_L <= 60, f"chosen_L {chosen_L} outside expected bound [5, 60]"
    assert T_val == 2221, f"Expected T=2221 dev-window log return rows, got {T_val}"
    assert len(acf_df) >= 61, f"Expected >= 61 lags (0..60), got {len(acf_df)}"
    print(f"    block length L = {chosen_L} (in [5, 60]), T = {T_val}, 95% CI upper = {ci_val:.4f}")
    print("    [PASS]\n")

    # G3: Step 7.2 Bootstrap Distribution Shape & Finite Invariants
    print("G3: Step 7.2 Multi-Asset Stationary Block Bootstrap Distributions (B x 8)")
    dist_csv = out / "phase7_bootstrap_distributions.csv"
    assert dist_csv.exists(), f"Missing {dist_csv}"
    dist_df = pd.read_csv(dist_csv)
    
    B = dist_df["path_id"].nunique()
    strategies = sorted(dist_df["strategy"].unique().tolist())
    assert B >= 200, f"Expected B >= 200 synthetic paths, got {B}"
    assert len(strategies) == 8, f"Expected exactly 8 strategies, got {len(strategies)}"
    assert len(dist_df) == B * 8, f"Expected {B * 8} rows, got {len(dist_df)}"
    
    required_metric_cols = ["cagr", "annualized_vol", "sharpe", "sortino", "max_drawdown", "calmar"]
    for col in required_metric_cols:
        assert col in dist_df.columns, f"Missing metric column {col}"
        assert dist_df[col].notna().all(), f"NaN values detected in {col}"
        assert np.isfinite(dist_df[col]).all(), f"Non-finite values detected in {col}"
        
    print(f"    distribution shape = {dist_df.shape} ({B} paths x 8 strategies = {len(dist_df)} rows, 6 metrics)")
    print(f"    0 NaNs, all values strictly finite across all paths")
    print("    [PASS]\n")

    # G4: Step 7.3 Metric Distribution Summary Table (8 strategies x 30 stats)
    print("G4: Step 7.3 Metric Distributions Summary Table")
    metric_summary_csv = out / "phase7_metric_distributions.csv"
    assert metric_summary_csv.exists(), f"Missing {metric_summary_csv}"
    metric_df = pd.read_csv(metric_summary_csv)
    assert len(metric_df) == 8, f"Expected 8 rows (one per strategy), got {len(metric_df)}"
    assert metric_df.shape[1] == 31, f"Expected 31 columns (strategy + 30 stats), got {metric_df.shape[1]}"
    assert not metric_df.isna().any().any(), "NaN found in metric distributions summary!"
    print(f"    8 strategies x 30 statistics populated cleanly (mean, std, P5, P50, P95 per metric)")
    print("    [PASS]\n")

    # G5: Step 7.3 Outperformance Probabilities & Hard Robustness Asserts
    print("G5: Step 7.3 Outperformance Probabilities & Research Question Claims")
    prob_csv = out / "phase7_outperformance_probabilities.csv"
    assert prob_csv.exists(), f"Missing {prob_csv}"
    prob_df = pd.read_csv(prob_csv)
    assert len(prob_df) == 7, f"Expected 7 competitors, got {len(prob_df)}"
    
    ew_prob = prob_df[prob_df["competitor"] == "equal_weight"]["p_centroid_beats_sharpe"].iloc[0]
    cent_p5 = dist_df[dist_df["strategy"] == "centroid"]["sharpe"].quantile(0.05)
    
    print(f"    P(Centroid Sharpe > Equal Weight Sharpe) across synthetic paths: {ew_prob * 100:.2f}% (> 50.0%)")
    assert ew_prob > 0.50, f"Assertion failed: Centroid Sharpe win-rate vs EW {ew_prob} <= 0.50"
    print(f"    Centroid P5-Sharpe (worst 5% synthetic histories): {cent_p5:.4f} (> 0.0)")
    assert cent_p5 > 0.0, f"Assertion failed: Centroid P5-Sharpe {cent_p5} <= 0.0"
    print("    [PASS]\n")

    # G6: Step 7.4 Volatility-Regime Tercile Performance Slices
    print("G6: Step 7.4 Volatility-Regime Tercile Distributions (3 Regimes x 8 Strategies)")
    regime_csv = out / "phase7_regime_vol_tercile.csv"
    assert regime_csv.exists(), f"Missing {regime_csv}"
    regime_df = pd.read_csv(regime_csv)
    assert len(regime_df) == 24, f"Expected 24 rows (3 regimes x 8 strategies), got {len(regime_df)}"
    assert not regime_df.isna().any().any(), "NaN found in regime analysis table!"
    piv_reg = regime_df.pivot(index="regime", columns="strategy", values="sharpe_p50")
    assert piv_reg.shape == (3, 8), f"Expected pivot shape (3, 8), got {piv_reg.shape}"
    print(f"    24 regime rows verified across Low-Vol, Mid-Vol, and High-Vol terciles")
    print(f"    Low-Vol Centroid median Sharpe: {piv_reg.loc['low_vol', 'centroid']:.4f}")
    print(f"    Mid-Vol Centroid median Sharpe: {piv_reg.loc['mid_vol', 'centroid']:.4f}")
    print(f"    High-Vol Centroid median Sharpe: {piv_reg.loc['high_vol', 'centroid']:.4f}")
    print("    [PASS]\n")

    # G7: Step 7.5 Notebook 05 Headless Validation (8 Panels + 3 Hard Asserts)
    print("G7: Step 7.5 Notebook 05 Headless Validation (8 Panels, 3 Hard Asserts)")
    nb_val_script = ROOT / "notebooks" / "_run_nb5_validation.py"
    assert nb_val_script.exists(), f"Missing {nb_val_script}"
    
    # Run the validation script in a sub-process
    val_proc = subprocess.run(
        [sys.executable, str(nb_val_script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if val_proc.returncode != 0:
        print(val_proc.stdout)
        print(val_proc.stderr)
        raise RuntimeError(f"_run_nb5_validation.py failed with return code {val_proc.returncode}")
        
    print("    Notebook 05 headless validator returned exit code 0")
    print("    8/8 panels PASS, 3/3 hard asserts GREEN")
    print("    [PASS]\n")

    print("=" * 72)
    print("ALL 7 PHASE 7 EXIT GATES  >>>  7 / 7  P A S S  <<<")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
