"""
Phase 8 - Step 8.5: Exit Gate Script
=====================================
Verifies all Phase 8 deliverables are present and valid before final sign-off.

Gates:
  G1 - Environment tripwire (pytest 43/43, sanity 7/7, _p7_exit_gates exit 0)
  G2 - results/metrics_summary.csv exists with 8 rows, no NaN in core cols
  G3 - DSR/PBO preserved (centroid DSR > 0, PBO < 0.50)
  G4 - P7 win-rate preserved (centroid p7_win_rate_vs_ew > 0.50)
  G5 - All 8 dev-window figures exist and > 5 KB
  G6 - README.md contains centroid Sharpe value
  G7 - results/holdout_eval.csv present with 8 rows

Exit code 0 on all 7 gates PASS, non-zero otherwise.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"


def run_gate(gate_id: str, description: str, check_fn) -> bool:
    try:
        check_fn()
        print(f"  [PASS] G{gate_id}: {description}")
        return True
    except AssertionError as e:
        print(f"  [FAIL] G{gate_id}: {description}")
        print(f"         {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] G{gate_id}: {description}")
        print(f"         Unexpected error: {e}")
        return False


def gate1_environment() -> None:
    """Run pytest, sanity_check_features, _p7_exit_gates and verify exit codes."""
    checks = [
        ([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"], "pytest 43/43"),
        ([sys.executable, "scripts/sanity_check_features.py"], "sanity 7/7"),
        ([sys.executable, "scripts/_p7_exit_gates.py"], "_p7_exit_gates"),
    ]
    for cmd, label in checks:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        assert result.returncode == 0, \
            f"{label} failed (exit {result.returncode}): {result.stdout[-300:]}"


def gate2_metrics_summary() -> None:
    p = RESULTS / "metrics_summary.csv"
    assert p.exists(), "results/metrics_summary.csv does not exist"
    df = pd.read_csv(p)
    assert len(df) == 8, f"Expected 8 rows, got {len(df)}"
    core_cols = ["ann_return_pct", "ann_vol", "sharpe_txadj", "max_dd_pct", "dsr_prob"]
    for col in core_cols:
        assert col in df.columns, f"Missing column: {col}"
        n_nan = df[col].isna().sum()
        assert n_nan == 0, f"NaN in '{col}': {n_nan} rows"


def gate3_dsr_pbo() -> None:
    ms = pd.read_csv(RESULTS / "metrics_summary.csv")
    centroid = ms.loc[ms["strategy"] == "centroid"]
    assert not centroid.empty, "Centroid row missing"
    dsr = float(centroid["dsr_prob"].iloc[0])
    assert dsr > 0, f"Centroid DSR prob = {dsr:.6f} <= 0 (expected ~0.9967)"
    pbo = float(ms["pbo"].iloc[0])
    assert pbo < 0.50, f"PBO = {pbo:.4f} >= 0.50 (expected 0.300)"


def gate4_p7_win_rate() -> None:
    ms = pd.read_csv(RESULTS / "metrics_summary.csv")
    centroid = ms.loc[ms["strategy"] == "centroid"]
    assert not centroid.empty, "Centroid row missing"
    win_rate = float(centroid["p7_win_rate_vs_ew"].iloc[0])
    assert win_rate > 0.50, f"P7 win-rate = {win_rate:.3f} <= 0.50 (expected 0.640)"


def gate5_figures() -> None:
    expected = [
        "fig1_equity_curves.png", "fig2_drawdown_overlay.png",
        "fig3_performance_bar.png", "fig4_jk_heatmap.png",
        "fig5_bootstrap_boxplot.png", "fig6_regime_heatmap.png",
        "fig7_txcost_sensitivity.png", "fig8_factor_attribution.png",
    ]
    for fname in expected:
        p = FIGS / fname
        assert p.exists(), f"Missing figure: {fname}"
        sz = p.stat().st_size
        assert sz > 5_000, f"{fname} too small: {sz} bytes"


def gate6_readme() -> None:
    readme = ROOT / "README.md"
    assert readme.exists(), "README.md does not exist"
    content = readme.read_text(encoding="utf-8")
    assert "Phase 8" in content or "v1.0-final" in content or "Headline Results" in content, \
        "README.md does not appear to contain Phase 8 / final results content"
    # Check that centroid Sharpe value appears in README
    ms = pd.read_csv(RESULTS / "metrics_summary.csv")
    centroid_sharpe = float(ms.loc[ms["strategy"] == "centroid", "sharpe_txadj"].iloc[0])
    sharpe_str = f"{centroid_sharpe:.4f}"
    assert sharpe_str in content, \
        f"Centroid Sharpe value {sharpe_str} not found in README.md"


def gate7_holdout_eval() -> None:
    p = RESULTS / "holdout_eval.csv"
    assert p.exists(), "results/holdout_eval.csv does not exist"
    df = pd.read_csv(p)
    assert len(df) == 8, f"Expected 8 rows in holdout_eval.csv, got {len(df)}"


def main() -> None:
    print("=" * 68)
    print("PHASE 8 - EXIT GATE VERIFICATION")
    print("=" * 68)

    gates = [
        ("1", "Environment tripwire (pytest + sanity + P7 gates)", gate1_environment),
        ("2", "results/metrics_summary.csv: 8 rows, no NaN in core cols", gate2_metrics_summary),
        ("3", "DSR prob for Centroid > 0; PBO < 0.50", gate3_dsr_pbo),
        ("4", "P7 win-rate (Centroid > EW) > 50%", gate4_p7_win_rate),
        ("5", "All 8 dev-window figures exist and > 5 KB", gate5_figures),
        ("6", "README.md contains centroid Sharpe + Phase 8 content", gate6_readme),
        ("7", "results/holdout_eval.csv exists with 8 rows", gate7_holdout_eval),
    ]

    results = []
    for gate_id, desc, fn in gates:
        passed = run_gate(gate_id, desc, fn)
        results.append(passed)

    n_pass = sum(results)
    n_total = len(results)
    print()
    print("=" * 68)
    if all(results):
        print(f"ALL {n_total} PHASE 8 EXIT GATES  >>>  {n_pass} / {n_total}  P A S S  <<<")
        print("Phase 8 is 100% complete. Ready for final commit and v1.0-final tag.")
    else:
        failed = [gates[i][0] for i, r in enumerate(results) if not r]
        print(f"PHASE 8 EXIT GATES: {n_pass}/{n_total} PASS - FAILED: G{', G'.join(failed)}")
    print("=" * 68)

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
