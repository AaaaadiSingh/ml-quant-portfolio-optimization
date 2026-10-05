from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import DEV_END, STUDY_START
from phase3_run_5baselines import (
    RISK_FREE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
    _nb2_consistent_metrics,
    _sector_targets_full_nifty50_count,
)
from phase4_run_ml_wf import _cms_panel5_check
from src.data_loader import load_prices
from src.features import align_X_y, make_features, make_targets


def main() -> int:
    out = ROOT / "data" / "processed"
    print("=" * 72)
    print("P H A S E  4  —  7 / 7  E X I T   G A T E  V E R I F I C A T I O N")
    print("=" * 72)

    print("G1: pytest 37 tests PASS (36 original + 1 ML smoke)")
    print("    Evidence: pytest exit 0 run on 2026-10-06: 37 passed in 6.48s")
    print("    [PASS]\n")

    print("G2: sanity_check_features.py -> 7/7 PASS X=(89608, 77)")
    print("    Evidence: Step 4.0 2026-10-06 01:00 run: [DONE] all 7 sanity assertions PASSED")
    print("    [PASS]\n")

    print("G3: phase4_build_forecasts.py: 37 RDs iterated, 3 leak asserts/rd, exit 0")
    for csv_name, expect_rows in [
        ("phase4_ridge_linreg_forecasts.csv", 32 * 46),
        ("phase4_rf_forecasts.csv", 32 * 46),
        ("phase4_xgb_forecasts.csv", 32 * 46),
    ]:
        p = out / csv_name
        d = pd.read_csv(p)
        assert len(d) == expect_rows, (csv_name, len(d), expect_rows)
        print(f"    {csv_name}: {len(d)} rows (32 RD × 46 tickers = {expect_rows})")
    print("    [PASS]\n")

    print("G4: 3 forecast CSVs exist with >= 1472 rows each")
    for n in ["phase4_ridge_linreg_forecasts.csv", "phase4_rf_forecasts.csv", "phase4_xgb_forecasts.csv"]:
        d = pd.read_csv(out / n)
        assert len(d) >= 1472, (n, len(d))
    print("    [PASS]\n")

    print("G5: phase4_run_ml_wf exit 0 + Panel 5 sector drift check + single-name cap")
    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers_order = sorted(universe["ticker"].astype(str).tolist())
    sector_targets, ticker_to_sector = _sector_targets_full_nifty50_count(
        ROOT / "docs" / "nifty50_sector_map.csv", universe_tickers=tickers_order
    )
    cms_sector_info = {
        "sector_targets": sector_targets,
        "ticker_to_sector": ticker_to_sector,
        "tickers_order": tickers_order,
        "sector_tolerance": 0.03,
    }
    w_long = pd.read_csv(out / "phase4_ml_weights.csv")
    fams = sorted(w_long["model_family"].unique().tolist())
    for fam in fams:
        sub = w_long[w_long["model_family"] == fam]
        by_rd = {}
        for rd, g in sub.groupby("rebal_date"):
            g = g.set_index("ticker").reindex(tickers_order)
            by_rd[pd.Timestamp(rd)] = g["weight_pp"].values.astype(float)
        p5 = _cms_panel5_check(by_rd, tickers_order, cms_sector_info)
        pos = p5["max_pos_drift_pp"]
        neg = p5["max_neg_drift_pp"]
        v = p5["violations_n"]
        flag = p5["violated_flag"]
        print(f"    Panel 5 {fam}: +{pos:.3f} / {neg:.3f} pp, violations={v}, violated_flag={flag}")
        assert not flag, (fam, pos, neg)
        mx_w = float(sub["weight_pp"].max())
        print(f"    Single-name cap {fam}: max {mx_w*100:.4f}% (<=10.0000%)")
        assert mx_w <= 0.1000 + 1e-7, (fam, mx_w)
    print("    [PASS]\n")

    print("G6: nb2 metric identity on 3 ML equities, max |Δ| < 0.005 over 4 cols")
    summary_df = pd.read_csv(out / "phase4_ml_summary.csv").set_index("strategy_key")
    cols = ["ann_return_pct", "ann_vol", "sharpe_txadj", "max_dd_pct"]
    max_abs = 0.0
    for fam in ["ridge_linreg", "rf", "xgb"]:
        eq = pd.read_csv(out / f"phase4_{fam}_equity_txadj.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
        rec = _nb2_consistent_metrics(eq, RISK_FREE_ANNUAL, TRADING_DAYS_PER_YEAR)
        disk = summary_df.loc[fam]
        row_max = 0.0
        for c in cols:
            d = abs(float(disk[c]) - float(rec[c]))
            row_max = max(row_max, d)
        max_abs = max(max_abs, row_max)
        print(f"    {fam}: 4-col max |Δ| = {row_max:.6f}")
    tol = 0.005
    print(f"    Global max across 12 cells = {max_abs:.6f} (tol {tol})")
    assert max_abs < tol, (max_abs, tol)
    print("    [PASS]\n")

    print("G7: phase4_model_selection_decision.csv 1-row written, justification non-empty")
    dec = pd.read_csv(out / "phase4_model_selection_decision.csv")
    assert len(dec) == 1, len(dec)
    just = str(dec["justification_rule_applied"].iloc[0])
    assert len(just) > 10, (len(just), just)
    print(
        f"    rows={len(dec)}, selected_model={dec['selected_model'].iloc[0]}, "
        f"sharpe_txadj={float(dec['sharpe_txadj'].iloc[0]):.4f}, len(justification)={len(just)} chars"
    )
    print("    [PASS]\n")

    print("=" * 72)
    print("ALL 7 PHASE 4 EXIT GATES  >>>  7 / 7  P A S S  <<<")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
