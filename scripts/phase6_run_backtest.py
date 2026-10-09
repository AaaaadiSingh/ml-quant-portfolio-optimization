from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import (
    DEV_END,
    QUARTERLY_STEP_D,
    STUDY_START,
    _raw_sharpe,
    _rebalance_dates,
)
from phase3_run_5baselines import (
    RISK_FREE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)
from src.backtest_engine import (
    DEFAULT_TURNOVER_ONE_SIDED_BPS,
    WalkForwardBacktestEngine,
    apply_tx_costs_to_equity,
    simulate_walk_forward,
)
from src.data_loader import FinalHoldoutDate, load_prices
from src.metrics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    compute_all_metrics,
    empirical_cvar,
    empirical_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

TURNOVER_ONE_SIDED_BPS = DEFAULT_TURNOVER_ONE_SIDED_BPS
CENTROID_WEIGHTS_CSV = ROOT / "data" / "processed" / "phase5_selected_centroid_weights.csv"
BASELINE_EQUITIES_CSV = ROOT / "data" / "processed" / "phase3_baseline_equities_txadj.csv"
RIDGE_EQUITY_CSV = ROOT / "data" / "processed" / "phase4_ridge_linreg_equity_txadj.csv"
RF_EQUITY_CSV = ROOT / "data" / "processed" / "phase4_rf_equity_txadj.csv"
XGB_EQUITY_CSV = ROOT / "data" / "processed" / "phase4_xgb_equity_txadj.csv"

OUT_ALL_EQUITIES_CSV = ROOT / "data" / "processed" / "phase6_all_equities_txadj.csv"
OUT_PERF_SUMMARY_CSV = ROOT / "data" / "processed" / "phase6_performance_summary.csv"
OUT_CENTROID_DRIFT_CSV = ROOT / "data" / "processed" / "phase6_centroid_drifted_weights.csv"


def reconstruct_centroid_equity_and_drift(
    close_prices: pd.DataFrame,
    centroid_weights_df: pd.DataFrame,
    bps_per_turnover: float = TURNOVER_ONE_SIDED_BPS,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    tickers = sorted(close_prices.columns.tolist())
    weights_piv = centroid_weights_df.pivot_table(
        index="rebal_date", columns="ticker", values="weight"
    ).reindex(columns=tickers).fillna(0.0)
    weights_piv.index = pd.to_datetime(weights_piv.index)
    weights_by_rebal = {rd: weights_piv.loc[rd].values.astype(float) for rd in weights_piv.index}
    
    engine = WalkForwardBacktestEngine(close_prices, bps_per_turnover=bps_per_turnover)
    raw_s, txadj_s, turnover_s, drift_df = engine.simulate(weights_by_rebal)
    raw_s.name = "centroid_raw"
    txadj_s.name = "centroid"
    return raw_s, txadj_s, turnover_s, drift_df


def main() -> None:
    print("=" * 72)
    print("P H A S E  6  --  S T E P  6 . 1  B A C K T E S T   E N G I N E")
    print("=" * 72)
    
    # 1. Load dev prices (never read 2024+)
    print("[6.1] Loading dev prices (2015-01-01 -> 2023-12-31)...")
    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    prices = load_prices(
        tickers,
        start_date=STUDY_START,
        end_date=DEV_END,
        final_holdout=False,
    )
    assert prices.index.max() < FinalHoldoutDate, f"Holdout leak: max date {prices.index.max()} >= {FinalHoldoutDate}"
    close = prices["Close"].ffill().bfill().sort_index().loc[:, tickers]
    print(f"      Close matrix: {close.shape} (dates x tickers), max date: {close.index.max().date()}")
    assert len(close) == 2222, f"Expected 2222 dates, got {len(close)}"
    
    # 2. Load Phase 5 Selected Centroid Weights
    print("[6.1] Loading Phase 5 Selected Centroid Weights...")
    assert CENTROID_WEIGHTS_CSV.exists(), f"Missing {CENTROID_WEIGHTS_CSV}"
    cent_w_df = pd.read_csv(CENTROID_WEIGHTS_CSV)
    assert len(cent_w_df) == 1702, f"Expected 1702 rows, got {len(cent_w_df)}"
    cent_w_df["rebal_date"] = pd.to_datetime(cent_w_df["rebal_date"])
    
    # 3. Reconstruct Centroid Equity & Weight Drift
    print("[6.1] Reconstructing Centroid daily equity with weight drift and 10 bps tx drag...")
    cent_raw, cent_txadj, cent_turnover, drift_df = reconstruct_centroid_equity_and_drift(
        close, cent_w_df, bps_per_turnover=TURNOVER_ONE_SIDED_BPS
    )
    drift_df.to_csv(OUT_CENTROID_DRIFT_CSV, index=False)
    print(f"      Centroid reconstructed: length={len(cent_txadj)}, drift max sum-error: {np.max(np.abs(drift_df['drift_sum'] - 1.0)):.2e}")
    
    # 4. Load & Align All 8 Strategies
    print("[6.1] Merging all 8 walk-forward strategies (tx-adjusted)...")
    p3_baselines = pd.read_csv(BASELINE_EQUITIES_CSV, index_col=0, parse_dates=True)
    ridge_eq = pd.read_csv(RIDGE_EQUITY_CSV, index_col=0, parse_dates=True)["ridge_linreg"]
    rf_eq = pd.read_csv(RF_EQUITY_CSV, index_col=0, parse_dates=True)["rf"]
    xgb_eq = pd.read_csv(XGB_EQUITY_CSV, index_col=0, parse_dates=True)["xgb"]
    
    # 8 strategies:
    # 1. centroid (Phase 5 selected centroid)
    # 2. ridge_linreg (Phase 4 winning ML model)
    # 3. rf (Phase 4 Random Forest)
    # 4. xgb (Phase 4 XGBoost)
    # 5. equal_weight (Phase 3 1/N)
    # 6. min_variance (Phase 3 MinVar LW)
    # 7. risk_parity (Phase 3 RiskParity LW)
    # 8. classic_max_sharpe (Phase 3 Classic Max Sharpe LW + CMS)
    
    all_equities = pd.DataFrame(index=close.index)
    all_equities["centroid"] = cent_txadj
    all_equities["ridge_linreg"] = ridge_eq
    all_equities["rf"] = rf_eq
    all_equities["xgb"] = xgb_eq
    all_equities["equal_weight"] = p3_baselines["equal_weight"]
    all_equities["min_variance"] = p3_baselines["min_variance"]
    all_equities["risk_parity"] = p3_baselines["risk_parity"]
    all_equities["classic_max_sharpe"] = p3_baselines["classic_max_sharpe"]
    
    assert all_equities.shape == (2222, 8), f"Expected shape (2222, 8), got {all_equities.shape}"
    assert not all_equities.isna().any().any(), "NaN found in all_equities!"
    all_equities.to_csv(OUT_ALL_EQUITIES_CSV)
    print(f"      Saved all equities -> {OUT_ALL_EQUITIES_CSV} shape {all_equities.shape}")
    
    # 5. Load Turnovers for All 8 Strategies
    # Centroid turnover:
    cent_ann_turnover = float(cent_turnover.mean() * (37.0 / 8.997) * 10000.0)  # quarterly avg inter-RD
    # From phase 5 stability AB:
    stab_csv = ROOT / "data" / "processed" / "phase5_stability_ab_vs_phase4_pointestimate.csv"
    stab_df = pd.read_csv(stab_csv).set_index("strategy")
    
    turnover_map = {
        "centroid": float(stab_df.loc["selected_centroid", "avg_inter_rd_turnover_bps_ann"]),
        "ridge_linreg": float(stab_df.loc["phase4_ridge_pointestimate", "avg_inter_rd_turnover_bps_ann"]),
        "rf": 13787.0,   # from phase 4 summary normalized to annual
        "xgb": 12799.3,
        "equal_weight": 1810.8,
        "min_variance": 19993.6,
        "risk_parity": 5816.2,
        "classic_max_sharpe": 24362.0,
    }
    
    # 6. Compute Comprehensive Performance Summary
    print("[6.1] Computing NB2-consistent metrics & empirical risk measures for 8 strategies...")
    summary_rows = []
    strategy_display_names = {
        "centroid": "MC Centroid (Mean, LW Σ, ±3pp)",
        "ridge_linreg": "Ridge μ̂ Point Estimate (LW Σ, ±3pp)",
        "rf": "RF μ̂ Point Estimate (LW Σ, ±3pp)",
        "xgb": "XGB μ̂ Point Estimate (LW Σ, ±3pp)",
        "equal_weight": "Equal Weight (1/N)",
        "min_variance": "Minimum Variance (LW Σ)",
        "risk_parity": "Risk Parity (LW Σ)",
        "classic_max_sharpe": "Classic Max Sharpe (LW Σ, ±3pp)",
    }
    
    for col in all_equities.columns:
        eq_s = all_equities[col]
        m = compute_all_metrics(eq_s, risk_free_annual=RISK_FREE_ANNUAL, trading_days=TRADING_DAYS_PER_YEAR)
        
        summary_rows.append({
            "strategy": col,
            "display_name": strategy_display_names.get(col, col),
            "cagr_pct": m["cagr_pct"],
            "ann_vol": m["ann_vol"],
            "sharpe_txadj": m["sharpe_txadj"],
            "max_dd_pct": m["max_dd_pct"],
            "sortino": m["sortino"],
            "calmar": m["calmar"],
            "turnover_bps_ann": turnover_map.get(col, float("nan")),
            "var_95_daily_pct": m["var_95_daily_pct"],
            "cvar_95_daily_pct": m["cvar_95_daily_pct"],
        })
        
    perf_df = pd.DataFrame(summary_rows)
    assert len(perf_df) == 8, f"Expected 8 rows, got {len(perf_df)}"
    perf_df.to_csv(OUT_PERF_SUMMARY_CSV, index=False)
    print(f"      Saved performance summary -> {OUT_PERF_SUMMARY_CSV} (8 rows)")
    print("\n" + perf_df[["strategy", "cagr_pct", "ann_vol", "sharpe_txadj", "max_dd_pct", "turnover_bps_ann"]].to_string(index=False))
    print("\n[6.1] Step 6.1 complete.")


if __name__ == "__main__":
    main()
