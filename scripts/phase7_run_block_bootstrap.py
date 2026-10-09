from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import FinalHoldoutDate, load_prices
from src.metrics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

STUDY_START = "2015-01-01"
DEV_END = "2023-12-31"
SEED = 42
B_DEFAULT = 200
TX_BPS = 10.0

OUT_DISTRIBUTIONS_CSV = ROOT / "data" / "processed" / "phase7_bootstrap_distributions.csv"
BLOCK_LENGTH_CSV = ROOT / "data" / "processed" / "phase7_block_length.csv"


def load_strategy_weights(tickers: List[str], daily_index: pd.DatetimeIndex) -> Dict[str, Dict[int, np.ndarray]]:
    p5_w = pd.read_csv(ROOT / "data" / "processed" / "phase5_selected_centroid_weights.csv")
    p5_w["rebal_date"] = pd.to_datetime(p5_w["rebal_date"])
    piv_cent = p5_w.pivot_table(index="rebal_date", columns="ticker", values="weight").reindex(columns=tickers).fillna(0.0)
    
    p4_w = pd.read_csv(ROOT / "data" / "processed" / "phase4_ml_weights.csv")
    p4_w["rebal_date"] = pd.to_datetime(p4_w["rebal_date"])
    ml_pivs = {
        m: p4_w[p4_w["model_family"] == m].pivot_table(index="rebal_date", columns="ticker", values="weight_pp").reindex(columns=tickers).fillna(0.0)
        for m in ["ridge_linreg", "rf", "xgb"]
    }
    
    p3_w = pd.read_csv(ROOT / "data" / "processed" / "phase3_baseline_weights.csv")
    p3_w["rebal_date"] = pd.to_datetime(p3_w["rebal_date"])
    base_pivs = {
        b: p3_w[p3_w["strategy"] == b].set_index("rebal_date")[tickers].fillna(0.0)
        for b in ["equal_weight", "min_variance", "risk_parity", "classic_max_sharpe"]
    }
    
    all_pivs = {
        "centroid": piv_cent,
        **ml_pivs,
        **base_pivs,
    }
    
    weights_by_strat: Dict[str, Dict[int, np.ndarray]] = {}
    for strat, piv in all_pivs.items():
        loc_dict: Dict[int, np.ndarray] = {}
        for rd in piv.index:
            if rd in daily_index:
                loc = int(daily_index.get_loc(rd))
                w = piv.loc[rd].values.astype(float)
                loc_dict[loc] = w
        weights_by_strat[strat] = loc_dict
        
    return weights_by_strat


def simulate_strategy_fast(
    daily_simple_returns: np.ndarray,
    rebal_locs: Dict[int, np.ndarray],
    n_days: int,
    bps: float = TX_BPS,
) -> np.ndarray:
    equity = np.ones(n_days, dtype=float)
    first_rd_idx = min(rebal_locs.keys())
    w_curr = rebal_locs[first_rd_idx].copy()
    turnovers: Dict[int, float] = {}
    
    for t in range(1, n_days):
        r_t = daily_simple_returns[t - 1]
        p_ret = float(np.dot(w_curr, r_t))
        equity[t] = equity[t - 1] * (1.0 + p_ret)
        
        denom = 1.0 + p_ret
        if denom > 1e-12:
            w_drift = (w_curr * (1.0 + r_t)) / denom
        else:
            w_drift = np.copy(w_curr)
            
        if t in rebal_locs:
            w_tgt = rebal_locs[t]
            turn = 0.5 * float(np.sum(np.abs(w_tgt - w_drift)))
            turnovers[t] = turn
            w_curr = w_tgt.copy()
        else:
            w_curr = w_drift
            
    drag = np.zeros(n_days, dtype=float)
    for t, turn in turnovers.items():
        drag[t] = (bps / 10000.0) * turn
    cum_drag = np.cumsum(drag)
    equity = equity * np.exp(-cum_drag)
    return equity


def evaluate_synthetic_path(
    path_id: int,
    synth_log_ret: np.ndarray,
    weights_by_strat: Dict[str, Dict[int, np.ndarray]],
    daily_index: pd.DatetimeIndex,
) -> List[dict]:
    synth_simple_ret = np.exp(synth_log_ret) - 1.0
    n_days = len(daily_index)
    rows: List[dict] = []
    
    for strat, locs in weights_by_strat.items():
        eq_arr = simulate_strategy_fast(synth_simple_ret, locs, n_days)
        eq_series = pd.Series(eq_arr, index=daily_index)
        
        cagr_val = cagr(eq_series)
        vol_val = annualized_volatility(eq_series)
        sharpe_val = sharpe_ratio(eq_series)
        sortino_val = sortino_ratio(eq_series)
        mdd_val = max_drawdown(eq_series)
        calmar_val = calmar_ratio(eq_series)
        
        rows.append(
            {
                "path_id": path_id,
                "strategy": strat,
                "cagr": cagr_val,
                "annualized_vol": vol_val,
                "sharpe": sharpe_val,
                "sortino": sortino_val,
                "max_drawdown": mdd_val,
                "calmar": calmar_val,
            }
        )
    return rows


def run_block_bootstrap(B: int = B_DEFAULT) -> pd.DataFrame:
    assert BLOCK_LENGTH_CSV.exists(), f"Missing {BLOCK_LENGTH_CSV}. Run Step 7.1 first."
    bl_df = pd.read_csv(BLOCK_LENGTH_CSV)
    L = int(bl_df["chosen_L"].iloc[0])
    
    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    
    prices = load_prices(
        tickers,
        start_date=STUDY_START,
        end_date=DEV_END,
        final_holdout=False,
    )
    assert prices.index.max() < FinalHoldoutDate, f"Holdout leak: {prices.index.max()}"
    
    close = prices["Close"].ffill().bfill().sort_index().loc[:, tickers]
    assert len(close) == 2222, f"Expected 2222 rows, got {len(close)}"
    
    log_ret = np.log(close / close.shift(1)).dropna().values
    assert log_ret.shape == (2221, 46), f"Expected (2221, 46), got {log_ret.shape}"
    
    weights_by_strat = load_strategy_weights(tickers, close.index)
    assert len(weights_by_strat) == 8, f"Expected 8 strategies, got {len(weights_by_strat)}"
    
    print(f"[7.2] Generating {B} synthetic paths via StationaryBootstrap (L={L}, SEED={SEED})...")
    bs = StationaryBootstrap(L, log_ret, seed=SEED)
    
    # Materialize synthetic return matrices
    synthetic_matrices: List[np.ndarray] = []
    for pos_args, _ in bs.bootstrap(B):
        synthetic_matrices.append(pos_args[0])
        
    print(f"[7.2] Simulating 8 strategies across {B} synthetic paths in parallel...")
    t0 = time.time()
    all_path_results = Parallel(n_jobs=-1, batch_size=5)(
        delayed(evaluate_synthetic_path)(
            b,
            synthetic_matrices[b],
            weights_by_strat,
            close.index,
        )
        for b in range(B)
    )
    elapsed = time.time() - t0
    print(f"[7.2] Simulation completed in {elapsed:.2f} s ({elapsed / B:.4f} s/path)")
    
    flat_rows = [row for path_res in all_path_results for row in path_res]
    dist_df = pd.DataFrame(flat_rows)
    
    expected_rows = B * 8
    assert len(dist_df) == expected_rows, f"Expected {expected_rows} rows, got {len(dist_df)}"
    assert dist_df.isna().sum().sum() == 0, f"NaNs found in bootstrap distribution:\n{dist_df.isna().sum()}"
    
    return dist_df


def main() -> None:
    print("=" * 72)
    print("P H A S E  7  --  S T E P  7 . 2   B L O C K   B O O T S T R A P")
    print("=" * 72)
    
    dist_df = run_block_bootstrap(B=B_DEFAULT)
    dist_df.to_csv(OUT_DISTRIBUTIONS_CSV, index=False)
    print(f"[7.2] Saved distributions -> {OUT_DISTRIBUTIONS_CSV}")
    print(f"      Shape: {dist_df.shape} (rows x cols)")
    
    summary = dist_df.groupby("strategy")["sharpe"].agg(["mean", "std", "median"])
    print("\n[7.2] Bootstrap Sharpe Summary across B paths:")
    print(summary.sort_values(by="median", ascending=False).to_string())
    print("\n[PASS] Step 7.2 block bootstrap simulation verified.")


if __name__ == "__main__":
    main()
