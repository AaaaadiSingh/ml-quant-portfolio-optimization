from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.phase7_run_block_bootstrap import (
    B_DEFAULT,
    SEED,
    STUDY_START,
    DEV_END,
    load_strategy_weights,
    simulate_strategy_fast,
)
from src.data_loader import FinalHoldoutDate, load_prices

OUT_REGIME_CSV = ROOT / "data" / "processed" / "phase7_regime_vol_tercile.csv"
BLOCK_LENGTH_CSV = ROOT / "data" / "processed" / "phase7_block_length.csv"
PHASE6_EQUITIES_CSV = ROOT / "data" / "processed" / "phase6_all_equities_txadj.csv"

REGIMES = ["low_vol", "mid_vol", "high_vol"]


def compute_historical_vol_terciles() -> tuple[float, float]:
    assert PHASE6_EQUITIES_CSV.exists(), f"Missing {PHASE6_EQUITIES_CSV}"
    p6_df = pd.read_csv(PHASE6_EQUITIES_CSV, index_col=0, parse_dates=True)
    ew_ret = p6_df["equal_weight"].pct_change().dropna()
    roll_vol = ew_ret.rolling(21).std() * np.sqrt(252)
    roll_vol = roll_vol.dropna()
    q33 = float(roll_vol.quantile(1.0 / 3.0))
    q67 = float(roll_vol.quantile(2.0 / 3.0))
    return q33, q67


def evaluate_path_regimes(
    path_id: int,
    synth_log_ret: np.ndarray,
    weights_by_strat: Dict[str, Dict[int, np.ndarray]],
    n_days: int,
    q33: float,
    q67: float,
    rf_daily: float,
) -> List[dict]:
    synth_simple_ret = np.exp(synth_log_ret) - 1.0
    
    # Simulate all strategies on this path
    equities = {
        strat: simulate_strategy_fast(synth_simple_ret, locs, n_days)
        for strat, locs in weights_by_strat.items()
    }
    
    # Calculate rolling 21-day annualized volatility of equal-weight portfolio
    ew_eq = equities["equal_weight"]
    ew_ret = ew_eq[1:] / ew_eq[:-1] - 1.0
    ew_s = pd.Series(ew_ret)
    roll_vol = (ew_s.rolling(21).std() * np.sqrt(252)).values  # length n_days - 1
    
    # Regime masks for days 1 to n_days - 1
    # Day t corresponds to daily return at index t-1
    valid_vol = np.isfinite(roll_vol)
    masks = {
        "low_vol": valid_vol & (roll_vol <= q33),
        "mid_vol": valid_vol & (roll_vol > q33) & (roll_vol <= q67),
        "high_vol": valid_vol & (roll_vol > q67),
    }
    
    rows: List[dict] = []
    for reg_name, mask in masks.items():
        if mask.sum() < 10:
            continue
        for strat, eq in equities.items():
            strat_ret = (eq[1:] / eq[:-1] - 1.0)[mask]
            std = float(np.std(strat_ret, ddof=1))
            if std > 1e-12:
                sh = float((np.mean(strat_ret) - rf_daily) / std * np.sqrt(252))
            else:
                sh = 0.0
            rows.append(
                {
                    "path_id": path_id,
                    "regime": reg_name,
                    "strategy": strat,
                    "sharpe": sh,
                }
            )
    return rows


def run_regime_bootstrap_slice(B: int = B_DEFAULT) -> pd.DataFrame:
    assert BLOCK_LENGTH_CSV.exists(), f"Missing {BLOCK_LENGTH_CSV}"
    bl_df = pd.read_csv(BLOCK_LENGTH_CSV)
    L = int(bl_df["chosen_L"].iloc[0])
    
    q33, q67 = compute_historical_vol_terciles()
    print(f"[7.4] Realized Volatility Tercile Cutoffs (21d rolling):")
    print(f"      Low-Vol <= {q33:.4f} | Mid-Vol in ({q33:.4f}, {q67:.4f}] | High-Vol > {q67:.4f}")
    
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
    assert len(close) == 2222
    
    log_ret = np.log(close / close.shift(1)).dropna().values
    weights_by_strat = load_strategy_weights(tickers, close.index)
    rf_daily = (1.0 + 0.04) ** (1.0 / 252.0) - 1.0
    
    print(f"[7.4] Generating {B} synthetic paths via StationaryBootstrap (L={L}, SEED={SEED})...")
    bs = StationaryBootstrap(L, log_ret, seed=SEED)
    synthetic_matrices: List[np.ndarray] = []
    for pos_args, _ in bs.bootstrap(B):
        synthetic_matrices.append(pos_args[0])
        
    print(f"[7.4] Evaluating volatility regimes across {B} paths in parallel...")
    t0 = time.time()
    path_reg_results = Parallel(n_jobs=-1, batch_size=5)(
        delayed(evaluate_path_regimes)(
            b,
            synthetic_matrices[b],
            weights_by_strat,
            len(close),
            q33,
            q67,
            rf_daily,
        )
        for b in range(B)
    )
    elapsed = time.time() - t0
    print(f"[7.4] Completed in {elapsed:.2f} s")
    
    flat_rows = [row for path_res in path_reg_results for row in path_res]
    raw_df = pd.DataFrame(flat_rows)
    
    # Aggregate distribution across B paths for each regime and strategy
    summary_rows: List[dict] = []
    for reg in REGIMES:
        for strat in sorted(weights_by_strat.keys()):
            sub = raw_df[(raw_df["regime"] == reg) & (raw_df["strategy"] == strat)]
            assert len(sub) == B, f"Expected {B} paths for {reg} x {strat}, got {len(sub)}"
            vals = sub["sharpe"].values.astype(float)
            summary_rows.append(
                {
                    "regime": reg,
                    "strategy": strat,
                    "sharpe_mean": float(np.mean(vals)),
                    "sharpe_std": float(np.std(vals, ddof=1)),
                    "sharpe_p5": float(np.percentile(vals, 5)),
                    "sharpe_p50": float(np.percentile(vals, 50)),
                    "sharpe_p95": float(np.percentile(vals, 95)),
                }
            )
            
    summary_df = pd.DataFrame(summary_rows)
    assert len(summary_df) == 24, f"Expected 24 rows (3 regimes x 8 strategies), got {len(summary_df)}"
    assert not summary_df.isna().any().any(), "NaN found in summary table!"
    return summary_df


def main() -> None:
    print("=" * 72)
    print("P H A S E  7  --  S T E P  7 . 4   V O L - R E G I M E   S L I C E")
    print("=" * 72)
    
    summary_df = run_regime_bootstrap_slice(B=B_DEFAULT)
    summary_df.to_csv(OUT_REGIME_CSV, index=False)
    print(f"[7.4] Saved regime summary -> {OUT_REGIME_CSV}")
    print(f"      Shape: {summary_df.shape} (3 regimes x 8 strategies)")
    
    piv = summary_df.pivot(index="regime", columns="strategy", values="sharpe_p50")
    print("\n[7.4] Median Sharpe Across B Paths by Realized Volatility Tercile:")
    print(piv[["centroid", "equal_weight", "min_variance", "classic_max_sharpe", "ridge_linreg"]].to_string())
    
    print("\n[PASS] Step 7.4 volatility regime slice verified.")


if __name__ == "__main__":
    main()
