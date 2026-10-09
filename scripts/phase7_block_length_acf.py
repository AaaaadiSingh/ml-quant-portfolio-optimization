from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from arch.bootstrap import optimal_block_length

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import FinalHoldoutDate, load_prices

STUDY_START = "2015-01-01"
DEV_END = "2023-12-31"

OUT_BLOCK_LENGTH_CSV = ROOT / "data" / "processed" / "phase7_block_length.csv"
OUT_ACF_SERIES_CSV = ROOT / "data" / "processed" / "phase7_acf_squared_returns.csv"


def compute_acf_and_block_length(
    max_lag: int = 60,
) -> tuple[int, int, float, pd.DataFrame, pd.DataFrame]:
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
    assert len(close) == 2222, f"Expected 2222 dates, got {len(close)}"
    
    log_ret = np.log(close / close.shift(1)).dropna()
    T = len(log_ret)
    assert T == 2221, f"Expected 2221 return rows, got {T}"
    
    # Cross-sectional market portfolio return and its squared returns
    market_ret = log_ret.mean(axis=1)
    sq_ret = market_ret ** 2
    
    acf_vals = acf(sq_ret, nlags=max_lag, fft=True)
    ci_val = 1.96 / np.sqrt(T)
    
    # First lag >= 1 where ACF drops below 95% CI upper band
    cutoff_idx = np.where(acf_vals[1:] < ci_val)[0]
    if len(cutoff_idx) > 0:
        acf_cutoff_lag = int(cutoff_idx[0] + 1)
    else:
        acf_cutoff_lag = max_lag
        
    chosen_L = acf_cutoff_lag
    
    # Politis-White (2004) cross-check on market squared returns
    pw_df = optimal_block_length(sq_ret)
    pw_stat = float(pw_df["stationary"].iloc[0]) if "stationary" in pw_df.columns else float(pw_df.iloc[0, 0])
    
    # Build ACF series dataframe for plotting
    acf_df = pd.DataFrame(
        {
            "lag": list(range(max_lag + 1)),
            "acf": acf_vals,
            "ci_upper": [ci_val] * (max_lag + 1),
            "ci_lower": [-ci_val] * (max_lag + 1),
        }
    )
    
    # Build summary dataframe
    summary_df = pd.DataFrame(
        [
            {
                "acf_cutoff_lag": acf_cutoff_lag,
                "chosen_L": chosen_L,
                "T": T,
                "ci_threshold": ci_val,
                "acf_at_cutoff": float(acf_vals[acf_cutoff_lag]),
                "pw_stationary": pw_stat,
            }
        ]
    )
    
    return acf_cutoff_lag, chosen_L, ci_val, summary_df, acf_df


def main() -> None:
    print("=" * 72)
    print("P H A S E  7  --  S T E P  7 . 1   B L O C K   L E N G T H   A C F")
    print("=" * 72)
    
    cutoff, chosen_L, ci, summary_df, acf_df = compute_acf_and_block_length(max_lag=60)
    
    print(f"[7.1] Return series length T: {summary_df['T'].iloc[0]}")
    print(f"      95% CI threshold (1.96 / sqrt(T)): {ci:.4f}")
    print(f"      ACF cutoff lag (squared returns): {cutoff}")
    print(f"      Chosen block length L: {chosen_L}")
    print(f"      Politis-White stationary L estimate: {summary_df['pw_stationary'].iloc[0]:.2f}")
    
    summary_df.to_csv(OUT_BLOCK_LENGTH_CSV, index=False)
    print(f"      Saved block length spec -> {OUT_BLOCK_LENGTH_CSV}")
    
    acf_df.to_csv(OUT_ACF_SERIES_CSV, index=False)
    print(f"      Saved ACF series -> {OUT_ACF_SERIES_CSV}")
    
    assert 5 <= chosen_L <= 60, f"chosen_L={chosen_L} outside expected range [5, 60]"
    print("[PASS] Step 7.1 block length selection verified.")


if __name__ == "__main__":
    main()
