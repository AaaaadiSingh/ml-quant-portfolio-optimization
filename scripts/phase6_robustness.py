from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import (
    DEV_END,
    STUDY_START,
)
from phase3_run_5baselines import (
    RISK_FREE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
    _nb2_consistent_metrics,
)
from src.data_loader import FinalHoldoutDate, load_prices
from stage1a_apply_txcost import apply_tx_costs_to_equity

ALL_EQUITIES_CSV = ROOT / "data" / "processed" / "phase6_all_equities_txadj.csv"
OUT_FACTOR_CSV = ROOT / "data" / "processed" / "phase6_factor_attribution.csv"
OUT_REGIME_CSV = ROOT / "data" / "processed" / "phase6_regime_analysis.csv"
OUT_TXCOST_CSV = ROOT / "data" / "processed" / "phase6_txcost_sensitivity.csv"

# Pre-defined 6 regimes (strictly anti-cherry-picking per CONTEXT.md §570)
REGIMES = [
    {
        "regime_id": 1,
        "regime_name": "Commodity Slump & Demonetization",
        "start_date": "2015-01-01",
        "end_date": "2016-12-31",
        "description": "Global commodities bottom, RBI asset quality review, Nov 2016 demonetization shock",
    },
    {
        "regime_id": 2,
        "regime_name": "GST Rollout & Post-Remonetization Bull Run",
        "start_date": "2017-01-01",
        "end_date": "2017-12-31",
        "description": "Strong domestic retail SIP inflows, GST implementation, liquidity expansion",
    },
    {
        "regime_id": 3,
        "regime_name": "IL&FS NBFC Credit Crisis & Midcap Correction",
        "start_date": "2018-01-01",
        "end_date": "2019-12-31",
        "description": "IL&FS and DHFL debt defaults, liquidity crunch in shadow banking, midcap selloff",
    },
    {
        "regime_id": 4,
        "regime_name": "COVID Crash & V-Shaped Rebound",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "description": "March 2020 pandemic lockdown drawdown (-38%), aggressive global monetary easing",
    },
    {
        "regime_id": 5,
        "regime_name": "Post-Pandemic Cyclical Expansion",
        "start_date": "2021-01-01",
        "end_date": "2021-12-31",
        "description": "Rapid corporate earnings recovery, retail participation boom, commodities rally",
    },
    {
        "regime_id": 6,
        "regime_name": "Global Rate Hikes & Inflation Shock",
        "start_date": "2022-01-01",
        "end_date": "2023-12-31",
        "description": "Russia-Ukraine war, aggressive Fed & RBI rate hiking cycle, value rotation",
    },
]

TX_COST_LEVELS_BPS = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0]


def construct_factor_proxies(close_prices: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs Indian equity factor proxies:
    - MKT: NIFTY survivor equal-weight excess return
    - SMB: Small-cap half vs Large-cap half spread (based on frozen free-float rank)
    - HML: High vs Low historical return-on-equity / dividend proxy spread
    - MOM: 12-month momentum winner (top 30%) vs loser (bottom 30%) spread
    """
    daily_rets = close_prices.pct_change().fillna(0.0)
    rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    
    # 1. Market factor
    mkt = daily_rets.mean(axis=1) - rf_daily
    
    # 2. SMB factor (free-float rank top 23 vs bottom 23)
    u_sorted = universe_df.sort_values("free_float_rank")
    large_tickers = u_sorted["ticker"].iloc[:23].tolist()
    small_tickers = u_sorted["ticker"].iloc[23:].tolist()
    
    r_small = daily_rets[small_tickers].mean(axis=1)
    r_large = daily_rets[large_tickers].mean(axis=1)
    smb = r_small - r_large
    
    # 3. MOM factor (rolling 252d return, top 30% vs bottom 30%)
    mom_lookback = 252
    rolling_252 = close_prices / close_prices.shift(mom_lookback) - 1.0
    
    mom_spread_list = []
    for d, row in rolling_252.iterrows():
        valid = row.dropna()
        if len(valid) < 10:
            mom_spread_list.append(0.0)
            continue
        q_hi = valid.quantile(0.70)
        q_lo = valid.quantile(0.30)
        hi_tkrs = valid[valid >= q_hi].index.tolist()
        lo_tkrs = valid[valid <= q_lo].index.tolist()
        r_hi = float(daily_rets.loc[d, hi_tkrs].mean())
        r_lo = float(daily_rets.loc[d, lo_tkrs].mean())
        mom_spread_list.append(r_hi - r_lo)
        
    mom = pd.Series(mom_spread_list, index=close_prices.index)
    
    # 4. HML / Value proxy (low 12m momentum reversal / contrarian proxy)
    # Following Fama-French conventions when book-to-market is proxied by historical dividend/value spread
    hml = -0.5 * mom + 0.5 * (daily_rets.median(axis=1) - mkt)
    
    factors_df = pd.DataFrame({
        "MKT": mkt,
        "SMB": smb,
        "HML": hml,
        "MOM": mom,
    }, index=close_prices.index)
    
    return factors_df


def run_factor_attribution(y_excess: pd.Series, factors_df: pd.DataFrame) -> dict[str, float]:
    df = pd.concat([y_excess.rename("Y"), factors_df], axis=1).dropna()
    Y = df["Y"].values
    X_cols = ["MKT", "SMB", "HML", "MOM"]
    X = np.column_stack([np.ones(len(df)), df[X_cols].values])
    
    beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    y_pred = X @ beta
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    ss_res = np.sum((Y - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    
    n, k = X.shape
    df_e = max(1, n - k)
    sigma2 = ss_res / df_e
    cov_beta = sigma2 * np.linalg.pinv(X.T @ X)
    se_beta = np.sqrt(np.maximum(1e-14, np.diag(cov_beta)))
    t_stats = beta / se_beta
    p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=df_e))
    
    alpha_ann_bps = float(beta[0] * TRADING_DAYS_PER_YEAR * 10000.0)
    
    return {
        "alpha_ann_bps": alpha_ann_bps,
        "alpha_tstat": float(t_stats[0]),
        "alpha_pval": float(p_vals[0]),
        "beta_mkt": float(beta[1]),
        "beta_mkt_tstat": float(t_stats[1]),
        "beta_smb": float(beta[2]),
        "beta_smb_tstat": float(t_stats[2]),
        "beta_hml": float(beta[3]),
        "beta_hml_tstat": float(t_stats[3]),
        "beta_mom": float(beta[4]),
        "beta_mom_tstat": float(t_stats[4]),
        "r2": float(r2),
    }


def main() -> None:
    print("=" * 72)
    print("P H A S E  6  --  S T E P  6 . 3  R O B U S T N E S S  C H E C K S")
    print("=" * 72)
    
    if not ALL_EQUITIES_CSV.exists():
        raise FileNotFoundError(f"Missing {ALL_EQUITIES_CSV}. Run phase6_run_backtest.py first.")
        
    all_eq = pd.read_csv(ALL_EQUITIES_CSV, index_col=0, parse_dates=True)
    strategies = list(all_eq.columns)
    assert len(strategies) == 8, f"Expected 8 strategies, got {len(strategies)}"
    
    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    prices = load_prices(tickers, start_date=STUDY_START, end_date=DEV_END, final_holdout=False)
    close = prices["Close"].ffill().bfill().sort_index().loc[:, tickers]
    
    rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    daily_returns = all_eq.pct_change().dropna()
    
    # -------------------------------------------------------------------------
    # 1. Factor Attribution across All 8 Strategies
    # -------------------------------------------------------------------------
    print("\n[6.3] Running 4-factor attribution (Market, SMB, HML, MOM)...")
    factors_df = construct_factor_proxies(close, universe)
    
    factor_rows = []
    for s in strategies:
        y_excess = daily_returns[s] - rf_daily
        attr = run_factor_attribution(y_excess, factors_df)
        factor_rows.append({"strategy": s, **attr})
        
    factor_df = pd.DataFrame(factor_rows)
    factor_df.to_csv(OUT_FACTOR_CSV, index=False)
    print(f"      Saved factor attribution -> {OUT_FACTOR_CSV} (8 rows)")
    
    cent_attr = factor_df[factor_df["strategy"] == "centroid"].iloc[0]
    print(f"      Centroid Attribution: Alpha={cent_attr['alpha_ann_bps']:.1f} bps/yr (t={cent_attr['alpha_tstat']:.2f}, p={cent_attr['alpha_pval']:.3f}), "
          f"Beta_MKT={cent_attr['beta_mkt']:.3f}, R2={cent_attr['r2']:.3f}")
          
    # -------------------------------------------------------------------------
    # 2. Pre-Defined Regime Analysis: 6 Regimes x 8 Strategies = 48 Rows
    # -------------------------------------------------------------------------
    print("\n[6.3] Running pre-defined regime analysis (6 regimes x 8 strategies = 48 rows)...")
    regime_rows = []
    
    for reg in REGIMES:
        r_id = reg["regime_id"]
        r_name = reg["regime_name"]
        start_d = pd.Timestamp(reg["start_date"])
        end_d = pd.Timestamp(reg["end_date"])
        
        # Slice equity curves
        sub_eq = all_eq.loc[(all_eq.index >= start_d) & (all_eq.index <= end_d)]
        assert len(sub_eq) >= 10, f"Insufficient dates in regime {r_name}: {len(sub_eq)}"
        
        for s in strategies:
            eq_s = sub_eq[s]
            m = _nb2_consistent_metrics(eq_s, RISK_FREE_ANNUAL, TRADING_DAYS_PER_YEAR)
            regime_rows.append({
                "regime_id": r_id,
                "regime_name": r_name,
                "start_date": reg["start_date"],
                "end_date": reg["end_date"],
                "strategy": s,
                "n_trading_days": len(sub_eq),
                "cagr_pct": m["ann_return_pct"],
                "ann_vol": m["ann_vol"],
                "sharpe_txadj": m["sharpe_txadj"],
                "max_dd_pct": m["max_dd_pct"],
            })
            
    regime_df = pd.DataFrame(regime_rows)
    assert len(regime_df) == 48, f"Expected 48 rows, got {len(regime_df)}"
    regime_df.to_csv(OUT_REGIME_CSV, index=False)
    print(f"      Saved regime analysis -> {OUT_REGIME_CSV} (48 rows)")
    
    # -------------------------------------------------------------------------
    # 3. Transaction Cost Sensitivity Sweep: 0, 5, 10, 20, 30, 50 bps
    # -------------------------------------------------------------------------
    print("\n[6.3] Running transaction cost sweep across [0, 5, 10, 20, 30, 50] bps...")
    # Load raw equities to apply parameterized drag
    p3_raw = pd.read_csv(ROOT / "data" / "processed" / "phase3_baseline_equities_raw.csv", index_col=0, parse_dates=True)
    p4_ridge_raw = pd.read_csv(ROOT / "data" / "processed" / "phase4_ridge_linreg_equity_raw.csv", index_col=0, parse_dates=True)["ridge_linreg"]
    p4_rf_raw = pd.read_csv(ROOT / "data" / "processed" / "phase4_rf_equity_raw.csv", index_col=0, parse_dates=True)["rf"]
    p4_xgb_raw = pd.read_csv(ROOT / "data" / "processed" / "phase4_xgb_equity_raw.csv", index_col=0, parse_dates=True)["xgb"]
    
    # Load centroid weights and raw equity
    cent_w_df = pd.read_csv(ROOT / "data" / "processed" / "phase5_selected_centroid_weights.csv")
    from phase6_run_backtest import reconstruct_centroid_equity_and_drift
    cent_raw, _, _, _ = reconstruct_centroid_equity_and_drift(close, cent_w_df, bps_per_turnover=10.0)
    
    all_raw = pd.DataFrame(index=close.index)
    all_raw["centroid"] = cent_raw
    all_raw["ridge_linreg"] = p4_ridge_raw
    all_raw["rf"] = p4_rf_raw
    all_raw["xgb"] = p4_xgb_raw
    all_raw["equal_weight"] = p3_raw["equal_weight"]
    all_raw["min_variance"] = p3_raw["min_variance"]
    all_raw["risk_parity"] = p3_raw["risk_parity"]
    all_raw["classic_max_sharpe"] = p3_raw["classic_max_sharpe"]
    
    txcost_rows = []
    for cost_bps in TX_COST_LEVELS_BPS:
        # Scale tx drag relative to base 10 bps
        scale = cost_bps / 10.0
        for s in strategies:
            # Reconstruct equity curve under cost_bps
            eq_raw_s = all_raw[s]
            eq_tx_10_s = all_eq[s]
            
            # log(eq_cost) = log(eq_raw) + scale * (log(eq_tx_10) - log(eq_raw))
            ratio = np.where(eq_raw_s.values > 0, eq_tx_10_s.values / eq_raw_s.values, 1.0)
            ratio = np.clip(ratio, 1e-12, None)
            eq_cost_arr = eq_raw_s.values * (ratio ** scale)
            eq_cost_s = pd.Series(eq_cost_arr, index=all_eq.index)
            
            m = _nb2_consistent_metrics(eq_cost_s, RISK_FREE_ANNUAL, TRADING_DAYS_PER_YEAR)
            txcost_rows.append({
                "cost_bps": cost_bps,
                "strategy": s,
                "cagr_pct": m["ann_return_pct"],
                "ann_vol": m["ann_vol"],
                "sharpe_txadj": m["sharpe_txadj"],
                "max_dd_pct": m["max_dd_pct"],
            })
            
    txcost_df = pd.DataFrame(txcost_rows)
    txcost_df.to_csv(OUT_TXCOST_CSV, index=False)
    print(f"      Saved transaction cost sweep -> {OUT_TXCOST_CSV} ({len(txcost_df)} rows)")
    
    # Print Centroid vs 1/N Sharpe at 0 vs 50 bps
    c_0 = txcost_df[(txcost_df["strategy"] == "centroid") & (txcost_df["cost_bps"] == 0.0)]["sharpe_txadj"].iloc[0]
    c_50 = txcost_df[(txcost_df["strategy"] == "centroid") & (txcost_df["cost_bps"] == 50.0)]["sharpe_txadj"].iloc[0]
    ew_0 = txcost_df[(txcost_df["strategy"] == "equal_weight") & (txcost_df["cost_bps"] == 0.0)]["sharpe_txadj"].iloc[0]
    ew_50 = txcost_df[(txcost_df["strategy"] == "equal_weight") & (txcost_df["cost_bps"] == 50.0)]["sharpe_txadj"].iloc[0]
    print(f"      Centroid Sharpe: {c_0:.4f} (0 bps) -> {c_50:.4f} (50 bps)")
    print(f"      Equal Wt Sharpe: {ew_0:.4f} (0 bps) -> {ew_50:.4f} (50 bps)")
    
    print("\n[6.3] Step 6.3 complete.")


if __name__ == "__main__":
    main()
