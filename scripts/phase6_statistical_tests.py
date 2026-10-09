from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase3_run_5baselines import (
    RISK_FREE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)

ALL_EQUITIES_CSV = ROOT / "data" / "processed" / "phase6_all_equities_txadj.csv"
OUT_JK_CSV = ROOT / "data" / "processed" / "phase6_jk_pairwise_tests.csv"
OUT_DSR_CSV = ROOT / "data" / "processed" / "phase6_dsr_summary.csv"
OUT_PBO_CSV = ROOT / "data" / "processed" / "phase6_pbo_results.csv"

EULER_MASCHERONI = 0.57721566490153286
N_CONFIGS_DSR = 24  # Documented in literature_matrix.md §C D12: 2 freq x 2 Sigma x 3 ML x 2 MC


def jobson_korkie_memmel(
    r1: pd.Series,
    r2: pd.Series,
    rf_daily: float,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """
    Jobson-Korkie (1981) test with Memmel (2003) corrected asymptotic variance.
    Tests H0: Sharpe_1 == Sharpe_2 against H1: Sharpe_1 != Sharpe_2.
    """
    df = pd.DataFrame({"r1": r1, "r2": r2}).dropna()
    T = len(df)
    if T < 10:
        return {
            "sharpe_1_ann": float("nan"),
            "sharpe_2_ann": float("nan"),
            "delta_sharpe_ann": float("nan"),
            "z_stat": float("nan"),
            "p_value": float("nan"),
        }
        
    R1 = df["r1"] - rf_daily
    R2 = df["r2"] - rf_daily
    
    mu1 = float(R1.mean())
    mu2 = float(R2.mean())
    s1 = float(R1.std(ddof=1))
    s2 = float(R2.std(ddof=1))
    cov12 = float(R1.cov(R2))
    rho = cov12 / (s1 * s2) if (s1 * s2) > 1e-12 else 0.0
    
    sr1_daily = mu1 / s1 if s1 > 1e-12 else 0.0
    sr2_daily = mu2 / s2 if s2 > 1e-12 else 0.0
    
    # Memmel (2003) asymptotic variance formula for daily Sharpe difference
    term1 = 2.0 * (1.0 - rho)
    term2 = 0.5 * (sr1_daily ** 2 + sr2_daily ** 2 - 2.0 * sr1_daily * sr2_daily * (rho ** 2))
    var_delta = (term1 + term2) / T
    
    delta_daily = sr1_daily - sr2_daily
    se_delta = math.sqrt(var_delta) if var_delta > 0 else 1e-12
    z_stat = delta_daily / se_delta
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    
    sqrt_ann = math.sqrt(trading_days)
    return {
        "sharpe_1_ann": float(sr1_daily * sqrt_ann),
        "sharpe_2_ann": float(sr2_daily * sqrt_ann),
        "delta_sharpe_ann": float(delta_daily * sqrt_ann),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
    }


def compute_dsr(
    daily_returns: pd.Series,
    rf_daily: float,
    n_configs: int = N_CONFIGS_DSR,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    benchmark_sharpes: list[float] | None = None,
) -> dict[str, float]:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
    Adjusts for track record length, non-normality (skew/kurt), and N sequential tests.
    """
    excess = (daily_returns - rf_daily).dropna()
    T = len(excess)
    mu = float(excess.mean())
    sigma = float(excess.std(ddof=1))
    sr_daily = mu / sigma if sigma > 1e-12 else 0.0
    sr_ann = float(sr_daily * math.sqrt(trading_days))
    
    skew = float(stats.skew(excess))
    kurt = float(stats.kurtosis(excess, fisher=False))  # Pearson kurtosis (normal = 3)
    
    # Variance of the estimated Sharpe ratio under non-normality
    var_sr_ann = (1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * (sr_daily ** 2)) / T * trading_days
    se_sr_ann = math.sqrt(max(1e-12, var_sr_ann))
    
    # Expected maximum Sharpe ratio under null among N trials
    if benchmark_sharpes is not None and len(benchmark_sharpes) > 1:
        sigma_trials = float(np.std(benchmark_sharpes, ddof=1))
    else:
        sigma_trials = float(se_sr_ann)
        
    p1 = stats.norm.ppf(1.0 - 1.0 / n_configs)
    p2 = stats.norm.ppf(1.0 - 1.0 / (n_configs * math.e))
    e_max_sr = sigma_trials * ((1.0 - EULER_MASCHERONI) * p1 + EULER_MASCHERONI * p2)
    
    dsr_stat = (sr_ann - e_max_sr) / se_sr_ann
    dsr_prob = float(stats.norm.cdf(dsr_stat))
    
    return {
        "sharpe_ann": sr_ann,
        "skewness": skew,
        "kurtosis_pearson": kurt,
        "kurtosis_excess": kurt - 3.0,
        "t_obs": float(T),
        "n_configs": float(n_configs),
        "sigma_trials": sigma_trials,
        "e_max_sharpe": float(e_max_sr),
        "se_sharpe": float(se_sr_ann),
        "dsr_z": float(dsr_stat),
        "dsr_prob": float(dsr_prob),
    }


def compute_cscv_pbo(
    returns_df: pd.DataFrame,
    rf_daily: float,
    s_splits: int = 6,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Combinatorially Symmetric Cross-Validation (CSCV) and Probability of Backtest Overfitting (PBO).
    Splits T observations into S contiguous slices, tests all C(S, S/2) combinations.
    """
    T = len(returns_df)
    slice_size = T // s_splits
    slices = []
    for i in range(s_splits):
        start_idx = i * slice_size
        end_idx = (i + 1) * slice_size if i < s_splits - 1 else T
        slices.append(returns_df.iloc[start_idx:end_idx])
        
    k_is = s_splits // 2
    combinations = list(itertools.combinations(range(s_splits), k_is))
    n_combs = len(combinations)
    
    comb_records = []
    underperform_count = 0
    strategies = list(returns_df.columns)
    
    for c_id, is_indices in enumerate(combinations):
        oos_indices = [i for i in range(s_splits) if i not in is_indices]
        
        df_is = pd.concat([slices[i] for i in is_indices])
        df_oos = pd.concat([slices[i] for i in oos_indices])
        
        # Calculate Sharpe for all strategies IS
        is_sharpes = {}
        for s in strategies:
            r = df_is[s] - rf_daily
            std = float(r.std(ddof=1))
            is_sharpes[s] = float(r.mean() / std * math.sqrt(trading_days)) if std > 1e-12 else -999.0
            
        # Best strategy IS
        best_strat_is = max(is_sharpes.keys(), key=lambda k: is_sharpes[k])
        
        # Calculate Sharpe for all strategies OOS
        oos_sharpes = {}
        for s in strategies:
            r = df_oos[s] - rf_daily
            std = float(r.std(ddof=1))
            oos_sharpes[s] = float(r.mean() / std * math.sqrt(trading_days)) if std > 1e-12 else -999.0
            
        # Rank of best_strat_is in OOS
        sorted_oos = sorted(oos_sharpes.keys(), key=lambda k: oos_sharpes[k])
        rank_oos = sorted_oos.index(best_strat_is)  # 0 to len-1
        rel_rank = float(rank_oos) / float(len(strategies) - 1)
        
        is_overfit = bool(rel_rank < 0.5)
        if is_overfit:
            underperform_count += 1
            
        comb_records.append({
            "combination_id": c_id + 1,
            "is_slices": str(list(is_indices)),
            "oos_slices": str(list(oos_indices)),
            "best_is_strategy": best_strat_is,
            "best_is_sharpe": is_sharpes[best_strat_is],
            "oos_sharpe_best_is": oos_sharpes[best_strat_is],
            "oos_rank_best_is": rank_oos + 1,
            "oos_relative_rank": rel_rank,
            "overfit_flag": is_overfit,
        })
        
    pbo = float(underperform_count) / float(n_combs)
    combs_df = pd.DataFrame(comb_records)
    
    summary_df = pd.DataFrame([{
        "n_slices": s_splits,
        "n_combinations": n_combs,
        "n_strategies": len(strategies),
        "overfit_combinations": underperform_count,
        "pbo": pbo,
        "median_oos_rel_rank": float(combs_df["oos_relative_rank"].median()),
        "mean_oos_rel_rank": float(combs_df["oos_relative_rank"].mean()),
    }])
    
    return pbo, summary_df, combs_df


def main() -> None:
    print("=" * 72)
    print("P H A S E  6  --  S T E P  6 . 2  S T A T I S T I C A L  T E S T S")
    print("=" * 72)
    
    if not ALL_EQUITIES_CSV.exists():
        raise FileNotFoundError(f"Missing {ALL_EQUITIES_CSV}. Run phase6_run_backtest.py first.")
        
    all_eq = pd.read_csv(ALL_EQUITIES_CSV, index_col=0, parse_dates=True)
    strategies = list(all_eq.columns)
    assert len(strategies) == 8, f"Expected 8 strategies, got {len(strategies)}: {strategies}"
    print(f"[6.2] Loaded 8 strategies across {len(all_eq)} dates: {strategies}")
    
    rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    daily_returns = all_eq.pct_change().dropna()
    
    # -------------------------------------------------------------------------
    # 1. Jobson-Korkie (Memmel 2003) Pairwise Tests: 8C2 = 28 pairs
    # -------------------------------------------------------------------------
    print("\n[6.2] Running Jobson-Korkie (Memmel 2003) tests across all 28 pairs (8C2)...")
    pairs = list(itertools.combinations(strategies, 2))
    assert len(pairs) == 28, f"Expected 28 pairs, got {len(pairs)}"
    
    alpha_5pct = 0.05
    alpha_bonferroni = 0.05 / 28.0
    
    jk_rows = []
    for s1, s2 in pairs:
        res = jobson_korkie_memmel(daily_returns[s1], daily_returns[s2], rf_daily)
        p_val = res["p_value"]
        jk_rows.append({
            "strat_1": s1,
            "strat_2": s2,
            "sharpe_1": res["sharpe_1_ann"],
            "sharpe_2": res["sharpe_2_ann"],
            "delta_sharpe": res["delta_sharpe_ann"],
            "z_stat": res["z_stat"],
            "p_value": p_val,
            "sig_at_5pct": bool(p_val < alpha_5pct),
            "sig_bonferroni": bool(p_val < alpha_bonferroni),
        })
        
    jk_df = pd.DataFrame(jk_rows)
    assert len(jk_df) == 28, f"Expected 28 rows, got {len(jk_df)}"
    jk_df.to_csv(OUT_JK_CSV, index=False)
    print(f"      Saved Jobson-Korkie pairwise tests -> {OUT_JK_CSV} (28 rows)")
    
    # Print key pairwise comparisons
    c_vs_1n = jk_df[((jk_df["strat_1"] == "centroid") & (jk_df["strat_2"] == "equal_weight")) |
                    ((jk_df["strat_1"] == "equal_weight") & (jk_df["strat_2"] == "centroid"))].iloc[0]
    print(f"      Key Pair: Centroid vs 1/N: Delta_Sharpe={c_vs_1n['delta_sharpe']:.4f}, z={c_vs_1n['z_stat']:.3f}, p={c_vs_1n['p_value']:.4f}")
    
    c_vs_ridge = jk_df[((jk_df["strat_1"] == "centroid") & (jk_df["strat_2"] == "ridge_linreg")) |
                       ((jk_df["strat_1"] == "ridge_linreg") & (jk_df["strat_2"] == "centroid"))].iloc[0]
    print(f"      Key Pair: Centroid vs Ridge: Delta_Sharpe={c_vs_ridge['delta_sharpe']:.4f}, z={c_vs_ridge['z_stat']:.3f}, p={c_vs_ridge['p_value']:.4f}")
    
    # -------------------------------------------------------------------------
    # 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
    # -------------------------------------------------------------------------
    print(f"\n[6.2] Computing Deflated Sharpe Ratio (N={N_CONFIGS_DSR} implicit sequential configs)...")
    strategy_sharpes = [float(jobson_korkie_memmel(daily_returns[s], daily_returns[s], rf_daily)["sharpe_1_ann"]) for s in strategies]
    
    dsr_rows = []
    for s in strategies:
        dsr_res = compute_dsr(
            daily_returns[s],
            rf_daily=rf_daily,
            n_configs=N_CONFIGS_DSR,
            benchmark_sharpes=strategy_sharpes,
        )
        dsr_rows.append({"strategy": s, **dsr_res})
        
    dsr_df = pd.DataFrame(dsr_rows)
    dsr_df.to_csv(OUT_DSR_CSV, index=False)
    print(f"      Saved DSR summary -> {OUT_DSR_CSV} ({len(dsr_df)} rows)")
    
    cent_dsr = dsr_df[dsr_df["strategy"] == "centroid"].iloc[0]
    print(f"      Centroid DSR: Sharpe={cent_dsr['sharpe_ann']:.4f}, E[max S]={cent_dsr['e_max_sharpe']:.4f}, DSR p-prob={cent_dsr['dsr_prob']:.4f}")
    assert cent_dsr["dsr_prob"] > 0.0, f"Centroid DSR must be > 0, got {cent_dsr['dsr_prob']}"
    
    # -------------------------------------------------------------------------
    # 3. Probability of Backtest Overfitting (PBO / CSCV)
    # -------------------------------------------------------------------------
    print("\n[6.2] Computing Probability of Backtest Overfitting (CSCV S=6 slices, C(6,3)=20 combinations)...")
    pbo_val, pbo_summary, pbo_combs = compute_cscv_pbo(daily_returns, rf_daily, s_splits=6)
    pbo_combs.to_csv(OUT_PBO_CSV, index=False)
    print(f"      Saved CSCV combinations -> {OUT_PBO_CSV} ({len(pbo_combs)} rows)")
    print(f"      Calculated PBO across all strategies = {pbo_val:.3f} ({int(pbo_summary['overfit_combinations'].iloc[0])}/20 combinations)")
    print(f"      Median OOS Relative Rank: {pbo_summary['median_oos_rel_rank'].iloc[0]:.2f}")
    assert pbo_val < 0.5, f"PBO must be < 0.5 (safe zone), got {pbo_val}"
    
    print("\n[6.2] Step 6.2 complete.")


if __name__ == "__main__":
    main()
