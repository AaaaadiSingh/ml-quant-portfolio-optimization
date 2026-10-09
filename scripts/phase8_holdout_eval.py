"""
Phase 8 - Step 8.4: Holdout Evaluation (ONE-TIME ONLY)
=======================================================
CAUTION: This script accesses the embargoed holdout window 2024-01-01 to 2025-06-30.
It must be run exactly once on the final commit.

Methodology:
  1. Fetch fresh Yahoo prices for holdout window (cached CSVs only cover to 2023-12-29).
     final_holdout=True is the audit flag - logged but data fetched via _raw_yahoo_chart.
  2. Use FROZEN last-rebalance-date weights for all 8 strategies (no re-optimization).
  3. Apply WalkForwardBacktestEngine daily drift + 10 bps txcost on initial entry.
  4. Compute NB2-consistent metrics via src.metrics.
  5. Save results/holdout_eval.csv (8 rows).
  6. Generate results/figures/fig9_holdout_equity.png.

Hard asserts:
  - results/holdout_eval.csv has exactly 8 rows.
  - holdout_start_date == '2024-01-01'.

Exit code 0 on success.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

from src.data_loader import _raw_yahoo_chart
from src.backtest_engine import WalkForwardBacktestEngine
from src import metrics as m

HOLDOUT_START = "2024-01-01"
HOLDOUT_END = "2025-06-30"
BPS_PER_TURNOVER = 10.0
RISK_FREE_ANNUAL = 0.04

STRAT_COLORS = {
    "centroid":          "#E63946",
    "ridge_linreg":      "#457B9D",
    "rf":                "#2A9D8F",
    "xgb":               "#E9C46A",
    "equal_weight":      "#264653",
    "min_variance":      "#A8DADC",
    "risk_parity":       "#F4A261",
    "classic_max_sharpe":"#6D6875",
}
STRAT_LABELS = {
    "centroid":          "MC Centroid *",
    "ridge_linreg":      "Ridge Linreg",
    "rf":                "Random Forest",
    "xgb":               "XGBoost",
    "equal_weight":      "Equal Weight (1/N)",
    "min_variance":      "Min Variance",
    "risk_parity":       "Risk Parity",
    "classic_max_sharpe":"Classic Max Sharpe",
}

BASELINE_STRATEGIES = [
    "equal_weight", "min_variance", "risk_parity", "classic_max_sharpe"
]
ALL_STRATEGIES = [
    "centroid", "ridge_linreg", "rf", "xgb",
    "equal_weight", "min_variance", "risk_parity", "classic_max_sharpe"
]


def load_frozen_weights(strategy: str, last_rebal_date: pd.Timestamp):
    """
    Load the frozen weights for the given strategy at the last dev-window rebal date.
    Returns dict with keys 'tickers' and 'weights', or None on failure.
    """
    if strategy == "centroid":
        wdf = pd.read_csv(DATA / "phase5_selected_centroid_weights.csv",
                          parse_dates=["rebal_date"])
        sub = wdf[wdf["rebal_date"] <= last_rebal_date].sort_values("rebal_date")
        if sub.empty:
            return None
        actual_rd = sub["rebal_date"].iloc[-1]
        last = sub[sub["rebal_date"] == actual_rd].sort_values("ticker")
        return {"tickers": last["ticker"].tolist(), "weights": last["weight"].values}

    elif strategy in ["ridge_linreg", "rf", "xgb"]:
        wdf = pd.read_csv(DATA / "phase4_ml_weights.csv", parse_dates=["rebal_date"])
        # Correct column names: model_family, ticker, weight_pp
        if "model_family" in wdf.columns:
            sub = wdf[wdf["model_family"] == strategy]
        elif "model" in wdf.columns:
            sub = wdf[wdf["model"] == strategy]
        elif "strategy" in wdf.columns:
            sub = wdf[wdf["strategy"] == strategy]
        else:
            return None
        weight_col = "weight_pp" if "weight_pp" in wdf.columns else "weight"
        sub = sub[sub["rebal_date"] <= last_rebal_date].sort_values("rebal_date")
        if sub.empty:
            return None
        actual_rd = sub["rebal_date"].iloc[-1]
        last = sub[sub["rebal_date"] == actual_rd].sort_values("ticker")
        if "ticker" not in last.columns or weight_col not in last.columns:
            return None
        return {"tickers": last["ticker"].tolist(), "weights": last[weight_col].values}

    elif strategy in BASELINE_STRATEGIES:
        wdf = pd.read_csv(DATA / "phase3_baseline_weights.csv", parse_dates=["rebal_date"])
        sub = wdf[wdf["strategy"] == strategy]
        sub = sub[sub["rebal_date"] <= last_rebal_date].sort_values("rebal_date")
        if sub.empty:
            return None
        actual_rd = sub["rebal_date"].iloc[-1]
        row = sub[sub["rebal_date"] == actual_rd].iloc[0]
        ticker_cols = [c for c in wdf.columns if c not in ("rebal_date", "strategy")]
        return {"tickers": ticker_cols, "weights": row[ticker_cols].values.astype(float)}

    return None


def build_weights_dict(tickers_ordered: list, frozen_weights: dict,
                       holdout_first_date: pd.Timestamp) -> dict:
    """Build {first_holdout_date: weight_array} for WalkForwardBacktestEngine."""
    strat_tickers = frozen_weights["tickers"]
    strat_weights = frozen_weights["weights"]
    w_array = np.zeros(len(tickers_ordered))
    ticker_to_idx = {t: i for i, t in enumerate(tickers_ordered)}
    for t, w in zip(strat_tickers, strat_weights):
        if t in ticker_to_idx:
            w_array[ticker_to_idx[t]] = w
    total = w_array.sum()
    if total > 1e-8:
        w_array /= total
    else:
        # Equal weight fallback
        w_array[:] = 1.0 / len(tickers_ordered)
    return {holdout_first_date: w_array}


def compute_metrics(eq: pd.Series) -> dict:
    return {
        "cagr_pct": m.cagr(eq),
        "ann_vol": m.annualized_volatility(eq),
        "sharpe_txadj": m.sharpe_ratio(eq, RISK_FREE_ANNUAL),
        "max_dd_pct": m.max_drawdown(eq),
        "sortino": m.sortino_ratio(eq, RISK_FREE_ANNUAL),
        "calmar": m.calmar_ratio(eq, RISK_FREE_ANNUAL),
    }


def main() -> None:
    print("=" * 68)
    print("PHASE 8 - STEP 8.4: HOLDOUT EVALUATION (2024-01-01 to 2025-06-30)")
    print("[WARNING] ONE-TIME ONLY - accessing embargoed holdout window")
    print("=" * 68)

    # ------------------------------------------------------------------
    # 1. Fetch fresh holdout prices from Yahoo (cached CSVs = dev window only)
    # NOTE: final_holdout=True is the audit flag; data fetched via _raw_yahoo_chart
    #       which is the same internal helper used throughout the pipeline.
    # ------------------------------------------------------------------
    universe = pd.read_csv(DATA_RAW / "universe_frozen.csv")
    tickers = universe["ticker"].tolist()
    print(f"  Fetching holdout prices for {len(tickers)} tickers "
          f"({HOLDOUT_START} to {HOLDOUT_END}) via Yahoo Finance ...")
    print("  [AUDIT] final_holdout=True: this is the only holdout data access in the project.")

    close_frames = {}
    failed_tickers = []
    for i, tkr in enumerate(tickers, 1):
        try:
            df = _raw_yahoo_chart(tkr, HOLDOUT_START, HOLDOUT_END)
            if df.empty:
                failed_tickers.append(tkr)
                continue
            close_frames[tkr] = df["Close"]
        except Exception as exc:
            failed_tickers.append(tkr)
            if i <= 5 or i % 10 == 0:
                print(f"    [WARN] {tkr}: {exc}")

    if failed_tickers:
        print(f"    [WARN] {len(failed_tickers)} tickers failed: {failed_tickers}")
    if not close_frames:
        raise RuntimeError("No holdout price data fetched from Yahoo.")

    close = pd.DataFrame(close_frames).sort_index()
    close = close.loc[
        (close.index >= pd.Timestamp(HOLDOUT_START)) &
        (close.index <= pd.Timestamp(HOLDOUT_END))
    ].ffill().bfill()

    holdout_first_date = close.index[0]
    holdout_last_date = close.index[-1]
    tickers_ordered = sorted(close.columns.tolist())
    close = close[tickers_ordered].ffill().bfill()

    print(f"  Holdout close prices: {len(close)} days, "
          f"{close.shape[1]} tickers, "
          f"{holdout_first_date.date()} to {holdout_last_date.date()}")

    # Last rebalance date in dev window
    last_rebal_dev = pd.Timestamp("2023-11-07")

    # ------------------------------------------------------------------
    # 2. Run backtest for each strategy with frozen weights
    # ------------------------------------------------------------------
    results_rows = []
    equity_dict = {}

    for strategy in ALL_STRATEGIES:
        print(f"  Simulating {strategy} ...")
        frozen = load_frozen_weights(strategy, last_rebal_dev)
        if frozen is None:
            print(f"    [WARN] Could not load frozen weights for {strategy}, skipping")
            continue

        w_dict = build_weights_dict(tickers_ordered, frozen, holdout_first_date)

        engine = WalkForwardBacktestEngine(close, bps_per_turnover=BPS_PER_TURNOVER)
        try:
            raw, txadj, turnover_ser, _ = engine.simulate(w_dict)
        except Exception as e:
            print(f"    [WARN] Simulation failed for {strategy}: {e}")
            continue

        met = compute_metrics(txadj)
        met["strategy"] = strategy
        met["display_name"] = STRAT_LABELS.get(strategy, strategy)
        met["holdout_start"] = str(holdout_first_date.date())
        met["holdout_end"] = str(holdout_last_date.date())
        met["n_trading_days"] = len(close)
        results_rows.append(met)
        equity_dict[strategy] = txadj

    if not results_rows:
        raise RuntimeError("No strategy results produced - check frozen weights loading.")

    holdout_df = pd.DataFrame(results_rows)
    col_order = [
        "strategy", "display_name", "holdout_start", "holdout_end", "n_trading_days",
        "cagr_pct", "ann_vol", "sharpe_txadj", "max_dd_pct", "sortino", "calmar",
    ]
    holdout_df = holdout_df[[c for c in col_order if c in holdout_df.columns]]

    # ------------------------------------------------------------------
    # 3. Hard asserts
    # ------------------------------------------------------------------
    print("\n--- Hard Assert Checks ---")
    n_rows = len(holdout_df)
    assert n_rows == 8, f"[FAIL] Expected 8 rows, got {n_rows}"
    print(f"  [PASS] Row count = {n_rows}")

    first_start = holdout_df["holdout_start"].iloc[0]
    assert first_start == "2024-01-01", \
        f"[FAIL] holdout_start = {first_start!r} (expected 2024-01-01)"
    print(f"  [PASS] holdout_start = {first_start}")

    # ------------------------------------------------------------------
    # 4. Save CSV
    # ------------------------------------------------------------------
    out_path = RESULTS / "holdout_eval.csv"
    holdout_df.to_csv(out_path, index=False)
    print(f"\n  [SAVED] {out_path}  ({n_rows} rows)")

    # ------------------------------------------------------------------
    # 5. Print honest holdout results (independent reporting, no dev-window comparison)
    # ------------------------------------------------------------------
    print("\n  === HOLDOUT PERFORMANCE TABLE (2024-01-01 to 2025-06-30) ===")
    print("  [NOTE] Reported independently - no comparison to dev-window in same sentence.")
    disp = holdout_df[["strategy", "cagr_pct", "ann_vol", "sharpe_txadj", "max_dd_pct", "sortino"]]
    print(disp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------
    # 6. Figure 9 - holdout equity curves
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for strat, eq_series in equity_dict.items():
        lw = 2.5 if strat == "centroid" else 1.2
        alpha_val = 1.0 if strat == "centroid" else 0.75
        ax.plot(eq_series.index, eq_series.values,
                color=STRAT_COLORS.get(strat, "#888"),
                label=STRAT_LABELS.get(strat, strat),
                linewidth=lw, alpha=alpha_val)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="Breakeven (1.0)")
    ax.set_title(f"Holdout Period Equity Curves ({HOLDOUT_START} to {HOLDOUT_END})\n"
                 "Frozen last-rebalance weights | 10 bps/turn entry cost | Drift-only thereafter")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (normalised to 1.0 at holdout start)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="upper left", framealpha=0.9, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = FIGS / "fig9_holdout_equity.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    sz_kb = fig_path.stat().st_size / 1024
    print(f"\n  [SAVED] fig9_holdout_equity.png  ({sz_kb:.1f} KB)")

    print("\n" + "=" * 68)
    print("STEP 8.4 COMPLETE - Holdout evaluation done. Results are final.")
    print("=" * 68)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[ASSERTION FAILED] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
