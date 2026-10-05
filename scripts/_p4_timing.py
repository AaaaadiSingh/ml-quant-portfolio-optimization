from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stage1a_baselines import DEV_END, STUDY_START
from src.data_loader import load_prices
from src.features import FORWARD_TARGET_HORIZON_DAYS, align_X_y, make_features, make_targets
from scripts.phase4_build_forecasts import (
    FAMILIES,
    QUARTERLY_REBAL_HORIZON_D,
    _build_latest_X_per_ticker,
    _rebalance_dates_from_phase3,
    _train_factory,
    predict_1d,
)


def main() -> int:
    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    prices = load_prices(tickers, start_date=STUDY_START, end_date=DEV_END, final_holdout=False)
    close, high, low, volume = prices["Close"], prices["High"], prices["Low"], prices["Volume"]
    X_all = make_features(close, high, low, volume)
    y_all = make_targets(close)
    X, y, common_dates = align_X_y(X_all, y_all)
    print(f"X={X.shape}", flush=True)
    rds = _rebalance_dates_from_phase3(37)
    HOR = int(FORWARD_TARGET_HORIZON_DAYS)
    train_dates = X.index.get_level_values("date")
    # Find first usable RD and a LATER one (half-way, near end) to estimate wall time
    usable: list[tuple[int, pd.Timestamp, int]] = []
    for k, rd in enumerate(rds):
        mask = np.asarray(train_dates < (rd - pd.Timedelta(days=HOR)), dtype=bool)
        n = int(np.where(mask)[0].shape[0])
        if n >= 500:
            usable.append((k, rd, n))
    picks = [0, len(usable) // 2, len(usable) - 1]
    total_est_sec = 0.0
    for pck in picks:
        k, rd, n = usable[pck]
        mask = np.asarray(train_dates < (rd - pd.Timedelta(days=HOR)), dtype=bool)
        row_idx = np.where(mask)[0]
        Xtr, ytr = X.iloc[row_idx], y.iloc[row_idx]
        Xl = _build_latest_X_per_ticker(X, tickers, rd)
        for fam in FAMILIES:
            t0 = time.time()
            m = _train_factory(fam, Xtr, ytr)
            p = predict_1d(m, Xl)
            dt = time.time() - t0
            print(f"  usable[{pck}] RD#{k} ({rd.date()}) n_tr={len(Xtr)} {fam:15s} {dt:6.1f}s preds range=[{p.min():.3f},{p.max():.3f}]", flush=True)
            total_est_sec += dt
    print(f"\n3 sampled (1/2 mid/final) SUM={total_est_sec:.1f}s", flush=True)
    n_families = len(FAMILIES)
    # Scale: 3 samples / usable total
    n_usable = len(usable)
    est_total = total_est_sec * (n_usable * n_families) / (len(picks) * n_families)
    print(f"usable RDs = {n_usable}/37. est full train+forecast = {est_total/60:.1f} min = {est_total/3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
