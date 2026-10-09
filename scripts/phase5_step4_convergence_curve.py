from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.optimizer import _sector_group_matrix, _post_verify_weights
from src.monte_carlo import SECTOR_TOLERANCE_LOCKED, WEIGHT_UPPER_LOCKED

RAW_WEIGHTS_CSV = ROOT / "data" / "processed" / "phase5_raw_weight_draws_long.csv.gz"
CENTROID_CSV = ROOT / "data" / "processed" / "phase5_centroid_median_weights.csv"
NIFTY_MAP_CSV = ROOT / "docs" / "nifty50_sector_map.csv"

CONVERGENCE_CSV = ROOT / "data" / "processed" / "phase5_convergence_curve.csv"
STABILITY_CSV = ROOT / "data" / "processed" / "phase5_centroid_stability_ab.csv"

K_PROBE_LIST = [10, 25, 50, 100, 200, 300, 400, 500]


def _load_sector_meta() -> tuple[list[str], dict[str, str], dict[str, float]]:
    from collections import Counter

    df = pd.read_csv(NIFTY_MAP_CSV)
    universe = pd.read_csv(ROOT / "data" / "raw" / "universe_frozen.csv")
    tickers = sorted(universe["ticker"].tolist())
    full_map = dict(zip(df["ticker"].astype(str).tolist(), df["sector_provisional"].astype(str).tolist()))
    t2s = {t: full_map[t] for t in tickers if t in full_map}
    cnt = Counter(t2s.values())
    total = sum(cnt.values())
    targets = {s: float(c) / total for s, c in cnt.items()}
    ssum = float(sum(targets.values()))
    if ssum > 0:
        targets = {s: v / ssum for s, v in targets.items()}
    return tickers, t2s, targets


def main() -> None:
    if not RAW_WEIGHTS_CSV.exists():
        raise FileNotFoundError(f"Step3 raw weights missing: {RAW_WEIGHTS_CSV}. Run step3 first.")
    if not CENTROID_CSV.exists():
        raise FileNotFoundError(f"Step3 centroid missing: {CENTROID_CSV}. Run step3 first.")

    tickers, ticker_to_sector, sector_targets = _load_sector_meta()
    print(f"[step4] tickers={len(tickers)}, sectors={len(sector_targets)}")

    print(f"[step4] Loading raw draws (may be large) ...")
    raw_df = pd.read_csv(RAW_WEIGHTS_CSV)
    print(f"[step4] raw_df loaded: rows={len(raw_df)} cols={list(raw_df.columns)}")
    assert {"rebal_date", "draw_id", "ticker", "weight"} <= set(raw_df.columns)

    raw_df["rebal_date"] = pd.to_datetime(raw_df["rebal_date"])
    rds_all = sorted(raw_df["rebal_date"].drop_duplicates().tolist())
    n_rd = len(rds_all)
    print(f"[step4] RDs: {n_rd}, draw_id range: {raw_df['draw_id'].min()}-{raw_df['draw_id'].max()}")

    secs_sort, G_mat = _sector_group_matrix(len(tickers), pd.Series(ticker_to_sector), tickers)
    tgt_vec = np.array([float(sector_targets[s]) for s in secs_sort], dtype=float)
    if abs(tgt_vec.sum() - 1.0) > 1e-9 and tgt_vec.sum() > 0:
        tgt_vec = tgt_vec / tgt_vec.sum()
    tol = SECTOR_TOLERANCE_LOCKED * 100.0 + 0.001

    cent_ref_full = pd.read_csv(CENTROID_CSV)
    cent_ref_full["rebal_date"] = pd.to_datetime(cent_ref_full["rebal_date"])

    print(f"[step4] Running K probe sweep = {K_PROBE_LIST} ...")
    convergence_rows = []
    seed_rng = np.random.default_rng(7)
    for K in K_PROBE_LIST:
        assert K <= int(raw_df["draw_id"].max()) + 1, f"K={K} exceeds available draws"
        subset_ids = seed_rng.choice(
            np.arange(int(raw_df["draw_id"].max()) + 1), size=K, replace=False
        )
        sub = raw_df[raw_df["draw_id"].isin(set(int(s) for s in subset_ids))]
        piv = sub.pivot_table(index=["rebal_date", "draw_id"], columns="ticker", values="weight").reindex(columns=tickers)
        median_vecs = {}
        violations_k_cap = 0
        violations_k_drift = 0
        for rd in rds_all:
            if K == 500:
                sub_ref_k500 = cent_ref_full[cent_ref_full["rebal_date"] == rd].set_index("ticker")["weight"].reindex(tickers).to_numpy(dtype=float)
                cent_rd = np.nan_to_num(sub_ref_k500, nan=0.0)
            else:
                if rd not in piv.index.get_level_values("rebal_date"):
                    continue
                block = piv.xs(rd, level="rebal_date").to_numpy(dtype=float)
                if block.shape[0] < 1:
                    continue
                block_clean = np.nan_to_num(block, nan=0.0)
                cent_rd = np.median(block_clean, axis=0)
                s = float(np.nansum(cent_rd))
                if abs(s) < 1e-15:
                    cent_rd = np.full_like(cent_rd, 1.0 / len(tickers))
                else:
                    cent_rd = cent_rd / s
                post_sec = {
                    "tickers_order": tickers,
                    "sector_targets": sector_targets,
                    "ticker_to_sector": ticker_to_sector,
                    "sector_tolerance": SECTOR_TOLERANCE_LOCKED,
                }
                cent_rd = _post_verify_weights(
                    cent_rd,
                    w_upper=WEIGHT_UPPER_LOCKED,
                    label=f"RD={rd}_K={K}",
                    sector_info=post_sec,
                )
            if float(cent_rd.max()) > WEIGHT_UPPER_LOCKED + 1e-3:
                violations_k_cap += 1
            drift_pp = (G_mat @ cent_rd - tgt_vec) * 100.0
            if abs(drift_pp).max() > tol:
                violations_k_drift += 1
            median_vecs[rd] = cent_rd
        l2_diffs = []
        for rd in rds_all:
            if rd not in median_vecs:
                continue
            sub_ref = cent_ref_full[cent_ref_full["rebal_date"] == rd].set_index("ticker")["weight"].reindex(tickers).to_numpy(dtype=float)
            sub_ref = np.nan_to_num(sub_ref, nan=0.0)
            d = float(np.linalg.norm(median_vecs[rd] - sub_ref))
            l2_diffs.append(d)
        avg_l2 = float(np.mean(l2_diffs)) if l2_diffs else float("nan")
        max_l2 = float(np.max(l2_diffs)) if l2_diffs else float("nan")
        total_rd_probed = len(median_vecs)
        convergence_rows.append(
            {
                "K_probe": int(K),
                "n_rd_probed": total_rd_probed,
                "cap_violations": int(violations_k_cap),
                "drift_violations": int(violations_k_drift),
                "l2_mean_vs_full500_centroid": avg_l2,
                "l2_max_vs_full500_centroid": max_l2,
            }
        )
        print(
            f"  K={K:>4d}: L2 mean/vs full = {avg_l2:.5f}/{max_l2:.5f}, "
            f"cap viol RD = {violations_k_cap}, drift viol RD = {violations_k_drift}"
        )

    conv_df = pd.DataFrame(convergence_rows)
    conv_df.to_csv(CONVERGENCE_CSV, index=False)
    print(f"[step4] wrote convergence curve -> {CONVERGENCE_CSV} ({len(conv_df)} rows, K_sweep={K_PROBE_LIST})")

    assert 500 in K_PROBE_LIST, "K=500 required in sweep"
    k500 = conv_df[conv_df["K_probe"] == 500].iloc[0]
    assert int(k500["cap_violations"]) == 0, "K=500 cap violations!"
    assert int(k500["drift_violations"]) == 0, "K=500 drift violations!"
    k500_l2 = float(k500["l2_mean_vs_full500_centroid"])
    assert k500_l2 < 0.01 or abs(k500_l2) < 1e-12, f"K=500 self-L2 = {k500_l2} must be ≈ 0!"
    print("[step4] K=500 integrity: cap-0, drift-0, self-L2≈0  [PASS]")

    if len(conv_df) >= 3:
        k100 = conv_df[conv_df["K_probe"] >= 100].sort_values("K_probe").iloc[0]
        first_row = conv_df.sort_values("K_probe").iloc[0]
        l2_drop = abs(float(first_row["l2_mean_vs_full500_centroid"]) - float(k100["l2_mean_vs_full500_centroid"]))
        print(f"[step4] stability: first_K({first_row['K_probe']}) → K≥100, ΔL2 = {l2_drop:.5f}")

    print(f"[step4] Stability A/B: equal-weight baseline vs centroid (1/N vs {WEIGHT_UPPER_LOCKED*100:.0f}% cap)")
    eq_w = np.full(len(tickers), 1.0 / len(tickers), dtype=float)
    eq_max_pp = float(eq_w.max()) * 100.0
    assert eq_max_pp <= 10.0 + 1e-6, f"equal-weight violates cap, max = {eq_max_pp}pp"
    eq_drift = (G_mat @ eq_w - tgt_vec) * 100.0
    eq_drift_max_pp = float(np.max(np.abs(eq_drift)))
    assert eq_drift_max_pp <= SECTOR_TOLERANCE_LOCKED * 100.0 + 0.001, (
        f"equal-weight violates drift, max = {eq_drift_max_pp}pp"
    )
    cent_all = pd.read_csv(CENTROID_CSV)
    cent_all["rebal_date"] = pd.to_datetime(cent_all["rebal_date"])
    cent_piv = cent_all.pivot_table(index="rebal_date", columns="ticker", values="weight").reindex(columns=tickers)
    cent_mat = np.nan_to_num(cent_piv.to_numpy(dtype=float), nan=0.0)
    eq_stack = np.tile(eq_w.reshape(1, -1), (cent_mat.shape[0], 1))
    l2_rd = np.linalg.norm(cent_mat - eq_stack, axis=1)
    stab_rows = [
        {
            "baseline": "equal_weight_1_over_N",
            "metric": "centroid_vs_eq_L2",
            "mean": float(np.mean(l2_rd)),
            "median": float(np.median(l2_rd)),
            "max": float(np.max(l2_rd)),
            "q95": float(np.quantile(l2_rd, 0.95)),
            "eq_drift_max_pp": eq_drift_max_pp,
            "eq_max_weight_pp": eq_max_pp,
            "n_rd": int(len(l2_rd)),
        }
    ]
    pd.DataFrame(stab_rows).to_csv(STABILITY_CSV, index=False)
    print(f"[step4] wrote stability A/B -> {STABILITY_CSV}")
    print(
        f"[step4] Stability A/B PASS: 1/N baseline drift ≤ {eq_drift_max_pp:.4f}pp (<=±{SECTOR_TOLERANCE_LOCKED*100:.1f}pp), "
        f"max weight = {eq_max_pp:.4f}pp (<=10pp)"
    )
    print("[step4] DONE")


if __name__ == "__main__":
    main()
