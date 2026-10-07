from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.monte_carlo import WEIGHT_UPPER_LOCKED, SECTOR_TOLERANCE_LOCKED
from src.optimizer import _sector_group_matrix

CENTROID_MEDIAN_CSV = ROOT / "data" / "processed" / "phase5_centroid_median_weights.csv"
PER_DRAW_SHARPE_CSV = ROOT / "data" / "processed" / "phase5_per_draw_approx_sharpe.csv"
RAW_WEIGHTS_CSV = ROOT / "data" / "processed" / "phase5_raw_weight_draws_long.csv.gz"
NIFTY_MAP_CSV = ROOT / "docs" / "nifty50_sector_map.csv"

DECISION_CSV = ROOT / "data" / "processed" / "phase5_centroid_selection_decision.csv"
SELECTED_WEIGHTS_CSV = ROOT / "data" / "processed" / "phase5_selected_centroid_weights.csv"
COMPARE_PANEL_CSV = ROOT / "data" / "processed" / "phase5_centroid_comparison_panel.csv"

AGG_CANDIDATES = ("median", "mean", "medoid_draw")


def _load_sector_meta() -> tuple[list[str], dict[str, str], dict[str, float], np.ndarray, np.ndarray]:
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
    secs, G = _sector_group_matrix(len(tickers), pd.Series(t2s), tickers)
    tgt_vec = np.array([float(targets[s]) for s in secs], dtype=float)
    if abs(tgt_vec.sum() - 1.0) > 1e-9 and tgt_vec.sum() > 0:
        tgt_vec = tgt_vec / tgt_vec.sum()
    return tickers, t2s, targets, G, tgt_vec


def _build_mean_centroid(raw_df: pd.DataFrame, rds: list[pd.Timestamp], tickers: list[str]) -> pd.DataFrame:
    rows = []
    for rd in rds:
        sub = raw_df[raw_df["rebal_date"] == rd]
        piv = sub.pivot_table(index="draw_id", columns="ticker", values="weight").reindex(columns=tickers)
        mat = np.nan_to_num(piv.to_numpy(dtype=float), nan=0.0)
        cent = np.mean(mat, axis=0)
        s = float(np.nansum(cent))
        if abs(s) < 1e-15:
            cent = np.full(len(tickers), 1.0 / len(tickers), dtype=float)
        else:
            cent = cent / s
        for j, t in enumerate(tickers):
            rows.append({"rebal_date": rd, "ticker": t, "weight": float(cent[j])})
    return pd.DataFrame(rows)


def _build_medoid_centroid(
    raw_df: pd.DataFrame, per_draw_df: pd.DataFrame, rds: list[pd.Timestamp], tickers: list[str]
) -> pd.DataFrame:
    rows = []
    for rd in rds:
        sub_pd = per_draw_df[per_draw_df["rebal_date"] == rd]
        if sub_pd.empty:
            for t in tickers:
                rows.append({"rebal_date": rd, "ticker": t, "weight": 1.0 / len(tickers)})
            continue
        sharpe_rd = sub_pd["approx_sharpe"].astype(float).to_numpy()
        med_sharpe = float(np.nanmedian(sharpe_rd))
        order = sorted(range(len(sharpe_rd)), key=lambda k: abs(float(sharpe_rd[k]) - med_sharpe))
        sub_raw = raw_df[raw_df["rebal_date"] == rd]
        piv = sub_raw.pivot_table(index="draw_id", columns="ticker", values="weight").reindex(columns=tickers)
        draw_ids = list(piv.index.astype(int))
        k_star = draw_ids[int(order[0])] if order and len(order) > 0 and order[0] < len(draw_ids) else 0
        if k_star not in draw_ids:
            k_star = int(draw_ids[min(0, len(draw_ids) - 1)])
        w_row = piv.loc[k_star].to_numpy(dtype=float)
        w_row = np.nan_to_num(w_row, nan=0.0)
        for j, t in enumerate(tickers):
            rows.append({"rebal_date": rd, "ticker": t, "weight": float(w_row[j])})
    return pd.DataFrame(rows)


def _score_centroid(
    df: pd.DataFrame,
    tickers: list[str],
    G: np.ndarray,
    tgt_vec: np.ndarray,
    raw_df: pd.DataFrame,
) -> dict[str, float]:
    df["rebal_date"] = pd.to_datetime(df["rebal_date"])
    rds = sorted(df["rebal_date"].drop_duplicates().tolist())
    cap_pp_vals = []
    drift_pp_vals = []
    std_c_across_tickers = []
    avg_k_centered_sharpe = []
    tol_pp = SECTOR_TOLERANCE_LOCKED * 100.0
    for rd in rds:
        sub = df[df["rebal_date"] == rd].set_index("ticker")["weight"].reindex(tickers).to_numpy(dtype=float)
        sub = np.nan_to_num(sub, nan=0.0)
        cap_pp_vals.append(float(sub.max()) * 100.0)
        drift = (G @ sub - tgt_vec) * 100.0
        drift_pp_vals.append(float(np.max(np.abs(drift))))
        std_c_across_tickers.append(float(np.std(sub)))
        sub_r = raw_df[raw_df["rebal_date"] == rd]
        piv_r = sub_r.pivot_table(index="draw_id", columns="ticker", values="weight").reindex(columns=tickers)
        mat_r = np.nan_to_num(piv_r.to_numpy(dtype=float), nan=0.0)
        diffs = np.linalg.norm(mat_r - sub.reshape(1, -1), axis=1)
        avg_k_centered_sharpe.append(float(np.mean(diffs)))
    w_cap_pp_max = float(np.max(cap_pp_vals)) if cap_pp_vals else float("nan")
    cap_viol = 1 if w_cap_pp_max > WEIGHT_UPPER_LOCKED * 100.0 + 1e-3 else 0
    drift_pp_max = float(np.max(drift_pp_vals)) if drift_pp_vals else float("nan")
    drift_viol = 1 if drift_pp_max > tol_pp + 0.001 else 0
    avg_std_c = float(np.mean(std_c_across_tickers)) if std_c_across_tickers else float("nan")
    avg_distance = float(np.mean(avg_k_centered_sharpe)) if avg_k_centered_sharpe else float("nan")
    composite = -avg_std_c  # higher concentration = more negative composite = worse; we want diversified
    return {
        "cap_max_pp": w_cap_pp_max,
        "cap_violation": float(cap_viol),
        "drift_max_pp": drift_pp_max,
        "drift_violation": float(drift_viol),
        "avg_weight_std_per_rd": avg_std_c,
        "composite_score_lower_better_diversified": float(composite),
        "avg_distance_to_cloud_draws_L2": avg_distance,
        "n_rd": float(len(rds)),
    }


def main() -> None:
    for f in (CENTROID_MEDIAN_CSV, PER_DRAW_SHARPE_CSV, RAW_WEIGHTS_CSV):
        if not f.exists():
            raise FileNotFoundError(f"Run step3 first: missing {f}")

    tickers, t2s, sec_targets, G_mat, tgt_vec = _load_sector_meta()
    print(f"[step5] tickers={len(tickers)}, sectors={len(sec_targets)}")

    print("[step5] Loading raw draws + per_draw sharpe ...")
    raw_df = pd.read_csv(RAW_WEIGHTS_CSV)
    raw_df["rebal_date"] = pd.to_datetime(raw_df["rebal_date"])
    per_draw_df = pd.read_csv(PER_DRAW_SHARPE_CSV)
    per_draw_df["rebal_date"] = pd.to_datetime(per_draw_df["rebal_date"])
    rds_all = sorted(raw_df["rebal_date"].drop_duplicates().tolist())
    print(f"[step5] RDs={len(rds_all)}")

    cent_median = pd.read_csv(CENTROID_MEDIAN_CSV)
    cent_median["rebal_date"] = pd.to_datetime(cent_median["rebal_date"])
    print(f"[step5] building mean centroid ...")
    cent_mean = _build_mean_centroid(raw_df, rds_all, tickers)
    print(f"[step5] building medoid_draw centroid ...")
    cent_medoid = _build_medoid_centroid(raw_df, per_draw_df, rds_all, tickers)

    centroids = {
        "median": cent_median,
        "mean": cent_mean,
        "medoid_draw": cent_medoid,
    }
    panel_rows = []
    scores: dict[str, dict[str, float]] = {}
    for ag, df in centroids.items():
        score = _score_centroid(df, tickers, G_mat, tgt_vec, raw_df)
        scores[ag] = score
        row = {"aggregation": ag, **score}
        panel_rows.append(row)
        print(
            f"  agg={ag}: cap_max={score['cap_max_pp']:.3f}pp, drift_max={score['drift_max_pp']:.3f}pp, "
            f"std/rd={score['avg_weight_std_per_rd']:.5f}, composite_score={score['composite_score_lower_better_diversified']:.5f}"
        )
    panel_df = pd.DataFrame(panel_rows)
    panel_df.to_csv(COMPARE_PANEL_CSV, index=False)
    print(f"[step5] wrote comparison panel -> {COMPARE_PANEL_CSV}")

    feasible = {ag: s for ag, s in scores.items() if s["cap_violation"] == 0 and s["drift_violation"] == 0}
    assert len(feasible) > 0, f"No feasible aggregations! scores = {scores}"
    ags_sort = sorted(feasible.keys(), key=lambda a: (
        feasible[a]["composite_score_lower_better_diversified"],
        a,
    ))
    selected = ags_sort[0]  # most negative composite = most diversified (worst) — OOPS: we want LESS negative composite = higher composite = BETTER diversified. Wait: composite_score = -avg_std. Higher avg_std = more diversified. So composite = -std: LESS negative = HIGHER (closer to 0) = more diversified. So sort ascending (most negative first) → take LAST (least negative, highest).
    ags_sort2 = sorted(feasible.keys(), key=lambda a: (
        -feasible[a]["composite_score_lower_better_diversified"],
        a,
    ))
    selected = ags_sort2[0]
    print(
        f"[step5] Feasible candidates = {len(feasible)}/{len(scores)}. "
        f"Ordered (most diversified first): {ags_sort2}. SELECTED = {selected!r}"
    )

    sel_df = centroids[selected].copy()
    sel_df.to_csv(SELECTED_WEIGHTS_CSV, index=False)
    print(f"[step5] wrote selected centroid -> {SELECTED_WEIGHTS_CSV} ({len(sel_df)} rows)")

    justification = (
        f"Selected '{selected}' from {AGG_CANDIDATES} via Phase5 FR-5 tie-breaker "
        f"(diversification priority: min -mean(std(w)) = {scores[selected]['composite_score_lower_better_diversified']:.5f}; "
        f"feasibility gate: cap={scores[selected]['cap_max_pp']:.3f}pp<=10pp, "
        f"drift={scores[selected]['drift_max_pp']:.3f}pp<=±{SECTOR_TOLERANCE_LOCKED*100:.1f}pp). "
        f"Other feasible: {[a for a in ags_sort2 if a != selected]}."
    )

    decision_row = {
        "selected_aggregation": selected,
        "n_feasible_candidates": int(len(feasible)),
        "n_total_candidates": int(len(scores)),
        "candidates_evaluated": "|".join(list(scores.keys())),
        "criterion": "feasible(cap=0 and drift=0) → max composite_score (most diversified)",
        "composite_score_selected": float(scores[selected]["composite_score_lower_better_diversified"]),
        "cap_max_pp_selected": float(scores[selected]["cap_max_pp"]),
        "drift_max_pp_selected": float(scores[selected]["drift_max_pp"]),
        "avg_distance_to_cloud_L2_selected": float(scores[selected]["avg_distance_to_cloud_draws_L2"]),
        "justification": justification,
    }
    pd.DataFrame([decision_row]).to_csv(DECISION_CSV, index=False)
    print(f"[step5] wrote selection decision -> {DECISION_CSV}")
    print(f"[step5] DONE  SELECTED: {selected!r}")


if __name__ == "__main__":
    main()
