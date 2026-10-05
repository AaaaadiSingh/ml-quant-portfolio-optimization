from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ModelFamily = Literal["ridge_linreg", "rf", "xgb"]

RANDOM_SEED = 7


def train_ridge_linreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    alpha: float = 1.0,
) -> Pipeline:
    if len(X_train) != len(y_train):
        raise ValueError(f"X/y length mismatch: {len(X_train)} vs {len(y_train)}")
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=RANDOM_SEED, fit_intercept=True)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: int | None = 8,
    min_samples_leaf: int = 20,
    random_state: int = RANDOM_SEED,
) -> RandomForestRegressor:
    if len(X_train) != len(y_train):
        raise ValueError(f"X/y length mismatch: {len(X_train)} vs {len(y_train)}")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = 500,
    max_depth: int = 4,
    learning_rate: float = 0.03,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    random_state: int = RANDOM_SEED,
    early_stopping_eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
) -> XGBRegressor:
    if len(X_train) != len(y_train):
        raise ValueError(f"X/y length mismatch: {len(X_train)} vs {len(y_train)}")
    xgb = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        random_state=random_state,
        tree_method="hist",
        objective="reg:squarederror",
        early_stopping_rounds=None if early_stopping_eval_set is None else 40,
    )
    if early_stopping_eval_set is not None:
        X_tv, y_tv = early_stopping_eval_set
        if len(X_tv) == 0:
            xgb.fit(X_train, y_train, verbose=False)
        else:
            xgb.fit(
                X_train,
                y_train,
                eval_set=[(X_tv, y_tv)],
                verbose=False,
            )
    else:
        xgb.fit(X_train, y_train, verbose=False)
    return xgb


def predict_1d(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    preds = model.predict(X_test)
    out = np.asarray(preds, dtype=float).flatten()
    if out.shape[0] != len(X_test):
        raise ValueError(
            f"predictor shape mismatch: expected {len(X_test)} got {out.shape[0]}"
        )
    return out


def _oos_predictions(
    model_factory: Any,
    X: pd.DataFrame,
    y: pd.Series,
    date_index: pd.DatetimeIndex,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch {len(X)} vs {len(y)}")
    if len(date_index) != len(X):
        date_index_unique = pd.DatetimeIndex(
            X.index.get_level_values("date").unique().sort_values()
        )
        if len(date_index_unique) != len(X):
            groups = X.index.get_level_values("date")
        else:
            groups = date_index_unique
    else:
        groups = date_index

    unique_dates_sorted = pd.DatetimeIndex(
        np.sort(np.unique(X.index.get_level_values("date").values))
    )
    tss = TimeSeriesSplit(n_splits=n_splits, gap=0)
    folds_for_split: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_date_idx, va_date_idx in tss.split(unique_dates_sorted):
        tr_dates = unique_dates_sorted[tr_date_idx]
        va_dates = unique_dates_sorted[va_date_idx]
        row_tr = np.asarray(X.index.get_level_values("date").isin(tr_dates), dtype=bool)
        row_va = np.asarray(X.index.get_level_values("date").isin(va_dates), dtype=bool)
        folds_for_split.append(
            (np.where(row_tr)[0], np.where(row_va)[0])
        )

    yhat_full = np.full(len(X), np.nan, dtype=float)
    fold_records: list[dict[str, Any]] = []
    for fold_id, (tr_idx, va_idx) in enumerate(folds_for_split):
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        factory_name = getattr(model_factory, "__name__", "")
        if "xgb" in factory_name:
            n_tr = len(X_tr)
            cut = max(1, int(n_tr * 0.85))
            order = np.argsort(X_tr.index.get_level_values("date").values)
            X_tr_sorted = X_tr.iloc[order]
            y_tr_sorted = y_tr.iloc[order]
            X_fit = X_tr_sorted.iloc[:cut]
            y_fit = y_tr_sorted.iloc[:cut]
            X_tv = X_tr_sorted.iloc[cut:]
            y_tv = y_tr_sorted.iloc[cut:]
            model = model_factory(X_fit, y_fit, early_stopping_eval_set=(X_tv, y_tv))
        else:
            model = model_factory(X_tr, y_tr)

        preds = predict_1d(model, X_va)
        yhat_full[va_idx] = preds

        resid_va = y_va.values - preds
        rmse = float(np.sqrt(np.mean(resid_va**2)))
        ss_res = float(np.sum(resid_va**2))
        mean_y = float(np.mean(y_va.values))
        ss_tot = float(np.sum((y_va.values - mean_y) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        dir_acc = float(
            np.mean(
                np.sign(y_va.values) == np.sign(preds)
            )
            * 100.0
        )
        med_abs_bps = float(np.median(np.abs(resid_va)) * 10000.0)
        fold_records.append(
            {
                "fold_id": fold_id,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "rmse_21d": rmse,
                "r2_21d": r2,
                "directional_accuracy_pct": dir_acc,
                "median_abs_residual_21d_logret_bps": med_abs_bps,
            }
        )

    mask = np.isfinite(yhat_full)
    yhat_clean = yhat_full[mask]
    y_clean = y.values[mask]
    residuals = y_clean - yhat_clean
    return yhat_clean, residuals, fold_records


def extract_oos_residuals(
    family: ModelFamily,
    X: pd.DataFrame,
    y: pd.Series,
    date_index: pd.DatetimeIndex | None = None,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if family == "ridge_linreg":
        factory = train_ridge_linreg
    elif family == "rf":
        factory = train_rf
    elif family == "xgb":
        factory = train_xgb
    else:
        raise ValueError(f"unknown model family: {family}")

    if date_index is None:
        date_index = pd.DatetimeIndex(X.index.get_level_values("date").unique())
    yhat, resid, fold_records = _oos_predictions(factory, X, y, date_index, n_splits=n_splits)

    try:
        jb_stat, jb_pvalue = jarque_bera(resid)
        jb_stat = float(jb_stat)
        jb_pvalue = float(jb_pvalue)
    except Exception:
        jb_stat = float("nan")
        jb_pvalue = float("nan")
    gaussian_rejected_1pct = bool((not np.isnan(jb_pvalue)) and jb_pvalue < 0.01)

    rmse_all = float(np.sqrt(np.mean(resid**2)))
    mean_y = float(np.mean(yhat + resid))
    ss_res_all = float(np.sum(resid**2))
    ss_tot_all = float(np.sum(((yhat + resid) - mean_y) ** 2))
    r2_all = float(1.0 - ss_res_all / ss_tot_all) if ss_tot_all > 0 else float("nan")
    dir_acc_all = float(
        np.mean(np.sign(yhat + resid) == np.sign(yhat)) * 100.0
    )
    summary_row = {
        "model_family": family,
        "oos_rmse": rmse_all,
        "oos_r2": r2_all,
        "oos_directional_acc_pct": dir_acc_all,
        "jb_stat": jb_stat,
        "jb_p_value": jb_pvalue,
        "gaussian_null_rejected_at_1pct": gaussian_rejected_1pct,
    }
    fold_records_aug = [
        {**{"model_family": family}, **fr} for fr in fold_records
    ]
    return (
        yhat,
        resid,
        [summary_row] + fold_records_aug,
    )


FEATURE_FAMILY_MAP: dict[str, str] = {
    "ret_1d": "returns",
    "ret_5d": "returns",
    "ret_21d": "returns",
    "ret_63d": "returns",
    "ret_126d": "returns",
    "vol_ann_21d": "volatility",
    "vol_ann_63d": "volatility",
    "vol_ann_126d": "volatility",
    "sma_cross_20d": "price_level_sma",
    "sma_cross_50d": "price_level_sma",
    "sma_fastslow_20_50": "price_level_sma",
    "mom_63d": "momentum",
    "mom_126d": "momentum",
    "mom_252d": "momentum",
    "mom_252_21_skip": "momentum",
    "vol_rel_mean63d": "volume",
    "illiquid_frac_21d": "volume",
    "var95_21d": "risk_drawdown",
    "mdd_21d": "risk_drawdown",
}


def feature_family_of(col: str) -> str:
    if col in FEATURE_FAMILY_MAP:
        return FEATURE_FAMILY_MAP[col]
    if col.startswith("cal_month_") or col.startswith("cal_qtr_") or col.startswith("cal_half_"):
        return "calendar_seasonal"
    if col.startswith("sector_"):
        return "sector_dummies"
    if col.startswith("cs_rank_"):
        return "cross_sectional_rank"
    return "other"
