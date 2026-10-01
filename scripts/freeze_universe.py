#!/usr/bin/env python3
"""
scripts/freeze_universe.py
==========================

Pulls the 50 NIFTY 50 (current project start) constituent tickers via yfinance,
applies three pre-specified universe filters, and writes:

    1. data/raw/universe_frozen.csv       — only filter-surviving tickers
    2. docs/nifty50_sector_map.csv        — all 50 tickers + filter results
    3. stdout                              — summary stats & rejection log

Filters applied (from CONTEXT.md §D & docs/literature_matrix.md):
    F1  Data availability : First non-null Close on/before 2015-01-06
                            AND  Close NaN fraction  ≤ 5 % over 2015-01 → 2023-12
    F2  Liquidity         : Fraction of days with (volume == 0 OR
                            volume < 10,000 shares) ≤ 2 %
    F3  IPO date          : First trade date  <  2015-01-01  (from yfinance
                            info when available — otherwise flagged as
                            "MANUAL IPO CHECK NEEDED" and tentatively kept)

The dev window only is pulled: 2015-01-01 → 2023-12-31.
The final holdout (2024-01-01 → 2025-06-30) is NEVER touched here.
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DOCS = ROOT / "docs"

# ---------------------------------------------------------------------------
# 1. Hard-code the 50 NIFTY 50 tickers from docs/literature_matrix.md §D.
#    Also carry the provisional sector labels + free-float rank we wrote there
#    so the sector map CSV is already half-populated.
# ---------------------------------------------------------------------------
NIFTY50 = [
    # order, ticker, company_name, sector_provisional, free_float_rank
    (1,  "RELIANCE.NS",    "Reliance Industries Ltd.",                     "Energy - Oil & Gas",                                   1),
    (2,  "TCS.NS",         "Tata Consultancy Services",                    "Information Technology",                               2),
    (3,  "HDFCBANK.NS",    "HDFC Bank",                                    "Financials - Banks",                                   3),
    (4,  "INFY.NS",        "Infosys",                                      "Information Technology",                               4),
    (5,  "HINDUNILVR.NS",  "Hindustan Unilever",                           "Consumer Staples - Household & FMCG",                  5),
    (6,  "ICICIBANK.NS",   "ICICI Bank",                                   "Financials - Banks",                                   6),
    (7,  "SBIN.NS",        "State Bank of India",                          "Financials - Banks - PSU",                             7),
    (8,  "BHARTIARTL.NS",  "Bharti Airtel",                                "Communication Services - Telecom",                     8),
    (9,  "ITC.NS",         "ITC Limited",                                  "Consumer Staples - Tobacco & FMCG",                    9),
    (10, "KOTAKBANK.NS",   "Kotak Mahindra Bank",                          "Financials - Banks - Private",                        10),
    (11, "LT.NS",          "Larsen & Toubro",                              "Industrials - Construction & Engineering",            11),
    (12, "AXISBANK.NS",    "Axis Bank",                                    "Financials - Banks - Private",                        12),
    (13, "HCLTECH.NS",     "HCL Technologies",                             "Information Technology",                              13),
    (14, "ASIANPAINT.NS",  "Asian Paints",                                 "Materials - Paints & Coatings",                       14),
    (15, "MARUTI.NS",      "Maruti Suzuki India",                          "Consumer Discretionary - 4-Wheelers",                 15),
    (16, "BAJFINANCE.NS",  "Bajaj Finance",                                "Financials - NBFCs",                                  16),
    (17, "WIPRO.NS",       "Wipro",                                        "Information Technology",                              17),
    (18, "HDFCLIFE.NS",    "HDFC Life Insurance",                          "Financials - Life Insurance",                         18),
    (19, "SUNPHARMA.NS",   "Sun Pharmaceutical Industries",                "Healthcare - Pharmaceuticals",                        19),
    (20, "TITAN.NS",       "Titan Company",                                "Consumer Discretionary - Gems & Jewellery",           20),
    (21, "ULTRACEMCO.NS",  "UltraTech Cement",                             "Materials - Cement",                                  21),
    (22, "NESTLEIND.NS",   "Nestle India",                                 "Consumer Staples - Packaged Foods",                   22),
    (23, "NTPC.NS",        "NTPC Limited",                                 "Utilities - Power Generation - PSU",                  23),
    (24, "TATAMOTORS.NS",  "Tata Motors",                                  "Consumer Discretionary - 4-Wheelers",                 24),
    (25, "ONGC.NS",        "Oil & Natural Gas Corporation",                "Energy - Oil & Gas E&P - PSU",                        25),
    (26, "M&M.NS",         "Mahindra & Mahindra",                          "Consumer Discretionary - 4-Wheelers & Farm Equip.",   26),
    (27, "POWERGRID.NS",   "Power Grid Corporation of India",              "Utilities - Power Transmission - PSU",                27),
    (28, "JSWSTEEL.NS",    "JSW Steel",                                    "Materials - Steel",                                   28),
    (29, "HINDALCO.NS",    "Hindalco Industries",                          "Materials - Aluminium & Non-Ferrous Metals",          29),
    (30, "BAJAJFINSV.NS",  "Bajaj Finserv",                                "Financials - Diversified NBFC + Insurance",           30),
    (31, "DRREDDY.NS",     "Dr. Reddy's Laboratories",                     "Healthcare - Pharmaceuticals (Generics + API)",       31),
    (32, "ADANIENT.NS",    "Adani Enterprises",                            "Conglomerate - Resources + Infra + Trading",          32),
    (33, "TATASTEEL.NS",   "Tata Steel",                                   "Materials - Steel",                                   33),
    (34, "COALINDIA.NS",   "Coal India Limited",                           "Energy - Coal Mining - PSU",                          34),
    (35, "ADANIPORTS.NS",  "Adani Ports and Special Economic Zone",        "Industrials - Ports & Logistics",                     35),
    (36, "SBILIFE.NS",     "SBI Life Insurance",                           "Financials - Life Insurance",                         36),
    (37, "BPCL.NS",        "Bharat Petroleum Corporation Ltd.",            "Energy - Oil Refining & Marketing - PSU",             37),
    (38, "BRITANNIA.NS",   "Britannia Industries",                         "Consumer Staples - Biscuits & Dairy",                 38),
    (39, "EICHERMOT.NS",   "Eicher Motors",                                "Consumer Discretionary - 2/3-Wheelers (Royal Enfield)",39),
    (40, "CIPLA.NS",       "Cipla Limited",                                "Healthcare - Pharmaceuticals (Generics)",             40),
    (41, "DIVISLAB.NS",    "Divi's Laboratories",                          "Healthcare - Pharmaceuticals (CRAMS + API)",          41),
    (42, "GRASIM.NS",      "Grasim Industries",                            "Materials - Diversified (Viscose Staple + Cement)",   42),
    (43, "HDFCAMC.NS",     "HDFC Asset Management Company",                "Financials - Asset Management",                       43),
    (44, "APOLLOHOSP.NS",  "Apollo Hospitals Enterprise",                  "Healthcare - Hospital Services",                      44),
    (45, "INDUSINDBK.NS",  "IndusInd Bank",                                "Financials - Banks - Private",                        45),
    (46, "BAJAJ-AUTO.NS",  "Bajaj Auto",                                   "Consumer Discretionary - 2/3-Wheelers",               46),
    (47, "HEROMOTOCO.NS",  "Hero MotoCorp",                                "Consumer Discretionary - 2/3-Wheelers",               47),
    (48, "TATACONSUM.NS",  "Tata Consumer Products",                       "Consumer Staples - Beverages & Packaged Foods",       48),
    (49, "TECHM.NS",       "Tech Mahindra",                                "Information Technology",                              49),
    (50, "UPL.NS",         "UPL Limited (United Phosphorus)",              "Materials - Agrochemicals",                           50),
]

# DEV window only — final holdout (2024+) is NEVER read
DATA_START = dt.date(2015, 1, 1)
DATA_END   = dt.date(2023, 12, 31)
F1_FIRST_CLOSE_DEADLINE = pd.Timestamp("2015-01-06")
F1_MAX_NAN_FRAC         = 0.05
F2_MAX_ILLIQUID_FRAC    = 0.02
F3_MIN_TRADE_DATE       = pd.Timestamp("2015-01-01")

def _first_valid_close(close: pd.Series) -> pd.Timestamp | None:
    idx = close.first_valid_index()
    return idx if idx is not None else None


def _ipo_first_trade_date(ticker_str: str) -> tuple[pd.Timestamp | None, bool]:
    """
    Return (first_trade_date, manual_check_needed).
    yfinance for .NS often doesn't ship an explicit "ipo date" field, so we
    fall back to the earliest trading date recorded in the Ticker history
    meta, then info["firstTradeDateEpochUtc"], then None (needs manual audit).
    """
    try:
        tk = yf.Ticker(ticker_str)
        info = tk.info or {}
        for key in ("firstTradeDateEpochUtc", "firstTradeDate"):
            val = info.get(key)
            if isinstance(val, (int, float)) and val > 0:
                first = pd.Timestamp(val, unit="s", tz="UTC").tz_convert(None)
                return first, False
    except Exception:
        pass
    return None, True


def _apply_filters(row: tuple) -> dict:
    (rank, ticker, co, sec, ff_rank) = row
    out = {
        "rank":                rank,
        "ticker":              ticker,
        "company_name":        co,
        "sector_provisional":  sec,
        "free_float_rank":     ff_rank,
        "f1_passed":           False,
        "f2_passed":           False,
        "f3_passed":           False,
        "f3_manual_ipo_check": False,
        "ipo_first_trade_date_str": "",
        "f1_reason":           "",
        "f2_reason":           "",
        "f3_reason":           "",
        "filters_passed":      False,
        "rejection_reason":    "",
    }

    # -------- download (DEV window only) ----------------------------------
    try:
        df = yf.download(
            ticker,
            start=DATA_START.isoformat(),
            end=(DATA_END + dt.timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        out["f1_reason"] = f"yfinance download failed: {e!r}"
        out["rejection_reason"] = out["f1_reason"]
        return out

    if df is None or df.empty:
        out["f1_reason"] = "yfinance returned empty DataFrame"
        out["rejection_reason"] = out["f1_reason"]
        return out

    # yfinance returns flat columns for single-ticker downloads since 0.2.x
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    if not isinstance(close, pd.Series):
        out["f1_reason"] = "Unexpected column structure in yfinance output"
        out["rejection_reason"] = out["f1_reason"]
        return out

    total_days = len(close)

    # -------- F1: data availability ---------------------------------------
    first_close = _first_valid_close(close)
    nan_frac = float(close.isna().mean())
    f1_ok = (
        first_close is not None
        and first_close <= F1_FIRST_CLOSE_DEADLINE
        and nan_frac <= F1_MAX_NAN_FRAC
    )
    if f1_ok:
        out["f1_passed"] = True
    else:
        parts = []
        if first_close is None:
            parts.append("all Close values are NaN")
        elif first_close > F1_FIRST_CLOSE_DEADLINE:
            parts.append(
                f"first valid Close {first_close.date()} > study-start deadline "
                f"{F1_FIRST_CLOSE_DEADLINE.date()}"
            )
        if nan_frac > F1_MAX_NAN_FRAC:
            parts.append(
                f"Close NaN fraction = {nan_frac:.2%} > {F1_MAX_NAN_FRAC:.0%}"
            )
        out["f1_reason"] = "; ".join(parts)

    # -------- F2: liquidity -----------------------------------------------
    vol_series = pd.to_numeric(volume, errors="coerce").fillna(0.0)
    illiquid_mask = (vol_series == 0.0) | (vol_series < 10_000.0)
    illiquid_frac = float(illiquid_mask.mean()) if total_days else 1.0
    if f1_ok and illiquid_frac <= F2_MAX_ILLIQUID_FRAC:
        out["f2_passed"] = True
    else:
        if not f1_ok:
            out["f2_reason"] = "skipped — F1 failed"
        else:
            out["f2_reason"] = (
                f"illiquid-day fraction = {illiquid_frac:.2%} > "
                f"{F2_MAX_ILLIQUID_FRAC:.0%} "
                f"({illiquid_mask.sum()}/{total_days} days with zero or <10k vol)"
            )

    # -------- F3: IPO / first-trade date ----------------------------------
    if f1_ok and first_close is not None:
        # Use the series first-close as a lower bound for first trade date
        # (conservative: the stock can't have listed after the first Close in
        # our data). Then combine with explicit yfinance info if available.
        ipo_info, ipo_manual = _ipo_first_trade_date(ticker)
        first_trade_candidates = [c for c in [ipo_info, first_close] if c is not None]
        first_trade = min(first_trade_candidates) if first_trade_candidates else None
        out["f3_manual_ipo_check"] = ipo_manual
        if first_trade is not None:
            out["ipo_first_trade_date_str"] = first_trade.strftime("%Y-%m-%d")
            if first_trade < F3_MIN_TRADE_DATE:
                if f1_ok and out["f2_passed"]:
                    out["f3_passed"] = True
                else:
                    # still mark F3 on its own merit but overall blocked by F1/F2
                    out["f3_passed"] = True
                    out["f3_reason"] = "passed but F1/F2 rejected overall"
            else:
                out["f3_reason"] = (
                    f"first trade date {first_trade.strftime('%Y-%m-%d')} "
                    f">= {F3_MIN_TRADE_DATE.strftime('%Y-%m-%d')} (IPO too late)"
                )
        else:
            # couldn't retrieve any first-trade evidence -> tentative pass,
            # but flagged for manual IPO review
            out["ipo_first_trade_date_str"] = ""
            if f1_ok and out["f2_passed"]:
                out["f3_passed"] = True
            out["f3_reason"] = (
                "first trade date unavailable via yfinance; IPO-date check "
                "must be MANUALLY confirmed against NSE listing records."
            )
    else:
        out["f2_reason"] = out["f2_reason"] or "skipped — F1 failed"
        out["f3_reason"] = "skipped — F1 failed"

    # -------- overall verdict ---------------------------------------------
    passed = out["f1_passed"] and out["f2_passed"] and out["f3_passed"]
    out["filters_passed"] = passed
    if not passed:
        reason = " | ".join(
            r for r in (out["f1_reason"], out["f2_reason"], out["f3_reason"]) if r
        )
        out["rejection_reason"] = reason
    elif out["f3_manual_ipo_check"]:
        out["rejection_reason"] = (
            "PASSED TENTATIVELY — manual IPO listing-date confirmation REQUIRED "
            "(see f3_reason)."
        )
    return out


def main() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # download and filter tickers one at a time (yfinance multi-ticker
    # download is often flaky for .NS with auto_adjust=True)
    # ------------------------------------------------------------------
    t0 = time.time()
    results: list[dict] = []
    for idx, row in enumerate(NIFTY50, 1):
        rank, ticker, _co, _sec, _ff = row
        print(f"[{idx:>2}/{len(NIFTY50)}] {ticker:<18s}", end="", flush=True)
        r = _apply_filters(row)
        status = "PASS" if r["filters_passed"] else "REJECT"
        extra = (
            " [IPO MANUAL-CHECK]" if (r["filters_passed"] and r["f3_manual_ipo_check"]) else ""
        )
        print(f"  -> {status}{extra}")
        results.append(r)
        # be polite to Yahoo Finance API
        time.sleep(0.35)
    elapsed = time.time() - t0

    all_df = pd.DataFrame(results).sort_values("rank").reset_index(drop=True)

    # ------------------------------------------------------------------
    # write docs/nifty50_sector_map.csv — all 50 tickers, full filter trail
    # ------------------------------------------------------------------
    sector_map_cols = [
        "rank", "ticker", "company_name", "sector_provisional",
        "free_float_rank", "f1_passed", "f2_passed", "f3_passed",
        "f3_manual_ipo_check", "ipo_first_trade_date_str",
        "f1_reason", "f2_reason", "f3_reason",
        "filters_passed", "rejection_reason",
    ]
    sector_map_path = DOCS / "nifty50_sector_map.csv"
    all_df[sector_map_cols].to_csv(sector_map_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # write data/raw/universe_frozen.csv — only filter survivors
    # ------------------------------------------------------------------
    survivors = all_df[all_df["filters_passed"]].copy()
    frozen_cols = [
        "rank", "ticker", "company_name", "sector_provisional",
        "free_float_rank", "ipo_first_trade_date_str", "f3_manual_ipo_check",
    ]
    frozen_path = DATA_RAW / "universe_frozen.csv"
    survivors[frozen_cols].to_csv(frozen_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # summary report to stdout
    # ------------------------------------------------------------------
    n_total    = len(all_df)
    n_survived = int(all_df["filters_passed"].sum())
    n_f1_fail  = int((~all_df["f1_passed"]).sum())
    n_f2_fail  = int((all_df["f1_passed"] & ~all_df["f2_passed"]).sum())
    n_f3_fail  = int((all_df["f1_passed"] & all_df["f2_passed"] & ~all_df["f3_passed"]).sum())
    n_ipo_manual = int((all_df["filters_passed"] & all_df["f3_manual_ipo_check"]).sum())

    sep = "=" * 72
    print()
    print(sep)
    print("FREEZE_UNIVERSE SUMMARY  |  data window:",
          DATA_START.isoformat(), "→", DATA_END.isoformat())
    print(sep)
    print(f"  total tickers input        : {n_total}")
    print(f"  passed all 3 filters       : {n_survived}  "
          f"({100*n_survived/n_total:.1f}% of input)")
    print(f"  ├─ F1 data-avail rejections: {n_f1_fail}")
    print(f"  ├─ F2 liquidity rejections : {n_f2_fail}")
    print(f"  └─ F3 IPO-date rejections  : {n_f3_fail}")
    print()
    print(f"  survivors needing IPO-date MANUAL CHECK: {n_ipo_manual}")
    print(f"  wall-clock time             : {elapsed:.1f} s")
    print()
    print("Outputs written:")
    print(f"  {frozen_path.relative_to(ROOT)}   ({len(survivors)} rows)")
    print(f"  {sector_map_path.relative_to(ROOT)} ({len(all_df)} rows, full trail)")
    print(sep)

    rejected = all_df[~all_df["filters_passed"]]
    if len(rejected):
        print("\nRejection log (rank | ticker | reason):")
        for _, r in rejected.iterrows():
            print(f"   #{r['rank']:<2d}  {r['ticker']:<18s}  {r['rejection_reason']}")
        print()

    if n_ipo_manual:
        print("IPO manual-check tickers (rank | ticker | provisional first trade):")
        mc = all_df[all_df["filters_passed"] & all_df["f3_manual_ipo_check"]]
        for _, r in mc.iterrows():
            print(f"   #{r['rank']:<2d}  {r['ticker']:<18s}  first-in-data: "
                  f"{r['ipo_first_trade_date_str'] or 'N/A'}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
