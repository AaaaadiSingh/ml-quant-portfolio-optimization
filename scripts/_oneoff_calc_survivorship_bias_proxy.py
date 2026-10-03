"""
One-off script: compute a PROXY for survivorship bias in the frozen
46-ticker universe (NIFTY 50 as-of 2026, survivors only, no historical
constituent series available at project scope).

Methodology (CHECKPOINT §5.3 rule (a)):
  For every ticker T in data/raw/universe_frozen.csv:
    1. Pull 2015-01-01 → 2023-12-31 OHLCV via the same raw Yahoo v8 chart
       helper used by freeze_universe.py (UA-patched session, avoids 429).
    2. Take the Close series (dividend/split-adjusted via Yahoo's native
       includeAdjustedClose=true → mapped to Close column, identical to
       yfinance auto_adjust=True semantics).
    3. Compute:
         P_start = first non-NaN Close on or after 2015-01-01
         P_end   = last non-NaN Close on or before 2023-12-31
         n_years = (date_end - date_start).days / 365.25
         CAGR_i = (P_end / P_start) ** (1/n_years) - 1
    4. Sort the 46 CAGR rows by `free_float_rank` ascending (smallest rank
       = top free-float weight in NIFTY 50, largest market-cap tier).
    5. Take mean of the 10 smallest ranks (top-10 free-float bucket) and
       mean of the 10 largest ranks (bottom-10 free-float bucket).
    6. Report:
         mean_cagr_top10, mean_cagr_bottom10, spread_bps = (top10 - bottom10)*1e4

The number this script produces is NOT a true survivorship-bias estimate
(which would require historical constituent series, merger/acquisition
delisting returns, etc.).  It is the WITHIN-SURVIVORS free-float-size
CAGR spread: if the 10 smallest-free-float survivors have LOWER 9-year
CAGRs than the 10 largest-free-float survivors, then dropping delisted
names (disproportionately the smaller, worse-performing ones) biases
our naive baseline CAGRs UPWARD by at least ~spread_bps.  We report
exactly that: as a LOW-BOUND PROXY, not a true estimate.

Outputs:
  - Per-ticker CSV: data/raw/_survivorship_bias_proxy_cagrs.csv
  - Stdout: per-bucket means + spread_bps (the number used to replace
    the 25-75 bps placeholder in docs/literature_matrix.md §E).
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
UF_CSV = ROOT / "data" / "raw" / "universe_frozen.csv"
OUT_CSV = ROOT / "data" / "raw" / "_survivorship_bias_proxy_cagrs.csv"

UA_STRING = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36"
)
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = UA_STRING
_SESSION.headers["Accept"] = (
    "application/json,text/html;q=0.9,application/xhtml+xml,"
    "application/xml;q=0.9,*/*;q=0.8"
)
_SESSION.headers["Accept-Language"] = "en-US,en;q=0.9"

START = dt.date(2015, 1, 1)
END = dt.date(2023, 12, 31)
INTERVAL_DAYS = 365.25


def _raw_chart(ticker: str) -> pd.DataFrame:
    p1 = int(dt.datetime(START.year, START.month, START.day).timestamp())
    p2 = int(dt.datetime(END.year, END.month, END.day).timestamp() + 86400)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    r = _SESSION.get(
        url,
        params=dict(
            period1=p1, period2=p2, interval="1d",
            includeAdjustedClose="true", events="div,splits",
        ),
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {ticker}: {r.text[:200]!r}")
    payload = r.json()
    result = payload.get("chart", {}).get("result") or []
    if not result:
        return pd.DataFrame(columns=["Close"])
    r0 = result[0]
    ts = r0.get("timestamp") or []
    quote = (r0.get("indicators") or {}).get("quote") or [{}]
    adj = (
        (((r0.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose"))
    )
    if not ts:
        return pd.DataFrame(columns=["Close"])
    q0 = quote[0]
    idx = pd.to_datetime(ts, unit="s").tz_localize(None).normalize()
    close = adj if adj is not None else q0.get("close")
    return pd.DataFrame({"Close": close}, index=idx).sort_index()


def _cagr_for_ticker(ticker: str) -> tuple[float | None, str | None]:
    try:
        df = _raw_chart(ticker)
        if df.empty or df["Close"].notna().sum() < 100:
            return None, f"not enough closes ({df['Close'].notna().sum()})"
        c = df["Close"].dropna()
        start_idx = c.index.asof(pd.Timestamp(START))
        end_idx = c.index.asof(pd.Timestamp(END))
        if start_idx is None or end_idx is None or start_idx == end_idx:
            return None, f"bad start_idx={start_idx} end_idx={end_idx}"
        p_start = float(c.loc[start_idx])
        p_end = float(c.loc[end_idx])
        if not (p_start > 0 and p_end > 0):
            return None, f"non-positive price start={p_start} end={p_end}"
        n_years = (pd.Timestamp(END) - pd.Timestamp(START)).days / INTERVAL_DAYS
        # CAGR formula: (P_end / P_start) ** (1/n) - 1
        cagr = (p_end / p_start) ** (1.0 / n_years) - 1.0
        return cagr, f"{p_start:.2f}→{p_end:.2f} n={n_years:.3f}y start={start_idx.date()} end={end_idx.date()}"
    except Exception as e:
        return None, f"err: {e!r}"


def main() -> int:
    if not UF_CSV.exists():
        print(f"ERROR: universe_frozen.csv not found at {UF_CSV}", file=sys.stderr)
        return 2
    uf = pd.read_csv(UF_CSV)
    if "free_float_rank" not in uf.columns:
        print("ERROR: universe_frozen.csv missing free_float_rank", file=sys.stderr)
        return 2
    uf = uf.sort_values("free_float_rank").reset_index(drop=True)

    rows = []
    t0 = time.time()
    for i, rec in enumerate(uf.to_dict("records"), 1):
        tkr = rec["ticker"]
        ff = rec["free_float_rank"]
        cagr, note = _cagr_for_ticker(tkr)
        rows.append({
            "free_float_rank": ff,
            "ticker": tkr,
            "company_name": rec.get("company_name", ""),
            "cagr_2015_2023": cagr,
            "note": note or "",
        })
        status = "OK" if cagr is not None else "MISS"
        pct = cagr * 100 if cagr is not None else float("nan")
        print(f"[{i:2d}/{len(uf):2d}] {tkr:17s} ff_rank={ff:2d} {status} cagr={pct:7.3f}%  {note}")
        # tiny polite delay (0.15s per ticker; 46 * 0.15s = ~7s extra)
        time.sleep(0.15)

    wall = time.time() - t0
    print(f"\nPull complete: {wall:.1f}s wall-clock")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}  ({len(out)} rows)\n")

    valid = out.dropna(subset=["cagr_2015_2023"]).sort_values("free_float_rank").reset_index(drop=True)
    n_valid = len(valid)
    print(f"Valid CAGR rows: {n_valid}/{len(out)}")
    if n_valid < 20:
        print("WARNING: fewer than 20 valid CAGR rows — spread estimate is noisy.")

    top10 = valid.nsmallest(10, "free_float_rank")["cagr_2015_2023"]
    bot10 = valid.nlargest(10, "free_float_rank")["cagr_2015_2023"]

    mean_top = float(top10.mean())
    mean_bot = float(bot10.mean())
    spread_bps = (mean_top - mean_bot) * 1e4

    print("\n=== WITHIN-SURVIVORS FREE-FLOAT-SIZE CAGR SPREAD ===")
    print(f"  Top-10 survivors (smallest free_float_rank, largest tier):")
    for _, r in valid.nsmallest(10, "free_float_rank").iterrows():
        print(f"    ff={int(r['free_float_rank']):2d}  {r['ticker']:17s}  CAGR={r['cagr_2015_2023']*100:7.3f}%")
    print(f"  → mean CAGR top10    = {mean_top*100:7.3f} %/yr")
    print()
    print(f"  Bottom-10 survivors (largest free_float_rank, smallest tier):")
    for _, r in valid.nlargest(10, "free_float_rank").iterrows():
        print(f"    ff={int(r['free_float_rank']):2d}  {r['ticker']:17s}  CAGR={r['cagr_2015_2023']*100:7.3f}%")
    print(f"  → mean CAGR bottom10 = {mean_bot*100:7.3f} %/yr")
    print()
    print(f"  Spread (top10 − bottom10) = {spread_bps: .1f} basis points annualized")
    print()
    print("  INTERPRETATION (§E survivorship bias):")
    print(f"    Because we selected the 2026 NIFTY 50 list (today's winners) and")
    print(f"    therefore dropped every name that was in NIFTY 50 during 2015–2023")
    print(f"    but was later removed (merger, de-listing, free-float decline,")
    print(f"    corporate action, etc.), and such drops are disproportionately the")
    print(f"    smaller / worse-performing names, a LOW-BOUND PROXY for the upward")
    print(f"    survivorship bias in naive baseline CAGRs is the within-survivors")
    print(f"    top-free-float vs. bottom-free-float CAGR spread = {spread_bps:.0f} bps")
    print(f"    = {mean_top*100:.2f}%/yr − {mean_bot*100:.2f}%/yr.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
