from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"

UA_STRING = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36"
)

_yf_session_lock = threading.Lock()
_yf_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _yf_session
    if _yf_session is None:
        with _yf_session_lock:
            if _yf_session is None:
                s = requests.Session()
                s.headers.update(
                    {
                        "User-Agent": UA_STRING,
                        "Accept": "application/json,text/html,application/xhtml+xml",
                        "Accept-Language": "en-US,en;q=0.9",
                    }
                )
                _yf_session = s
    return _yf_session


def _raw_yahoo_chart(
    ticker: str,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    interval: str = "1d",
    include_adj: bool = True,
    events: str = "div,splits",
    timeout: int = 30,
    max_retries: int = 5,
) -> pd.DataFrame:
    import datetime as dt

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert(None)
    period1 = int(dt.datetime(start_ts.year, start_ts.month, start_ts.day).timestamp())
    period2 = int(dt.datetime(end_ts.year, end_ts.month, end_ts.day).timestamp())
    params = {
        "period1": period1,
        "period2": period2,
        "interval": interval,
        "includeAdjustedClose": "true" if include_adj else "false",
        "events": events,
    }
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _get_session().get(url, params=params, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"yahoo chart HTTP {resp.status_code} for {ticker}: {resp.text[:200]}"
                )
            payload = resp.json()
            results = payload.get("chart", {}).get("result")
            if not results or results[0] is None:
                err = (payload.get("chart", {}) or {}).get("error")
                raise RuntimeError(
                    f"yahoo chart null result for {ticker}: {err!r}. "
                    "No data found, symbol may be delisted"
                )
            res0 = results[0]
            ts = res0.get("timestamp") or []
            if not ts:
                return pd.DataFrame(
                    columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                )
            idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
            quote = (res0.get("indicators", {}) or {}).get("quote", [{}])[0] or {}
            adj_series = None
            if include_adj:
                ac = (res0.get("indicators", {}) or {}).get("adjclose")
                if ac:
                    adj_series = ac[0].get("adjclose")
            o = quote.get("open") or [None] * len(ts)
            h = quote.get("high") or [None] * len(ts)
            l = quote.get("low") or [None] * len(ts)
            c = quote.get("close") or [None] * len(ts)
            v = quote.get("volume") or [None] * len(ts)
            if adj_series is None:
                adj_series = list(c)
            df = pd.DataFrame(
                {
                    "Open": o,
                    "High": h,
                    "Low": l,
                    "Close": c,
                    "Adj Close": adj_series,
                    "Volume": v,
                },
                index=idx,
            )
            if include_adj:
                df["Close"] = df["Adj Close"]
            df = df[~df.index.duplicated(keep="last")].sort_index()
            return df
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_s = min(2.0 ** (attempt - 1), 15.0)
                time.sleep(sleep_s)
            else:
                break
    raise RuntimeError(
        f"_raw_yahoo_chart failed for {ticker} after {max_retries} attempts: {last_exc!r}"
    ) from last_exc


FinalHoldoutDate = pd.Timestamp("2024-01-01")


def load_prices(
    tickers: Iterable[str],
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    *,
    final_holdout: bool = False,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
) -> pd.DataFrame:
    if not isinstance(final_holdout, bool):
        raise ValueError(
            "final_holdout must be a Python bool (keyword-only). "
            "Pass final_holdout=True only in the Phase 8 final-evaluation script; "
            "all other callers MUST leave it defaulted to False to avoid "
            "accidental final-holdout look-ahead."
        )
    if cache_dir is None:
        cache_dir = DATA_RAW
    cache_path = Path(cache_dir)
    wanted = list(tickers)
    frames: dict[str, pd.DataFrame] = {}
    for tkr in wanted:
        pqt = cache_path / f"{tkr}.parquet"
        csv = cache_path / f"{tkr}.csv"
        df_t: Optional[pd.DataFrame] = None
        fetched_from_network = False
        if pqt.exists():
            df_t = pd.read_parquet(pqt)
        elif csv.exists():
            df_t = pd.read_csv(csv, index_col=0, parse_dates=True)
        if df_t is None or df_t.empty:
            df_t = _raw_yahoo_chart(tkr, start_date, end_date)
            fetched_from_network = True
        if df_t is None or df_t.empty:
            continue
        df_t = df_t[~df_t.index.duplicated(keep="last")].sort_index()
        if fetched_from_network:
            try:
                df_t.to_parquet(pqt)
            except Exception:
                try:
                    df_t.to_csv(csv)
                except Exception:
                    pass
        frames[tkr] = df_t
    if not frames:
        return pd.DataFrame()
    closes = {t: frames[t]["Close"] for t in frames}
    opens = {t: frames[t]["Open"] for t in frames}
    highs = {t: frames[t]["High"] for t in frames}
    lows = {t: frames[t]["Low"] for t in frames}
    vols = {t: frames[t]["Volume"] for t in frames}
    result = pd.concat(
        [
            pd.DataFrame(closes).sort_index(),
            pd.DataFrame(opens).sort_index(),
            pd.DataFrame(highs).sort_index(),
            pd.DataFrame(lows).sort_index(),
            pd.DataFrame(vols).sort_index(),
        ],
        axis=1,
        keys=["Close", "Open", "High", "Low", "Volume"],
    )
    result.index = pd.to_datetime(result.index)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    result = result.loc[(result.index >= start_ts) & (result.index <= end_ts)]
    if not final_holdout:
        result = result.loc[result.index < FinalHoldoutDate]
    return result
