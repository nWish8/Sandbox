"""finrl_patch.py — compatibility shims for FinRL 0.3.7 against modern yfinance/pandas.

Importing this module monkeypatches FinRL so its examples run unchanged on this stack:

  * YahooDownloader.fetch_data: FinRL 0.3.7 assumes an 8-column yfinance layout, but
    yfinance 0.2.65 returns auto-adjusted MultiIndex columns (7 after reset). We replace
    fetch_data with a version that flattens the MultiIndex and emits FinRL's exact
    contract: long df [date(str), open, high, low, close, volume, tic, day], adjusted
    close, sorted by (date, tic). This also fixes FeatureEngineer.add_vix (it uses the
    same downloader for ^VIX).

Import this BEFORE using any FinRL downloader/feature code.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from finrl.meta.preprocessor import yahoodownloader as _yd


def _fetch_data(self, proxy=None) -> pd.DataFrame:
    frames = []
    for tic in self.ticker_list:
        df = yf.download(tic, start=self.start_date, end=self.end_date,
                         auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        dcol = next((c for c in ("date", "datetime", "index") if c in df.columns),
                    df.columns[0])
        out = pd.DataFrame({
            "date": pd.to_datetime(df[dcol]),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),   # auto-adjusted
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
            "tic": tic,
        })
        frames.append(out)
    if not frames:
        raise ValueError("no data is fetched.")
    data = pd.concat(frames, ignore_index=True)
    data["day"] = data["date"].dt.dayofweek
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    data = data.dropna().sort_values(["date", "tic"]).reset_index(drop=True)
    print("Shape of DataFrame: ", data.shape)
    return data


_yd.YahooDownloader.fetch_data = _fetch_data
print("[finrl_patch] YahooDownloader.fetch_data patched for yfinance", yf.__version__)
