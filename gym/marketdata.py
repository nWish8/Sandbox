"""marketdata.py — provider-swappable OHLCV data layer with a local parquet cache (Vision M1).

Replaces FinRL's YahooDownloader (and the finrl_patch monkeypatch) as the project's only
data source. One facade, two views:

    md = MarketData()                                                  # yfinance + cache
    long_df = md.get_finrl(["AAPL", "MSFT"], "2020-01-01", "2024-01-01")  # FinRL contract
    px      = md.get_ohlcv("AAPL", "2020-01-01", "2024-01-01")            # wide, DatetimeIndex

Cache: one parquet per ticker under ``gym/data/ohlcv/1d/`` plus ``manifest.json`` recording
the widest fetched [start, end] range per ticker. A request fully inside that range is
served locally; a wider request refetches the union range from the provider and overwrites
the ticker's file (daily bars — a refetch is cheap, an incremental-merge bug is not).

Providers implement ``fetch(tic, start, end) -> DataFrame[date, open, high, low, close,
volume]`` with **adjusted** daily prices (so cached series are split/dividend-consistent).
Swap providers by passing another source to ``MarketData(source=...)``.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "data" / "ohlcv" / "1d"

OHLCV_COLS = ["date", "open", "high", "low", "close", "volume"]


def _slug(symbol: str) -> str:
    """Yahoo symbol -> filesystem-safe filename (``^VIX`` -> ``VIX``, ``BZ=F`` -> ``BZ_F``)."""
    return re.sub(r"[^0-9A-Za-z]+", "_", symbol).strip("_").upper()


# ─────────────────────────────────────────── providers

class YFinanceSource:
    """Free-tier default provider: yfinance daily bars, auto-adjusted OHLC."""

    name = "yfinance"

    def fetch(self, tic: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(tic, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=OHLCV_COLS)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        dcol = next((c for c in ("date", "datetime", "index") if c in df.columns),
                    df.columns[0])
        out = pd.DataFrame({
            "date": pd.to_datetime(df[dcol]).dt.tz_localize(None).dt.normalize(),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        })
        return out.dropna().sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────── the facade

class MarketData:
    """Cached, provider-swappable access to daily OHLCV."""

    def __init__(self, source=None, cache_dir: Path | str | None = None):
        self.source = source if source is not None else YFinanceSource()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_file = self.cache_dir / "manifest.json"

    # ---- manifest
    def _manifest(self) -> dict:
        if self._manifest_file.exists():
            return json.loads(self._manifest_file.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self, man: dict) -> None:
        self._manifest_file.write_text(json.dumps(man, indent=2, sort_keys=True),
                                       encoding="utf-8")

    # ---- core: one ticker, cached
    def _get_one(self, tic: str, start: str, end: str, force: bool = False) -> pd.DataFrame:
        """Daily OHLCV for one ticker over [start, end], from cache when covered."""
        man = self._manifest()
        entry = man.get(tic)
        fpath = self.cache_dir / f"{_slug(tic)}.parquet"

        covered = (entry is not None and fpath.exists()
                   and entry["start"] <= start and entry["end"] >= end)
        if covered and not force:
            df = pd.read_parquet(fpath)
        else:
            # fetch the union of the cached and requested range so coverage only grows
            f_start = min(start, entry["start"]) if entry else start
            f_end = max(end, entry["end"]) if entry else end
            fresh = self.source.fetch(tic, f_start, f_end)
            if len(fresh) == 0:
                if fpath.exists():
                    df = pd.read_parquet(fpath)   # provider empty/offline → serve stale cache
                else:
                    raise ValueError(f"no data for {tic!r} in {f_start}..{f_end} "
                                     f"(provider={getattr(self.source, 'name', '?')})")
            else:
                df = fresh
                df.to_parquet(fpath)
                man[tic] = {"start": f_start, "end": f_end,
                            "provider": getattr(self.source, "name", "?"),
                            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                self._save_manifest(man)

        s, e = pd.Timestamp(start), pd.Timestamp(end)
        out = df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)
        if len(out) == 0:
            raise ValueError(f"no bars for {tic!r} inside {start}..{end}")
        return out

    # ---- public views
    def get_ohlcv(self, tic: str, start: str, end: str, force: bool = False) -> pd.DataFrame:
        """Wide per-ticker frame: DatetimeIndex 'date', columns open/high/low/close/volume."""
        df = self._get_one(tic, start, end, force=force)
        return df.set_index("date")[["open", "high", "low", "close", "volume"]]

    def get_finrl(self, tickers: list[str], start: str, end: str,
                  force: bool = False, log=None) -> pd.DataFrame:
        """FinRL long contract: [date(str), open, high, low, close, volume, tic, day],
        adjusted close, sorted by (date, tic) — what YahooDownloader.fetch_data returned."""
        frames = []
        for tic in tickers:
            df = self._get_one(tic, start, end, force=force).copy()
            df["tic"] = tic
            frames.append(df)
            if log is not None:
                log(f"[marketdata] {tic}: {len(df)} bars")
        data = pd.concat(frames, ignore_index=True)
        data["day"] = data["date"].dt.dayofweek
        data["date"] = data["date"].dt.strftime("%Y-%m-%d")
        return data.dropna().sort_values(["date", "tic"]).reset_index(drop=True)
