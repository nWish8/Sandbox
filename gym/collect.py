"""collect.py — download 1-hour OHLCV from Yahoo (yfinance) in FinRL long format.

Replaces the old TradingView-CDP scrape (rev 4). For each Yahoo symbol in the manifest
we pull ~730 days of 1h bars and write a per-ticker raw parquet with the FinRL standard
columns ``date, tic, open, high, low, close, volume``. Prices are auto-adjusted
(Yahoo adjusted close), matching FinRL's convention.

This is the only network step; everything downstream (build/evolve/validate) is offline.

Run:  python -m gym.run collect            [--force] [SYM ...]
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from gym.config import DEFAULT_CONFIG, Config, load_tickers, slug

log = logging.getLogger(__name__)


def _to_finrl(raw: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Normalise a yfinance frame to the FinRL long schema for one ticker."""
    if raw is None or len(raw) == 0:
        return None
    df = raw.copy()
    # yfinance may return MultiIndex columns (field, ticker); keep the field level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    dcol = next((c for c in ("datetime", "date", "index") if c in df.columns), df.columns[0])
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        return None

    out = pd.DataFrame({
        "date": pd.to_datetime(df[dcol], utc=True),
        "tic": symbol,
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0.0),
    })
    out = (out.dropna(subset=["open", "high", "low", "close"])
              .sort_values("date")
              .drop_duplicates(subset=["date"])
              .reset_index(drop=True))
    return out


def collect(tickers: list[str] | None = None, cfg: Config = DEFAULT_CONFIG,
            force: bool = False) -> list[str]:
    """Download every symbol's 1h OHLCV → ``cfg.raw_data/<slug>.parquet``.

    Resumable: a symbol with an existing raw parquet is skipped unless *force*.
    Returns the list of symbols with usable data this run.
    """
    import yfinance as yf

    if tickers is None:
        tickers = load_tickers(cfg.tickers_file)
    cfg.raw_data.mkdir(parents=True, exist_ok=True)

    min_bars = cfg.lookback + 10
    done, failed = [], []
    t0 = time.time()
    log.info("Downloading %d tickers (%s bars, interval=%s) → %s",
             len(tickers), cfg.period, cfg.interval, cfg.raw_data)

    for i, sym in enumerate(tickers, 1):
        path = cfg.raw_data / f"{slug(sym)}.parquet"
        if path.exists() and not force:
            log.info("[%d/%d] %s already downloaded — skipping", i, len(tickers), sym)
            done.append(sym)
            continue

        try:
            raw = yf.download(sym, period=cfg.period, interval=cfg.interval,
                              progress=False, auto_adjust=True)
        except Exception as exc:  # one bad ticker shouldn't kill the sweep
            log.warning("[%d/%d] %s download error: %s", i, len(tickers), sym, exc)
            failed.append(sym)
            continue

        df = _to_finrl(raw, sym)
        if df is None or len(df) < min_bars:
            n = 0 if df is None else len(df)
            log.warning("[%d/%d] %s: only %d bars (<%d) — skipping",
                        i, len(tickers), sym, n, min_bars)
            failed.append(sym)
            continue

        df.to_parquet(path, index=False)
        log.info("[%d/%d] %s: %d bars  %s → %s", i, len(tickers), sym, len(df),
                 str(df["date"].iloc[0])[:10], str(df["date"].iloc[-1])[:10])
        done.append(sym)

    log.info("Collection done in %.0fs — %d ok, %d failed%s",
             time.time() - t0, len(done), len(failed),
             (": " + ", ".join(failed)) if failed else "")
    return done
