"""screener.py — per-ticker technical screen on cached daily OHLCV (Vision M1).

Computes standard indicators with **causal** pandas operations only (rolling / ewm — a
value at bar t sees bars ≤ t, never later ones) and reduces each ticker to one snapshot
row: latest values plus discrete states (RSI zone, MACD side + bars since cross, Bollinger
position, trend vs SMAs, volume anomaly). The screen reports facts; it deliberately does
not manufacture a composite "buy score" — ranking a snapshot column is the caller's call.

    from marketdata import MarketData
    from screener import screen, format_screen
    table = screen(["AAPL", "MSFT"], md=MarketData())
    print(format_screen(table))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketdata import MarketData

# warm-up bars needed before the slowest indicator (52-week high) is meaningful
_DEFAULT_WINDOW_DAYS = 420


# ─────────────────────────────────────────── indicators (each: Series in → Series out, causal)

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI. All-gain history → 100, all-loss → 0, flat → 50."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    out = 100.0 - 100.0 / (1.0 + avg_up / avg_down)
    out = out.where(avg_down > 0, 100.0)                     # no losses in window → 100
    out[(avg_up == 0) & (avg_down == 0)] = 50.0              # perfectly flat → neutral
    out[avg_up.isna() | avg_down.isna()] = np.nan            # warm-up
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    line = (close.ewm(span=fast, adjust=False).mean()
            - close.ewm(span=slow, adjust=False).mean())
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0
              ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (%B, bandwidth, mid). Population std (ddof=0), the charting convention.
    Zero-width bands (flat prices) give %B = 0.5, bandwidth = 0."""
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = upper - lower
    pct_b = ((close - lower) / width).where(width > 0, 0.5)
    bandwidth = (width / mid).where(mid != 0)
    return pct_b, bandwidth, mid


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder ATR from true range (uses the *previous* close — causal)."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def volume_zscore(volume: pd.Series, n: int = 20) -> pd.Series:
    """Volume anomaly vs its trailing window; 0 when the window has zero variance."""
    mean = volume.rolling(n, min_periods=n).mean()
    sd = volume.rolling(n, min_periods=n).std(ddof=0)
    return ((volume - mean) / sd).where(sd > 0, 0.0)


def add_indicators(px: pd.DataFrame) -> pd.DataFrame:
    """Add all screen indicators to a wide OHLCV frame (DatetimeIndex, ohlcv columns)."""
    out = px.copy()
    c = out["close"]
    out["rsi14"] = rsi(c)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["bb_pct_b"], out["bb_bandwidth"], out["sma20"] = bollinger(c)
    out["sma50"] = c.rolling(50, min_periods=50).mean()
    out["atr14"] = atr(out["high"], out["low"], c)
    out["atr_pct"] = 100.0 * out["atr14"] / c
    out["vol_z"] = volume_zscore(out["volume"])
    out["ret_20d"] = c.pct_change(20)
    out["high_52w"] = c.rolling(252, min_periods=60).max()
    out["off_52w_high"] = c / out["high_52w"] - 1.0
    return out


# ─────────────────────────────────────────── snapshot + screen

def _bars_since_sign_change(s: pd.Series) -> int | None:
    """Bars since the latest sign flip of s (0 = flipped on the last bar)."""
    sign = np.sign(s.dropna().to_numpy())
    if len(sign) < 2:
        return None
    flips = np.nonzero(sign[1:] != sign[:-1])[0]
    return None if len(flips) == 0 else int(len(sign) - 2 - flips[-1])


def snapshot(px: pd.DataFrame, tic: str) -> dict:
    """Reduce one ticker's indicator frame to the latest screen row."""
    ind = add_indicators(px)
    last = ind.iloc[-1]
    r, pb = float(last["rsi14"]), float(last["bb_pct_b"])
    row = {
        "tic": tic,
        "date": ind.index[-1].strftime("%Y-%m-%d"),
        "close": float(last["close"]),
        "rsi14": round(r, 1),
        "rsi_zone": "oversold" if r < 30 else "overbought" if r > 70 else "neutral",
        "macd_side": "bull" if last["macd"] > last["macd_signal"] else "bear",
        "macd_cross_age": _bars_since_sign_change(ind["macd_hist"]),
        "bb_pct_b": round(pb, 2),
        "bb_state": "above" if pb > 1 else "below" if pb < 0 else "inside",
        "atr_pct": round(float(last["atr_pct"]), 2),
        "vol_z": round(float(last["vol_z"]), 1),
        "ret_20d": round(100 * float(last["ret_20d"]), 1),
        "off_52w_high": round(100 * float(last["off_52w_high"]), 1),
        "trend": ("up" if last["close"] > last["sma20"] > last["sma50"]
                  else "down" if last["close"] < last["sma20"] < last["sma50"]
                  else "mixed"),
    }
    return row


def screen(tickers: list[str], end: str | None = None, window_days: int = _DEFAULT_WINDOW_DAYS,
           md: MarketData | None = None, log=None) -> pd.DataFrame:
    """One snapshot row per ticker, as of `end` (default: today). Sorted by 20-day return."""
    md = md if md is not None else MarketData()
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.Timedelta(days=window_days)
    rows = []
    for tic in tickers:
        try:
            px = md.get_ohlcv(tic, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
            rows.append(snapshot(px, tic))
        except Exception as e:  # noqa: BLE001 — one bad ticker must not kill the screen
            if log is not None:
                log(f"[screen] {tic}: skipped ({e})")
    if not rows:
        raise ValueError("screen produced no rows (all tickers failed?)")
    return (pd.DataFrame(rows)
            .sort_values("ret_20d", ascending=False)
            .reset_index(drop=True))


def format_screen(table: pd.DataFrame) -> str:
    """Fixed-width text table for the CLI / panel log."""
    cols = ["tic", "close", "trend", "rsi14", "rsi_zone", "macd_side", "macd_cross_age",
            "bb_pct_b", "bb_state", "atr_pct", "vol_z", "ret_20d", "off_52w_high"]
    hdr = (f"{'tic':<8} {'close':>9} {'trend':<6} {'rsi':>5} {'zone':<10} {'macd':<5}"
           f" {'x-age':>5} {'%B':>6} {'bb':<7} {'atr%':>5} {'volz':>5} {'r20%':>6} {'off-hi%':>7}")
    lines = [hdr, "-" * len(hdr)]
    for _, r in table[cols].iterrows():
        age = "-" if r["macd_cross_age"] is None or pd.isna(r["macd_cross_age"]) \
            else int(r["macd_cross_age"])
        lines.append(
            f"{r['tic']:<8} {r['close']:>9.2f} {r['trend']:<6} {r['rsi14']:>5.1f}"
            f" {r['rsi_zone']:<10} {r['macd_side']:<5} {age!s:>5} {r['bb_pct_b']:>6.2f}"
            f" {r['bb_state']:<7} {r['atr_pct']:>5.2f} {r['vol_z']:>5.1f}"
            f" {r['ret_20d']:>6.1f} {r['off_52w_high']:>7.1f}")
    return "\n".join(lines)
