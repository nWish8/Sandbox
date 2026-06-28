"""features.py — build FinRL-schema feature matrices from raw 1h OHLCV (yfinance).

Rev 4 (FinRL migration): the per-bar frame conforms to FinRL's standard layout —
``date, tic, open, high, low, close, volume`` followed by the price-derived technical
indicators the agent observes (``config.TECH_INDICATORS``). All Prophets/signal,
condition, interaction, context and supervised-target columns were removed; the only
model inputs are price-derived. RSI(14) is computed for charting only and stored as a
``_display_`` column excluded from the feature matrix.

Causality (headline invariant): every indicator at bar t uses only bars <= t.
"""
from __future__ import annotations

import json
import logging
import random

import numpy as np
import pandas as pd

from gym.config import (
    DEFAULT_CONFIG,
    Config,
    asset_class_of,
    load_tickers,
    slug as ticker_slug,
)

log = logging.getLogger(__name__)

# Columns never fed to the agent: identity/exec/raw-bar + display-only.
META_COLS = ["date", "tic", "ticker", "timestamp", "open", "high", "low", "close", "volume"]


# ----------------------------------------------------------------- price features
def _wilder_rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder's RSI(n) — causal, classic smoothing (EMA with alpha=1/n)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _price_features(df: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    """The FinRL tech_indicators (config.TECH_INDICATORS), all trailing/causal."""
    c, h, l = df["close"], df["high"], df["low"]
    lc = np.log(c.where(c > 0))
    w = cfg.vol_window
    ret_1 = lc.diff(1)
    lo = l.rolling(w, min_periods=w).min()
    hi = h.rolling(w, min_periods=w).max()
    span = (hi - lo).replace(0, np.nan)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return {
        "ret_1": ret_1,
        "ret_3": lc - lc.shift(3),
        "ret_6": lc - lc.shift(6),
        "vol_20": ret_1.rolling(w, min_periods=w).std(),
        "range_pos_20": ((c - lo) / span).fillna(0.5),
        "drawdown_20": (c / hi - 1.0),
        "atr_14": tr.rolling(cfg.atr_window, min_periods=cfg.atr_window).mean() / c,
    }


# ---------------------------------------------------------------- ticker frame build
def build_ticker_frame(symbol: str, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Read one ticker's raw OHLCV parquet → FinRL-schema feature frame."""
    raw = pd.read_parquet(cfg.raw_data / f"{ticker_slug(symbol)}.parquet")
    raw = raw.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    out = pd.DataFrame({
        "date": pd.to_datetime(raw["date"], utc=True),
        "tic": symbol,
        "open": raw["open"].astype("float64"),
        "high": raw["high"].astype("float64"),
        "low": raw["low"].astype("float64"),
        "close": raw["close"].astype("float64"),
        "volume": raw["volume"].astype("float64"),
    })

    feats = _price_features(out, cfg)
    for name in cfg.tech_indicators:
        out[name] = feats[name].astype("float32")
    out["_display_rsi14"] = _wilder_rsi(out["close"], cfg.rsi_window).astype("float32")

    # the agent needs NaN-free observations; fill warmup NaNs in model-input columns
    tech = list(cfg.tech_indicators)
    out[tech] = out[tech].fillna(0.0)
    out["_display_rsi14"] = out["_display_rsi14"].fillna(50.0)
    return out


def feature_columns(df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> list[str]:
    """The agent's observation columns: the FinRL tech_indicators present in df."""
    return [c for c in cfg.tech_indicators if c in df.columns]


# ------------------------------------------------------------------------- splits
def make_splits(tickers: list[str], cfg: Config = DEFAULT_CONFIG) -> dict:
    """Stratified, deterministic 3-way TICKER split. Each asset class is allocated
    proportionally; training always receives >=1 of every class (tiny classes may land
    wholly in training). Validation/test span multiple classes for the generalization
    test."""
    r_train, r_val, r_test = cfg.split_ratios
    by_class: dict[int, list[str]] = {}
    for t in tickers:
        by_class.setdefault(asset_class_of(t), []).append(t)

    rng = random.Random(cfg.seed)
    train, val, test = [], [], []
    for ac in sorted(by_class, key=lambda x: (x is None, x)):
        group = sorted(by_class[ac])
        rng.shuffle(group)
        nn = len(group)
        n_val = int(round(nn * r_val))
        n_test = int(round(nn * r_test))
        if n_val + n_test >= nn:                       # guarantee >=1 in training
            n_test = max(0, nn - 1 - n_val)
            if n_val + n_test >= nn:
                n_val = max(0, nn - 1 - n_test)
        val += group[:n_val]
        test += group[n_val:n_val + n_test]
        train += group[n_val + n_test:]
    return {
        "training_tickers": sorted(train),
        "validation_tickers": sorted(val),
        "test_tickers": sorted(test),
        "stratify": cfg.stratify_col,
        "seed": cfg.seed,
    }


# --------------------------------------------------------------------- orchestration
def build_features(cfg: Config = DEFAULT_CONFIG) -> dict[str, pd.DataFrame]:
    """Build every ticker's FinRL-schema parquet + the 3-way split.

    Reads ``cfg.raw_data/<slug>.parquet`` (written by collect.py). Tickers with no raw
    data are skipped (run collect first).
    """
    tickers = load_tickers(cfg.tickers_file)
    present = [t for t in tickers if (cfg.raw_data / f"{ticker_slug(t)}.parquet").exists()]
    missing = [t for t in tickers if t not in present]
    if missing:
        log.warning("No raw data for %d tickers (run `collect`): %s",
                    len(missing), ", ".join(missing))
    if not present:
        raise FileNotFoundError(
            f"No raw parquets in {cfg.raw_data}. Run `python -m gym.run collect` first."
        )

    frames = {ticker_slug(t): build_ticker_frame(t, cfg) for t in present}

    cfg.gym_data.mkdir(parents=True, exist_ok=True)
    for s, df in frames.items():
        df.to_parquet(cfg.gym_data / f"{s}.parquet", index=False)

    splits = make_splits(present, cfg)
    splits["feature_columns"] = list(cfg.tech_indicators)
    splits["interval"] = cfg.interval
    cfg.splits_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.splits_file.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    log.info("Built %d frames | train=%d val=%d test=%d | %d features",
             len(frames), len(splits["training_tickers"]),
             len(splits["validation_tickers"]), len(splits["test_tickers"]),
             len(cfg.tech_indicators))
    return frames


def load_ticker(slug: str, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load a processed parquet, adding back-compat ``ticker``/``timestamp`` columns the
    env, population and replay layers read (kept stable across the FinRL migration)."""
    df = pd.read_parquet(cfg.gym_data / f"{slug}.parquet")
    if "ticker" not in df.columns:
        df["ticker"] = df["tic"]
    if "timestamp" not in df.columns:
        df["timestamp"] = (pd.to_datetime(df["date"], utc=True).astype("int64") // 10**9
                           ).astype("int64")
    return df
