"""screener.py — indicator correctness, causality, screen integration. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener import (add_indicators, atr, bollinger, format_screen, macd, rsi, screen,
                      volume_zscore)


def _px(close, high=None, low=None, volume=None):
    n = len(close)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    close = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({
        "open": close,
        "high": pd.Series(high, index=idx, dtype=float) if high is not None else close,
        "low": pd.Series(low, index=idx, dtype=float) if low is not None else close,
        "close": close,
        "volume": pd.Series(volume, index=idx, dtype=float) if volume is not None
                  else pd.Series(1e6, index=idx),
    })


# ─── individual indicators

def test_rsi_extremes_and_flat():
    up = rsi(pd.Series(np.arange(1.0, 61.0)))          # gains only → 100
    assert up.iloc[-1] == pytest.approx(100.0)
    down = rsi(pd.Series(np.arange(60.0, 0.0, -1.0)))  # losses only → 0
    assert down.iloc[-1] == pytest.approx(0.0)
    flat = rsi(pd.Series(np.full(60, 42.0)))           # no movement → neutral 50
    assert flat.iloc[-1] == pytest.approx(50.0)
    assert flat.iloc[:14].isna().all()                 # warm-up is NaN, not fake values


def test_rsi_bounded_on_random_walk():
    rng = np.random.default_rng(0)
    series = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))))
    r = rsi(series).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_macd_constant_is_zero():
    line, sig, hist = macd(pd.Series(np.full(100, 55.0)))
    assert line.abs().max() == pytest.approx(0.0)
    assert sig.abs().max() == pytest.approx(0.0)
    assert hist.abs().max() == pytest.approx(0.0)


def test_bollinger_flat_bands():
    pct_b, bandwidth, mid = bollinger(pd.Series(np.full(60, 10.0)))
    assert pct_b.iloc[-1] == pytest.approx(0.5)        # zero-width band → centred
    assert bandwidth.iloc[-1] == pytest.approx(0.0)
    assert mid.iloc[-1] == pytest.approx(10.0)


def test_bollinger_pct_b_tracks_position():
    # steady rise: the latest close sits in the upper half of its own trailing band
    pct_b, _, _ = bollinger(pd.Series(np.linspace(100, 150, 80)))
    assert pct_b.iloc[-1] > 0.5


def test_atr_constant_true_range():
    n = 80
    close = np.full(n, 100.0)
    px_atr = atr(pd.Series(close + 1.0), pd.Series(close - 1.0), pd.Series(close))
    assert px_atr.iloc[-1] == pytest.approx(2.0)       # TR is exactly 2 every bar


def test_volume_zscore_flat_is_zero():
    z = volume_zscore(pd.Series(np.full(60, 5e6)))
    assert z.iloc[-1] == pytest.approx(0.0)


# ─── causality: past indicator values must not change when the future changes

def test_indicators_are_causal():
    rng = np.random.default_rng(1)
    n = 300
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    px = _px(close, high=close * 1.01, low=close * 0.99,
             volume=rng.integers(1, 5, n) * 1e6)
    base = add_indicators(px)

    mutated = px.copy()
    mutated.iloc[250:] *= 7.0                           # violently rewrite the future
    after = add_indicators(mutated)

    pd.testing.assert_frame_equal(base.iloc[:250], after.iloc[:250])


# ─── screen integration (fake provider, no network)

def test_screen_one_row_per_ticker(fake_md):
    table = screen(["AAA", "BBB", "CCC"], end="2022-12-31", md=fake_md)
    assert list(table["tic"].sort_values()) == ["AAA", "BBB", "CCC"]
    assert {"rsi14", "macd_side", "bb_state", "trend", "atr_pct"} <= set(table.columns)
    assert table["rsi_zone"].isin(["oversold", "neutral", "overbought"]).all()
    assert table["trend"].isin(["up", "down", "mixed"]).all()


def test_screen_skips_bad_ticker(fake_md):
    real = type(fake_md.source).fetch

    def flaky(self, tic, start, end):
        if tic == "BAD":
            raise RuntimeError("provider exploded")
        return real(self, tic, start, end)

    type(fake_md.source).fetch = flaky
    try:
        msgs = []
        table = screen(["AAA", "BAD"], end="2022-12-31", md=fake_md, log=msgs.append)
        assert list(table["tic"]) == ["AAA"]
        assert any("BAD" in m for m in msgs)
    finally:
        type(fake_md.source).fetch = real


def test_snapshot_detects_crash_as_oversold(fake_md):
    # craft a crash: long flat history then a 25% slide over the last 15 bars
    from screener import snapshot
    n = 300
    close = np.concatenate([np.full(n - 15, 100.0),
                            100.0 * np.linspace(1.0, 0.75, 15)])
    px = _px(close)
    row = snapshot(px, "CRASH")
    assert row["rsi_zone"] == "oversold"
    assert row["trend"] == "down"
    assert row["off_52w_high"] < -20


def test_format_screen_renders(fake_md):
    table = screen(["AAA", "BBB"], end="2022-12-31", md=fake_md)
    text = format_screen(table)
    assert "AAA" in text and "BBB" in text
    assert text.count("\n") >= 3                        # header + rule + 2 rows
