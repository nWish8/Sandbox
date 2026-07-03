"""Shared pytest path setup + fixtures for Signal Gym tests.

Ensures the repo root is importable so ``import gym.<module>`` resolves to this package
regardless of where pytest is invoked from.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]   # .../Sandbox  (enables `import gym.<module>`)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The FinRL-side modules (pipeline, portfolio, marketdata, …) use flat imports
# (`from pipeline import …`), matching how they run as scripts. Put gym/ on the
# path so those resolve under pytest too.
GYM = Path(__file__).resolve().parents[1]    # .../Sandbox/gym
if str(GYM) not in sys.path:
    sys.path.insert(0, str(GYM))


class FakeOHLCVSource:
    """Deterministic synthetic daily bars; records fetch calls so cache tests can count."""

    name = "fake"

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.calls: list[tuple[str, str, str]] = []

    def fetch(self, tic: str, start: str, end: str):
        import numpy as np
        import pandas as pd

        self.calls.append((tic, start, end))
        dates = pd.date_range(start, end, freq="D")
        rng = np.random.default_rng((zlib.crc32(tic.encode()) + self.seed) % (2 ** 32))
        rets = rng.normal(0.0005, 0.01, len(dates))
        close = 100.0 * np.exp(np.cumsum(rets))
        spread = np.abs(rng.normal(0.005, 0.002, len(dates)))
        return pd.DataFrame({
            "date": dates,
            "open": close * (1 - spread / 2),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
        })


@pytest.fixture
def fake_md(tmp_path):
    """A MarketData facade wired to the fake provider and a tmp cache dir."""
    from marketdata import MarketData
    return MarketData(source=FakeOHLCVSource(), cache_dir=tmp_path / "ohlcv")
