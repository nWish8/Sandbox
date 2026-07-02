"""Shared pytest fixtures for Signal Gym tests.

Ensures the repo root is importable so ``import gym.<module>`` resolves to this package
regardless of where pytest is invoked from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]   # .../Sandbox  (enables `import gym.<module>`)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The FinRL-side modules moved in from finrl_lab (pipeline, finrl_patch, portfolio, …) use
# flat imports (`from pipeline import …`), matching how they run as scripts. Put gym/ on the
# path so those resolve under pytest too.
GYM = Path(__file__).resolve().parents[1]    # .../Sandbox/gym
if str(GYM) not in sys.path:
    sys.path.insert(0, str(GYM))


@pytest.fixture
def config():
    from gym.config import Config
    return Config()


@pytest.fixture
def tmp_cfg(tmp_path):
    """A Config whose data/results/splits all live under tmp, seeded from the real
    signal_study results so feature tests run on realistic data without touching the
    repo's gym/data."""
    import shutil
    from gym.config import Config, SIGNAL_STUDY_RESULTS, TICKERS_FILE

    res = tmp_path / "results"
    res.mkdir()
    for f in SIGNAL_STUDY_RESULTS.glob("effectiveness_*.json"):
        shutil.copy(f, res / f.name)
    tickers = tmp_path / "tickers.txt"
    tickers.write_text(TICKERS_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    return Config(
        signal_study_results=res,
        tickers_file=tickers,
        gym_data=tmp_path / "data",
        splits_file=tmp_path / "data" / "splits.json",
    )


@pytest.fixture
def write_fixture(tmp_path):
    """Factory: write a minimal effectiveness_<slug>.json with controllable 1D closes
    and signal fires. Returns (config, write_fn)."""
    import json
    from gym.config import Config

    res = tmp_path / "results"
    res.mkdir()
    tickers = tmp_path / "tickers.txt"

    def _write(symbol, t, c, signals=None, baseline_h6=0.0, ohlc=None):
        from gym.config import slug as sl
        tf1d = {
            "label": "1D", "resolution": "D", "ohlc_bars": len(c),
            "baseline": {"long": {"6": [len(c), 50.0, baseline_h6 * 100.0]},
                         "short": {"6": [len(c), 50.0, -baseline_h6 * 100.0]}},
            "series": {"t": list(t), "c": list(c)},
            "signals": signals or {},
        }
        if ohlc:
            tf1d["series"].update(ohlc)
        data = {"symbol": symbol, "timeframes": [tf1d]}
        (res / f"effectiveness_{sl(symbol)}.json").write_text(json.dumps(data), encoding="utf-8")
        tickers.write_text((tickers.read_text(encoding="utf-8") if tickers.exists() else "")
                           + symbol + "\n", encoding="utf-8")

    cfg = Config(signal_study_results=res, tickers_file=tickers,
                 gym_data=tmp_path / "data", splits_file=tmp_path / "data" / "splits.json")
    return cfg, _write
