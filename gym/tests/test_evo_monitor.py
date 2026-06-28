"""Tests for evo_monitor.py — the pure per-generation payload builder (not the Qt window)."""
from __future__ import annotations

import numpy as np
import pytest

from gym import features as F
from gym.baselines import BuyAndHoldPolicy
from gym.evo_monitor import build_generation_event, downsample
from gym.population import run_population
from gym.stats import objective_value

DAY = 86_400


class FlatPolicy:
    def act(self, obs):  # noqa: ARG002
        return 0.0
    def reset(self):
        pass


# ─────────────────────────────────── downsample

def test_downsample_passthrough_when_short():
    a = np.arange(50.0)
    out = downsample(a, 200)
    np.testing.assert_array_equal(out, a)


def test_downsample_reduces_length():
    a = np.arange(10_000.0)
    out = downsample(a, 200)
    assert len(out) == 200
    assert out[0] == 0.0 and out[-1] == 9_999.0


def test_downsample_preserves_nan():
    a = np.arange(1000.0)
    a[500:] = np.nan                       # ruin truncation
    out = downsample(a, 100)
    assert np.isnan(out[-1])               # tail still NaN after subsampling


# ─────────────────────────────────── build_generation_event

@pytest.fixture
def results(write_fixture):
    cfg, write = write_fixture
    cfg.lookback = 5
    rng = np.random.default_rng(1)
    for s in ("MON:A", "MON:B"):
        ts = [1_700_000_000 + i * DAY for i in range(60)]
        write(s, ts, list(100 * np.cumprod(1 + rng.normal(0.0008, 0.012, 60))))
    vocab = F.build_vocab(["MON:A", "MON:B"], cfg)
    dfs = []
    for s in ("MON:A", "MON:B"):
        df = F.build_ticker_frame(s, vocab, cfg)
        feat = F.feature_columns(df, cfg); df[feat] = df[feat].fillna(0.0)
        dfs.append(df)
    pols = [BuyAndHoldPolicy(), FlatPolicy(), BuyAndHoldPolicy()]
    return run_population(pols, dfs, cfg, objective="timing_sortino")


def test_event_structure(results):
    names = [f"agent {i:02d}" for i in range(len(results))]
    fits = [objective_value(r.stats, "timing_sortino") if not r.ruined else float("-inf")
            for r in results]
    ev = build_generation_event(3, results, fits, names, champion_idx=0, ruin_frac=0.45,
                                best_train=max(fits), mean_train=float(np.mean(fits)),
                                val_fit=0.1, n_points=50)
    assert ev["type"] == "generation" and ev["generation"] == 3
    assert len(ev["equity"]) == len(results)          # one curve per agent
    assert all(len(c) <= 50 for c in ev["equity"])    # downsampled
    assert len(ev["bh"]) <= 50
    assert ev["champion_idx"] == 0
    assert len(ev["ruined"]) == len(results)


def test_event_leaderboard_sorted(results):
    names = [f"agent {i:02d}" for i in range(len(results))]
    fits = [objective_value(r.stats, "timing_sortino") if not r.ruined else float("-inf")
            for r in results]
    ev = build_generation_event(0, results, fits, names, 0, 0.45,
                                max(fits), float(np.mean(fits)), 0.0)
    board_fits = [r["fitness"] for r in ev["leaderboard"]]
    assert board_fits == sorted(board_fits, reverse=True)
    assert all("weight_std" in r for r in ev["leaderboard"])


def test_event_is_picklable(results):
    """Events cross a thread queue, so they must be plain picklable data (no DataFrames)."""
    import pickle
    names = [f"agent {i:02d}" for i in range(len(results))]
    fits = [0.0 for _ in results]
    ev = build_generation_event(0, results, fits, names, 0, 0.45, 0.0, 0.0, 0.0)
    restored = pickle.loads(pickle.dumps(ev))
    assert restored["generation"] == 0
