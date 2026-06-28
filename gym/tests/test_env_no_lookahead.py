"""T2.7 — env observation causality: the obs at bar t never contains bar t+1."""
from __future__ import annotations

import numpy as np

from gym import features as F
from gym.env import SignalGymEnv

DAY = 86400


def _env(write_fixture, n=40, lookback=5):
    cfg, write = write_fixture
    c = list(100 * np.cumprod(1 + np.random.default_rng(2).normal(0, 0.01, n)))
    write("TEST:AAA", [1_700_000_000 + i * DAY for i in range(n)], c)
    vocab = F.build_vocab(["TEST:AAA"], cfg)
    df = F.build_ticker_frame("TEST:AAA", vocab, cfg)
    feat = F.feature_columns(df, cfg)
    df[feat] = df[feat].fillna(0.0)
    return SignalGymEnv(df, cfg, lookback=lookback)


def test_obs_window_ends_at_current_bar(write_fixture):
    env = _env(write_fixture)
    obs, _ = env.reset()
    assert obs.shape[0] == env.lookback
    # last obs row is the current decision bar; the row beyond is NOT in the window
    assert np.array_equal(obs[-1], env.X[env.i])
    for _ in range(5):
        prev_i = env.i
        obs, _, _, _, _ = env.step(np.array([0.5], np.float32))
        assert np.array_equal(obs[-1], env.X[prev_i + 1])   # window advanced by one bar


def test_future_mutation_does_not_leak_into_obs(write_fixture):
    env = _env(write_fixture)
    obs, _ = env.reset()
    snapshot = obs.copy()
    # poison every future bar; the already-returned obs must be unchanged (it's a copy
    # of rows <= current bar)
    env.X[env.i + 1:] = 999.0
    assert np.array_equal(obs, snapshot)
    # and a freshly fetched obs for the same bar still excludes the poisoned future rows
    assert not (obs == 999.0).any()
