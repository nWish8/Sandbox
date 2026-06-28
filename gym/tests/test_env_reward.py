"""T2.6 — differential-Sharpe excess reward: warmup, sign, numerical safety."""
from __future__ import annotations

import numpy as np

from gym import features as F
from gym.env import SignalGymEnv

DAY = 86400


def _days(n, start=1_700_000_000):
    return [start + i * DAY for i in range(n)]


def _env_from_closes(write_fixture, closes, lookback=5):
    cfg, write = write_fixture
    write("TEST:AAA", _days(len(closes)), list(closes))
    vocab = F.build_vocab(["TEST:AAA"], cfg)
    df = F.build_ticker_frame("TEST:AAA", vocab, cfg)
    feat = F.feature_columns(df, cfg)
    df[feat] = df[feat].fillna(0.0)
    return SignalGymEnv(df, cfg, lookback=lookback, commission_bps=0.0)


def test_reward_zero_during_warmup_and_finite(write_fixture):
    closes = list(100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 40)))
    env = _env_from_closes(write_fixture, closes, lookback=6)
    env.reset()
    rewards = []
    done = False
    while not done:
        _, r, done, _, _ = env.step(np.array([1.0], np.float32))
        rewards.append(r)
    assert all(r == 0.0 for r in rewards[:6])               # warmup == lookback steps
    assert all(np.isfinite(r) for r in rewards)             # no div-by-zero / NaN


def test_flat_beats_falling_market(write_fixture):
    # steadily falling price: staying flat (w=0) avoids the loss -> positive excess
    closes = list(100 * np.cumprod(np.full(40, 1 - 0.01)))
    env = _env_from_closes(write_fixture, closes, lookback=5)
    env.reset()
    total = 0.0
    done = False
    while not done:
        _, r, done, _, info = env.step(np.array([0.0], np.float32))
        total += r
    assert total > 0
    assert env.equity == 1.0                                # flat -> equity unchanged
    assert env.bh_equity < 1.0                              # B&H lost money


def test_flat_lags_rising_market(write_fixture):
    closes = list(100 * np.cumprod(np.full(40, 1 + 0.01)))
    env = _env_from_closes(write_fixture, closes, lookback=5)
    env.reset()
    total = 0.0
    done = False
    while not done:
        _, r, done, _, _ = env.step(np.array([0.0], np.float32))
        total += r
    assert total < 0                                        # missing the rally underperforms B&H
