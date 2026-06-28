"""Reward-mode reframe: absolute / absolute_dd vs excess; drawdown tracking."""
from __future__ import annotations

import numpy as np

from gym import features as F
from gym.env import SignalGymEnv

DAY = 86_400


def _env(write_fixture, closes, reward_mode="excess", dd_penalty=0.0, lookback=5):
    cfg, write = write_fixture
    cfg.reward_mode = reward_mode
    cfg.drawdown_penalty = dd_penalty
    write("TEST:RM", [1_700_000_000 + i * DAY for i in range(len(closes))], list(closes))
    vocab = F.build_vocab(["TEST:RM"], cfg)
    df = F.build_ticker_frame("TEST:RM", vocab, cfg)
    df[F.feature_columns(df, cfg)] = df[F.feature_columns(df, cfg)].fillna(0.0)
    return SignalGymEnv(df, cfg, lookback=lookback, commission_bps=0.0)


def _run(env, weight):
    env.reset()
    total = 0.0
    done = False
    while not done:
        _, r, done, _, info = env.step(np.array([weight], np.float32))
        total += r
    return total, info


def test_drawdown_tracked_in_info(write_fixture):
    closes = list(100 * np.cumprod(np.r_[np.full(10, 1.01), np.full(10, 0.97)]))
    env = _env(write_fixture, closes, reward_mode="absolute")
    env.reset()
    dds = []
    done = False
    while not done:
        _, _, done, _, info = env.step(np.array([1.0], np.float32))
        dds.append(info["drawdown"])
    assert min(dds) < 0          # a drawdown occurred during the falling leg
    assert max(dds) <= 1e-9      # drawdown is never positive


def test_absolute_mode_rewards_own_return_not_excess(write_fixture):
    """In a steadily rising market, full-long earns positive cumulative reward under
    'absolute' (own diff-Sharpe), whereas 'excess' gives ~0 (it matches B&H)."""
    closes = list(100 * np.cumprod(np.full(40, 1.005)))
    abs_total, _ = _run(_env(write_fixture, closes, reward_mode="absolute"), 1.0)
    exc_total, _ = _run(_env(write_fixture, closes, reward_mode="excess"), 1.0)
    assert abs_total > 0.0                       # own returns are positive & smooth
    assert abs(exc_total) < abs(abs_total)       # excess ~0 (full-long == B&H)


def test_absolute_dd_penalizes_drawdown(write_fixture):
    """absolute_dd with a positive penalty yields lower total reward than plain absolute
    when the equity curve spends time in drawdown (same actions, same prices)."""
    closes = list(100 * np.cumprod(np.r_[np.full(15, 1.01), np.full(15, 0.96)]))
    plain, _ = _run(_env(write_fixture, closes, reward_mode="absolute"), 1.0)
    penal, _ = _run(_env(write_fixture, closes, reward_mode="absolute_dd",
                         dd_penalty=1.0), 1.0)
    assert penal < plain                          # penalty strictly reduces reward


def test_excess_mode_unchanged_default(write_fixture):
    """Default reward_mode is 'excess' and matches the historical behavior: flat beats
    a falling market on excess."""
    cfg, _ = write_fixture
    assert cfg.reward_mode == "excess"            # default preserved
    closes = list(100 * np.cumprod(np.full(40, 0.99)))
    total, info = _run(_env(write_fixture, closes, reward_mode="excess"), 0.0)
    assert total > 0                              # staying flat beats a falling B&H
