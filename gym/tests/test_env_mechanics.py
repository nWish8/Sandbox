"""T2.5 — env mechanics: spaces, fills, costs, equity, B&H tracking."""
from __future__ import annotations

import numpy as np
import pytest

from gym import features as F
from gym.env import SignalGymEnv, DISCRETE_WEIGHTS

DAY = 86400


def _days(n, start=1_700_000_000):
    return [start + i * DAY for i in range(n)]


def _frame(write_fixture, n=40, seed=0):
    cfg, write = write_fixture
    rng = np.random.default_rng(seed)
    c = list(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
    write("TEST:AAA", _days(n), c)
    vocab = F.build_vocab(["TEST:AAA"], cfg)
    df = F.build_ticker_frame("TEST:AAA", vocab, cfg)
    feat = F.feature_columns(df, cfg)
    df[feat] = df[feat].fillna(0.0)            # mimic build_features fill
    return cfg, df, np.asarray(c)


def test_spaces(write_fixture):
    cfg, df, _ = _frame(write_fixture)
    env = SignalGymEnv(df, cfg, lookback=5)
    obs, info = env.reset()
    assert obs.shape == (5, len(env.feat_cols))
    assert env.action_space.shape == (1,)


def test_equity_matches_close_to_close_math(write_fixture):
    cfg, df, c = _frame(write_fixture)
    lb = 5
    env = SignalGymEnv(df, cfg, lookback=lb, commission_bps=10.0)
    env.reset()
    cost = 10.0 / 1e4

    # reference: weight 1.0 every step. Close-only => exact r = w*(c[i+1]/c[i]-1) - cost*|dw|
    i = lb - 1
    eq = 1.0
    bh = 1.0
    w_prev = 0.0
    done = False
    while not done:
        _, _, done, _, info = env.step(np.array([1.0], np.float32))
        r = 1.0 * (c[i + 1] / c[i] - 1.0) - cost * abs(1.0 - w_prev)
        eq *= (1 + r)
        bh *= (1 + c[i + 1] / c[i] - 1.0)
        assert np.isclose(info["equity"], eq, rtol=1e-9)
        assert np.isclose(info["bh_equity"], bh, rtol=1e-9)
        w_prev = 1.0
        i += 1


def test_cost_charged_only_on_rebalance(write_fixture):
    cfg, df, c = _frame(write_fixture)
    env = SignalGymEnv(df, cfg, lookback=5, commission_bps=50.0)
    env.reset()
    # hold flat (no cost), then enter once (cost on the 0->1 change)
    _, _, _, _, i0 = env.step(np.array([0.0], np.float32))
    assert i0["trade"] is None
    assert np.isclose(i0["ret"], 0.0)                       # flat => no return, no cost
    _, _, _, _, i1 = env.step(np.array([1.0], np.float32))
    assert i1["trade"] is not None and i1["trade"]["to"] == 1.0
    # the only cost component this step is 50bps on dw=1
    bh = i1["bh_ret"]
    assert np.isclose(i1["ret"], 1.0 * bh - 50.0 / 1e4)


def test_discrete_mode(write_fixture):
    cfg, df, _ = _frame(write_fixture)
    env = SignalGymEnv(df, cfg, lookback=5, continuous=False)
    assert env.action_space.n == len(DISCRETE_WEIGHTS)
    env.reset()
    _, _, _, _, info = env.step(2)                          # index 2 -> 0.5
    assert info["weight"] == 0.5


def test_check_env(write_fixture):
    from gymnasium.utils.env_checker import check_env
    cfg, df, _ = _frame(write_fixture)
    env = SignalGymEnv(df, cfg, lookback=5)
    check_env(env, skip_render_check=True)
