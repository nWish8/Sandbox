"""T3.2 — backtest: run_backtest and fast_vectorized produce identical equity."""
from __future__ import annotations

import numpy as np
import pytest

from gym import features as F
from gym.backtest import BacktestResult, fast_vectorized, run_backtest
from gym.baselines import BuyAndHoldPolicy
from gym.env import SignalGymEnv

DAY = 86_400


def _setup(write_fixture, n=50, seed=7):
    cfg, write = write_fixture
    rng = np.random.default_rng(seed)
    c = list(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
    ts = [1_700_000_000 + i * DAY for i in range(n)]
    write("TEST:BT", ts, c)
    vocab = F.build_vocab(["TEST:BT"], cfg)
    df = F.build_ticker_frame("TEST:BT", vocab, cfg)
    feat = F.feature_columns(df, cfg)
    df[feat] = df[feat].fillna(0.0)
    return cfg, df


# ─── BacktestResult structure

def test_result_has_required_fields(write_fixture):
    cfg, df = _setup(write_fixture)
    env = SignalGymEnv(df, cfg, lookback=5)
    res = run_backtest(env, BuyAndHoldPolicy())
    assert isinstance(res, BacktestResult)
    assert not res.equity_curve.empty
    for col in ("timestamp", "equity", "bh_equity", "weight", "ret", "bh_ret", "excess"):
        assert col in res.equity_curve.columns, col
    for key in ("total_return", "bh_return", "sharpe", "max_drawdown", "beat_rate"):
        assert key in res.metrics, key


# ─── run_backtest equity math

def test_bah_equity_equals_bh_equity(write_fixture):
    """BuyAndHoldPolicy should produce equity == bh_equity (same weight=1 every bar)."""
    cfg, df = _setup(write_fixture)
    env = SignalGymEnv(df, cfg, lookback=5, commission_bps=0.0)
    res = run_backtest(env, BuyAndHoldPolicy())
    ec = res.equity_curve
    # with zero cost and weight=1, equity == bh_equity exactly
    np.testing.assert_allclose(ec["equity"].values, ec["bh_equity"].values, rtol=1e-9)


def test_flat_policy_equity_stays_one(write_fixture):
    class FlatPolicy:
        def act(self, obs):
            return 0.0
    cfg, df = _setup(write_fixture)
    env = SignalGymEnv(df, cfg, lookback=5, commission_bps=0.0)
    res = run_backtest(env, FlatPolicy())
    np.testing.assert_allclose(res.equity_curve["equity"].values, 1.0, atol=1e-12)


# ─── fast_vectorized vs env loop

def test_fast_vectorized_matches_env_loop(write_fixture):
    """Weights replayed through fast_vectorized must match env equity curve exactly."""
    cfg, df = _setup(write_fixture, n=60)
    env = SignalGymEnv(df, cfg, lookback=5, commission_bps=10.0)

    # record weights from a step-through (random weights)
    rng = np.random.default_rng(42)
    obs, _ = env.reset()
    weights_used = []
    env_equity = []
    done = False
    while not done:
        w = float(rng.uniform(0, 1))
        weights_used.append(w)
        obs, _, done, _, info = env.step(np.array([w], dtype=np.float32))
        env_equity.append(info["equity"])

    # pad weights array to full df length (first lookback-1 bars are skipped by env)
    n = len(df)
    lb = 5
    full_weights = np.zeros(n)
    full_weights[lb - 1: lb - 1 + len(weights_used)] = weights_used

    fv = fast_vectorized(full_weights, df, cfg)
    fv_equity = fv.equity_curve["equity"].values

    # fast_vectorized starts from bar 0 (not bar lb-1), so trim to compare
    # env starts at i = lb-1 and records equity after each step; fv starts at bar 0.
    # align: fv bar indices 0..(n-2); env bar indices lb-1..(n-2)
    fv_slice = fv_equity[lb - 1:]

    np.testing.assert_allclose(fv_slice, env_equity, rtol=1e-7,
                               err_msg="fast_vectorized must match env equity exactly")


# ─── metrics sanity

def test_metrics_finite(write_fixture):
    cfg, df = _setup(write_fixture)
    env = SignalGymEnv(df, cfg, lookback=5)
    res = run_backtest(env, BuyAndHoldPolicy())
    for k, v in res.metrics.items():
        assert np.isfinite(v), f"metric {k!r} is not finite: {v}"


def test_beat_rate_in_unit_interval(write_fixture):
    cfg, df = _setup(write_fixture)
    env = SignalGymEnv(df, cfg, lookback=5)
    res = run_backtest(env, BuyAndHoldPolicy())
    br = res.metrics["beat_rate"]
    assert 0.0 <= br <= 1.0
