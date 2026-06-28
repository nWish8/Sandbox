"""T3.5 — train.py: checkpoint/eval protocol, supervised path, split integrity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gym import features as F
from gym.train import RunConfig, RunResult, eval_split, training_run

DAY = 86_400


def _make_df(write_fixture, ticker="TEST:TR1", n=60, seed=1):
    cfg, write = write_fixture
    rng = np.random.default_rng(seed)
    c = list(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
    ts = [1_700_000_000 + i * DAY for i in range(n)]
    write(ticker, ts, c)
    vocab = F.build_vocab([ticker], cfg)
    df = F.build_ticker_frame(ticker, vocab, cfg)
    feat = F.feature_columns(df, cfg)
    df[feat] = df[feat].fillna(0.0)
    return cfg, df


# ─── eval_split

def test_eval_split_returns_finite(write_fixture):
    from gym.baselines import BuyAndHoldPolicy
    cfg, df = _make_df(write_fixture, n=60)
    res = eval_split(BuyAndHoldPolicy(), [df], cfg)
    assert np.isfinite(res.pooled_excess_sharpe)
    assert 0.0 <= res.beat_rate <= 1.0
    assert res.n_bars > 0


def test_eval_split_pools_multiple_tickers(write_fixture):
    from gym.baselines import BuyAndHoldPolicy
    cfg, df1 = _make_df(write_fixture, ticker="TEST:TR1", n=50, seed=1)
    _, df2   = _make_df(write_fixture, ticker="TEST:TR2", n=50, seed=2)
    res = eval_split(BuyAndHoldPolicy(), [df1, df2], cfg)
    assert res.n_bars == pytest.approx((50 - cfg.lookback - 1) * 2, abs=4)


# ─── supervised training run

def test_sup_run_returns_run_result(write_fixture):
    cfg, df = _make_df(write_fixture, n=80)
    rc = RunConfig(agent="sup", seed=0)
    result = training_run([df], [df], cfg=cfg, run_cfg=rc)
    assert isinstance(result, RunResult)
    assert result.agent == "sup"
    assert len(result.checkpoints) == 1


def test_sup_run_val_result_is_finite(write_fixture):
    cfg, df = _make_df(write_fixture, n=80)
    rc = RunConfig(agent="sup", seed=0)
    result = training_run([df], [df], cfg=cfg, run_cfg=rc)
    assert np.isfinite(result.val_result.pooled_excess_sharpe)


def test_sup_run_test_result_when_provided(write_fixture):
    cfg, df = _make_df(write_fixture, n=80)
    rc = RunConfig(agent="sup", seed=0)
    result = training_run([df], [df], [df], cfg=cfg, run_cfg=rc)
    assert result.test_result is not None
    assert np.isfinite(result.test_result.pooled_excess_sharpe)


def test_sup_run_test_is_none_when_not_provided(write_fixture):
    cfg, df = _make_df(write_fixture, n=80)
    rc = RunConfig(agent="sup", seed=0)
    result = training_run([df], [df], cfg=cfg, run_cfg=rc)
    assert result.test_result is None


def test_training_run_raises_on_empty_train(write_fixture):
    cfg, df = _make_df(write_fixture, n=80)
    with pytest.raises(ValueError, match="empty"):
        training_run([], [df])


def test_unknown_agent_raises(write_fixture):
    cfg, df = _make_df(write_fixture, n=80)
    with pytest.raises(ValueError, match="Unknown agent"):
        training_run([df], [df], run_cfg=RunConfig(agent="bad"))
