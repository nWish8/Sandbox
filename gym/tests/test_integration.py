"""T6.1 — end-to-end integration: collect(mock) → build → train(sup) → analyze → validate."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gym import features as F
from gym.analysis import build_edge_report
from gym.backtest import run_backtest
from gym.baselines import BuyAndHoldPolicy
from gym.env import SignalGymEnv
from gym.train import RunConfig, training_run
from gym.valid import mcpt, runs_test

DAY = 86_400


# ─── 3-ticker synthetic universe

def _build_universe(write_fixture, n=80):
    cfg, write = write_fixture
    tickers = ["TEST:E2E1", "TEST:E2E2", "TEST:E2E3"]
    for i, ticker in enumerate(tickers):
        rng = np.random.default_rng(i * 10)
        c = list(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
        ts = [1_700_000_000 + j * DAY for j in range(n)]
        write(ticker, ts, c)

    vocab = F.build_vocab(tickers, cfg)
    dfs = []
    for ticker in tickers:
        df = F.build_ticker_frame(ticker, vocab, cfg)
        feat = F.feature_columns(df, cfg)
        df[feat] = df[feat].fillna(0.0)
        dfs.append(df)
    return cfg, dfs


# ─── integration test

def test_full_pipeline_sup(write_fixture, tmp_path):
    """Supervised: build → train → backtest → analyze → validate, on 3 synthetic tickers."""
    cfg, dfs = _build_universe(write_fixture)
    train_dfs = dfs[:2]
    val_dfs   = dfs[2:]

    # 1. Train supervised
    rc = RunConfig(agent="sup", seed=42)
    result = training_run(train_dfs, val_dfs, test_dfs=None, cfg=cfg, run_cfg=rc)

    assert result.agent == "sup"
    assert np.isfinite(result.val_result.pooled_excess_sharpe)

    # 2. Run backtest on val ticker
    policy = result.checkpoints[0].policy_snapshot
    env = SignalGymEnv(val_dfs[0], cfg)
    bt = run_backtest(env, policy)
    assert not bt.equity_curve.empty
    assert np.isfinite(bt.metrics["sharpe"])

    # 3. Analyze → edge report
    report = build_edge_report(
        result, val_dfs, cfg, out=tmp_path, ablation_features=[]
    )
    assert (tmp_path / "edge_report.json").exists()
    assert "solo_signal_edge" in report
    assert "gbt_importances" in report

    # 4. Validate (MCPT + runs test)
    excess = bt.equity_curve["excess"].to_numpy()
    rets   = bt.equity_curve["ret"].to_numpy()
    mcpt_r = mcpt(excess, n_perms=500)
    runs_r = runs_test(rets)
    assert np.isfinite(mcpt_r.p_value)
    assert np.isfinite(runs_r.p_value)


def test_baselines_are_beaten_by_flat_policy_on_falling_market(write_fixture):
    """In a monotonically falling market, BuyAndHold loses and a flat policy wins."""
    cfg, write = write_fixture
    n = 60
    c = list(100 * np.cumprod(np.full(n, 0.995)))   # steady -0.5%/bar
    ts = [1_700_000_000 + i * DAY for i in range(n)]
    write("TEST:FALL", ts, c)
    vocab = F.build_vocab(["TEST:FALL"], cfg)
    df = F.build_ticker_frame("TEST:FALL", vocab, cfg)
    feat = F.feature_columns(df, cfg)
    df[feat] = df[feat].fillna(0.0)

    # B&H
    env_bh = SignalGymEnv(df, cfg, commission_bps=0.0)
    bt_bh = run_backtest(env_bh, BuyAndHoldPolicy())

    # flat policy
    class FlatPolicy:
        def act(self, obs): return 0.0
    env_fl = SignalGymEnv(df, cfg, commission_bps=0.0)
    bt_fl = run_backtest(env_fl, FlatPolicy())

    assert bt_bh.metrics["total_return"] < 0
    assert bt_fl.metrics["total_return"] == pytest.approx(0.0, abs=1e-10)
    assert bt_fl.metrics["excess_sharpe"] > 0   # flat beats B&H


def test_determinism(write_fixture):
    """Two supervised runs with the same seed produce identical val_sharpe."""
    cfg, dfs = _build_universe(write_fixture)
    train_dfs, val_dfs = dfs[:2], dfs[2:]
    rc = RunConfig(agent="sup", seed=99)

    r1 = training_run(train_dfs, val_dfs, cfg=cfg, run_cfg=rc)
    r2 = training_run(train_dfs, val_dfs, cfg=cfg, run_cfg=rc)

    assert r1.val_result.pooled_excess_sharpe == pytest.approx(
        r2.val_result.pooled_excess_sharpe, rel=1e-6
    )
