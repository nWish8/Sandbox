"""runlog.py — checkpoint recording, save/load roundtrip, SAC/algo support. Synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import FinRLConfig
from portfolio import add_cov_features, train_portfolio
from runlog import RunRecord, RunRecorder, list_runs

TICS = ["AAA", "BBB", "CCC"]
TECH = ["macd", "rsi_30"]


def _cov_df(n_bars=40, seed=0, lookback=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="D")
    rows = []
    for tic in TICS:
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, n_bars))
        for d, p in zip(dates, price):
            rows.append({"date": d, "tic": tic, "close": p,
                         "macd": rng.normal(), "rsi_30": rng.uniform(20, 80)})
    return add_cov_features(pd.DataFrame(rows), lookback=lookback)


def _cfg():
    return FinRLConfig(tickers=TICS, indicators=tuple(TECH), use_vix=False,
                       use_turbulence=False)


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory):
    """One tiny recorded PPO training run, shared across tests (training is the slow bit)."""
    runs_dir = tmp_path_factory.mktemp("runs")
    df = _cov_df()
    cfg = _cfg()
    rec = RunRecorder(df, cfg, reward="logret", algo="ppo", lookback=3, every=128,
                      seed=1, timesteps=300, runs_dir=runs_dir, log=lambda *a: None)
    train_portfolio(df, cfg, reward="logret", algo="ppo", timesteps=300, seed=1,
                    lookback=3, recorder=rec, log=lambda *a: None)
    return runs_dir, rec


def test_checkpoints_recorded(recorded_run):
    _, rec = recorded_run
    # generation 0 (untrained) + intermediate marks + final state
    assert rec.steps[0] == 0
    assert len(rec.steps) >= 3
    assert list(rec.steps) == sorted(rec.steps)


def test_roundtrip_shapes_and_values(recorded_run):
    runs_dir, rec = recorded_run
    r = RunRecord.load(rec.run_id, runs_dir=runs_dir)
    C, T, N = r.n_checkpoints, r.n_bars, len(r.tics)
    assert C == len(rec.steps) and N == len(TICS)
    assert r.equity.shape == (C, T)
    assert r.weights.shape == (C, T, N)
    assert r.returns.shape == (C, T) and r.turnover.shape == (C, T)
    assert r.dates.shape == (T,) and r.bench_ret.shape == (T,)
    # equity starts at 1 and is consistent with the recorded returns
    assert np.allclose(r.equity[:, 0], 1.0)
    assert np.allclose(r.equity, np.cumprod(1.0 + r.returns, axis=1), rtol=1e-8)
    # weights are simplex points every bar of every checkpoint
    assert np.all(r.weights >= -1e-9)
    assert np.allclose(r.weights.sum(axis=2), 1.0, atol=1e-6)


def test_model_saved_and_listed(recorded_run):
    runs_dir, rec = recorded_run
    r = RunRecord.load(rec.run_id, runs_dir=runs_dir)
    assert r.model_path() is not None and r.model_path().exists()
    assert rec.run_id in list_runs(runs_dir)
    assert list_runs(runs_dir / "nonexistent") == []


def test_bench_equity_matches_returns(recorded_run):
    runs_dir, rec = recorded_run
    r = RunRecord.load(rec.run_id, runs_dir=runs_dir)
    assert np.allclose(r.bench_equity, np.cumprod(1.0 + r.bench_ret), rtol=1e-12)


def test_sac_trains_on_portfolio_env(tmp_path):
    """SAC (off-policy) must work on the same env/action space as PPO."""
    df = _cov_df()
    model = train_portfolio(df, _cfg(), reward="logret", algo="sac", timesteps=150,
                            seed=1, lookback=3, log=lambda *a: None)
    from portfolio import run_portfolio
    hist = run_portfolio(model, df, _cfg(), reward="logret", lookback=3)
    assert len(hist) == df.index.nunique()
    assert np.isfinite(hist["value"]).all()


def test_bad_algo_rejected():
    with pytest.raises(ValueError, match="algo"):
        train_portfolio(_cov_df(), _cfg(), algo="dqn", timesteps=10, lookback=3)
