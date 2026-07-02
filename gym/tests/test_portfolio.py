"""T2 — multi-asset PortfolioEnv: causal returns, softmax weights, turnover cost, and the
no-lookahead covariance data-prep. Uses synthetic data only (no network / no FinRL download)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio import PortfolioEnv, add_cov_features

TICS = ["AAA", "BBB", "CCC"]
TECH = ["macd", "rsi_30"]


def _raw_df(n_bars=12, seed=0):
    """Synthetic FinRL long df: date, tic, close, + two tech cols. Deterministic."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="D")
    rows = []
    for ti, tic in enumerate(TICS):
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, n_bars))
        for d, p in zip(dates, price):
            rows.append({"date": d, "tic": tic, "close": p,
                         "macd": rng.normal(), "rsi_30": rng.uniform(20, 80)})
    return pd.DataFrame(rows)


def _env(cost_pct=0.0, lookback=3, **kw):
    df = add_cov_features(_raw_df(), lookback=lookback)
    stock_dim = df.tic.nunique()
    return PortfolioEnv(df, stock_dim=stock_dim, tech_indicator_list=TECH,
                        cost_pct=cost_pct, lookback=lookback, **kw), df


# ─────────────────────────────────────────── covariance data-prep

def test_add_cov_is_causal_no_lookahead():
    """cov at bar i uses only bars < i, so perturbing the LAST close cannot change any
    earlier bar's covariance."""
    raw = _raw_df()
    base = add_cov_features(raw, lookback=3)
    perturbed = raw.copy()
    last_date = perturbed["date"].max()
    perturbed.loc[perturbed["date"] == last_date, "close"] *= 1.5     # shock the final bar
    pert = add_cov_features(perturbed, lookback=3)

    # all bars except the final one must have identical covariance matrices
    n_days = base.index.nunique()
    for i in range(n_days - 1):
        c_base = base.loc[i, "cov_list"].values[0]
        c_pert = pert.loc[i, "cov_list"].values[0]
        assert np.allclose(c_base, c_pert), f"cov at bar {i} changed by a future close — lookahead!"


def test_add_cov_drops_warmup():
    raw = _raw_df(n_bars=12)
    out = add_cov_features(raw, lookback=3)
    assert out.index.nunique() == 12 - 3          # first `lookback` bars dropped


# ─────────────────────────────────────────── env mechanics

def test_obs_shape_and_reset():
    env, _ = _env()
    state, _ = env.reset()
    assert state.shape == (len(TICS) + len(TECH), len(TICS))    # (cov rows + tech) × N


def test_weights_are_a_simplex_point():
    env, _ = _env()
    env.reset()
    env.step(np.array([2.0, -1.0, 0.5]))
    w = env.weights_memory[-1]
    assert w.shape == (len(TICS),)
    assert np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-9         # long-only, sums to 1


def test_step_return_is_causal_and_matches_formula():
    """The weight chosen at bar t earns exactly the t→t+1 basket return (no cost case)."""
    env, df = _env(cost_pct=0.0)
    env.reset()
    day0 = df.loc[0]
    actions = np.array([2.0, 0.0, 0.0])
    w = env.softmax_normalization(actions)
    env.step(actions)
    day1 = df.loc[1]
    asset_ret = day1.close.values / day0.close.values - 1.0
    expected = float(np.sum(asset_ret * w))
    assert env.returns_memory[-1] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_turnover_cost_reduces_return():
    actions = np.array([3.0, 0.0, 0.0])           # away from the equal-weight start → turnover
    free, df = _env(cost_pct=0.0)
    free.reset(); free.step(actions)
    costed, _ = _env(cost_pct=0.01)
    costed.reset(); costed.step(actions)
    turnover = costed.turnover_memory[-1]
    assert turnover > 0
    assert costed.returns_memory[-1] == pytest.approx(free.returns_memory[-1] - 0.01 * turnover,
                                                      rel=1e-9, abs=1e-12)


def test_history_export_runs_full_episode():
    env, df = _env(cost_pct=0.001)
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(np.zeros(len(TICS)))    # equal-weight every bar
    hist = env.portfolio_history()
    assert len(hist) == df.index.nunique()                   # one row per bar (incl. reset row)
    assert hist["ret"].iloc[0] == 0.0
    assert {"date", "ret", "bench_ret", "value", "turnover"} <= set(hist.columns)
