"""Tests for stats.py — backtesting.py-style metrics + objective registry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gym.stats import (
    DEFAULT_OBJECTIVE,
    OBJECTIVES,
    compute_stats,
    holding_period_returns,
    objective_value,
)


def _ec(equity, weight, ret, bh_equity=None, bh_ret=None):
    """Build an equity-curve DataFrame like backtest/population produce."""
    n = len(equity)
    d = {"equity": equity, "weight": weight, "ret": ret}
    if bh_equity is not None:
        d["bh_equity"] = bh_equity
    if bh_ret is not None:
        d["bh_ret"] = bh_ret
    return pd.DataFrame(d, index=range(n))


# ─────────────────────────────────── holding-period trade derivation

def test_holding_periods_single_run():
    # weight on for bars 1..3, off elsewhere -> one trade
    w = np.array([0, 1, 1, 1, 0, 0])
    r = np.array([0.0, 0.10, -0.05, 0.02, 0.0, 0.0])
    tr = holding_period_returns(w, r)
    assert len(tr) == 1
    np.testing.assert_allclose(tr[0], (1.10 * 0.95 * 1.02) - 1.0, rtol=1e-9)


def test_holding_periods_multiple_runs():
    w = np.array([1, 0, 1, 1, 0, 1])
    r = np.array([0.05, 0.0, 0.01, 0.01, 0.0, -0.03])
    tr = holding_period_returns(w, r)
    assert len(tr) == 3            # three separate nonzero-exposure runs
    np.testing.assert_allclose(tr[0], 0.05, rtol=1e-9)
    np.testing.assert_allclose(tr[2], -0.03, rtol=1e-9)


def test_holding_periods_none_when_flat():
    w = np.zeros(5)
    r = np.zeros(5)
    assert len(holding_period_returns(w, r)) == 0


# ─────────────────────────────────── core stats

def test_total_return_and_final():
    eq = np.array([1.0, 1.1, 1.05, 1.2])
    ec = _ec(eq, weight=[1, 1, 1, 1], ret=[0.0, 0.10, -0.0454, 0.1428])
    s = compute_stats(ec)
    assert s["equity_final"] == pytest.approx(1.2)
    assert s["total_return"] == pytest.approx(0.2)
    assert s["equity_peak"] == pytest.approx(1.2)


def test_max_drawdown_is_negative_and_correct():
    # peak 1.2 then down to 0.9 -> dd = 0.9/1.2 - 1 = -0.25
    eq = np.array([1.0, 1.2, 0.9, 1.0])
    ec = _ec(eq, weight=[1, 1, 1, 1], ret=[0.0, 0.2, -0.25, 0.111])
    s = compute_stats(ec)
    assert s["max_drawdown"] == pytest.approx(-0.25, rel=1e-6)
    assert s["max_drawdown"] <= 0.0


def test_sharpe_sign_follows_mean_return():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 200)          # positive drift
    eq = np.cumprod(1 + r)
    ec = _ec(eq, weight=np.ones(200), ret=r)
    s = compute_stats(ec)
    assert s["sharpe"] > 0
    assert np.isfinite(s["sortino"])


def test_exposure_pct():
    w = np.array([0, 1, 1, 0, 1])            # 3 of 5 bars in market
    ec = _ec(np.ones(5), weight=w, ret=np.zeros(5))
    s = compute_stats(ec)
    assert s["exposure_pct"] == pytest.approx(60.0)


def test_win_rate_and_profit_factor():
    # two winning trades, one losing
    w = np.array([1, 0, 1, 0, 1])
    r = np.array([0.10, 0.0, -0.05, 0.0, 0.20])
    eq = np.cumprod(1 + r)
    ec = _ec(eq, weight=w, ret=r)
    s = compute_stats(ec)
    assert s["n_trades"] == 3
    assert s["win_rate"] == pytest.approx(2 / 3 * 100)
    assert s["profit_factor"] == pytest.approx((0.10 + 0.20) / 0.05, rel=1e-6)
    assert s["best_trade"] == pytest.approx(0.20)
    assert s["worst_trade"] == pytest.approx(-0.05)


def test_empty_curve_returns_empty_dict():
    assert compute_stats(pd.DataFrame()) == {}


# ─────────────────────────────────── objective registry

def test_default_objective_registered():
    assert DEFAULT_OBJECTIVE in OBJECTIVES


def test_objective_value_nan_is_worst():
    s = {"sortino": float("nan"), "n_trades": 100}
    assert objective_value(s, "sortino") == float("-inf")


def test_trade_objective_guarded_by_min_trades():
    # great profit factor but only 1 trade -> demoted to -inf
    s = {"profit_factor": 5.0, "n_trades": 1}
    assert objective_value(s, "profit_factor", min_trades=3) == float("-inf")
    s2 = {"profit_factor": 5.0, "n_trades": 10}
    assert objective_value(s2, "profit_factor", min_trades=3) == 5.0


def test_unknown_objective_raises():
    with pytest.raises(KeyError):
        objective_value({"sortino": 1.0}, "nonexistent")


def test_higher_is_better_convention():
    good = {"sortino": 2.0, "n_trades": 50}
    bad = {"sortino": -1.0, "n_trades": 50}
    assert objective_value(good, "sortino") > objective_value(bad, "sortino")


# ─────────────────────────────────── timing objective (vs matched-constant twin)

def _market(n=200, seed=0):
    """A market return series with mixed up/down bars."""
    return np.random.default_rng(seed).normal(0.0005, 0.012, n)


def test_constant_exposure_scores_zero_timing():
    """A constant-weight agent IS its own twin -> timing fitness exactly 0, not -inf."""
    bh_r = _market()
    w = np.full(len(bh_r), 0.5)
    ret = w * bh_r                                   # constant 0.5 exposure
    eq = np.cumprod(1 + ret)
    ec = _ec(eq, weight=w, ret=ret, bh_equity=np.cumprod(1 + bh_r), bh_ret=bh_r)
    s = compute_stats(ec)
    assert s["timing_sortino"] == pytest.approx(0.0, abs=1e-9)
    assert s["weight_std"] == pytest.approx(0.0)
    assert objective_value(s, "timing_sortino") == pytest.approx(0.0, abs=1e-9)


def test_good_timer_scores_positive():
    """Long on up bars, flat on down bars -> beats its average-exposure twin."""
    bh_r = _market(seed=1)
    w = (bh_r > 0).astype(float)                     # perfect (cheating) timing
    ret = w * bh_r
    eq = np.cumprod(1 + ret)
    ec = _ec(eq, weight=w, ret=ret, bh_equity=np.cumprod(1 + bh_r), bh_ret=bh_r)
    s = compute_stats(ec)
    assert s["timing_sortino"] > 0
    assert objective_value(s, "timing_sortino") > 0


def test_bad_timer_scores_negative():
    """Long on down bars, flat on up bars -> worse than its twin."""
    bh_r = _market(seed=2)
    w = (bh_r < 0).astype(float)                     # anti-timing
    ret = w * bh_r
    eq = np.cumprod(1 + ret)
    ec = _ec(eq, weight=w, ret=ret, bh_equity=np.cumprod(1 + bh_r), bh_ret=bh_r)
    s = compute_stats(ec)
    assert s["timing_sortino"] < 0


def test_timing_objective_beats_constant_for_timer():
    bh_r = _market(seed=3)
    w_t = (bh_r > 0).astype(float)
    w_c = np.full(len(bh_r), 0.5)
    timer = compute_stats(_ec(np.cumprod(1 + w_t * bh_r), w_t, w_t * bh_r,
                              bh_equity=np.cumprod(1 + bh_r), bh_ret=bh_r))
    const = compute_stats(_ec(np.cumprod(1 + w_c * bh_r), w_c, w_c * bh_r,
                              bh_equity=np.cumprod(1 + bh_r), bh_ret=bh_r))
    assert objective_value(timer, "timing_sortino") > objective_value(const, "timing_sortino")
