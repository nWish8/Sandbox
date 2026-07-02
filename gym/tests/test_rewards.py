"""T3 — reward registry (per-step, online, causal) + corrected portfolio_stats."""
from __future__ import annotations

import math

import numpy as np
import pytest

from rewards import (REWARDS, get_reward, reward_active, reward_active_dsr,
                     reward_diff_sharpe, reward_logret, reward_return)
from stats import portfolio_stats


class _StubEnv:
    """Minimal stand-in exposing what reward fns read."""
    def __init__(self):
        self.returns_memory = [0.0]
        self.bench_returns_memory = [0.0]
        self.reward_state: dict = {}

    def push(self, r, b=0.0):
        self.returns_memory.append(r)
        self.bench_returns_memory.append(b)


# ─────────────────────────────────────────── simple rewards

def test_return_and_logret():
    env = _StubEnv(); env.push(0.01)
    assert reward_return(env) == pytest.approx(0.01)
    assert reward_logret(env) == pytest.approx(math.log(1.01))


def test_active_is_excess_over_benchmark():
    env = _StubEnv(); env.push(0.02, b=0.013)
    assert reward_active(env) == pytest.approx(0.007)


def test_registry_has_at_least_four_and_lookup():
    assert len(REWARDS) >= 4
    assert get_reward("diff_sharpe") is reward_diff_sharpe
    with pytest.raises(KeyError):
        get_reward("nope")


# ─────────────────────────────────────────── differential Sharpe (online, causal)

def test_diff_sharpe_is_finite_and_uses_only_state():
    env = _StubEnv()
    out = []
    for r in [0.01, 0.012, 0.009, 0.011]:
        env.push(r)
        out.append(reward_diff_sharpe(env))
    assert all(np.isfinite(o) for o in out)
    assert out[0] == 0.0                       # variance undefined on the first bar → 0, not NaN


def test_diff_sharpe_rewards_lower_variance_at_equal_mean():
    """The defining DSR property: at the SAME mean return, a lower-variance (higher-Sharpe)
    stream accumulates more total differential Sharpe than a high-variance one. (Note a single
    large positive return can score negative DSR — it spikes the variance estimate — so the
    meaningful assertion is cumulative, not per-bar sign.)"""
    def total(stream):
        env = _StubEnv(); t = 0.0
        for r in stream:
            env.push(r); t += reward_diff_sharpe(env)
        return t
    low_vol = [0.011, 0.009] * 10              # mean 0.01, low variance
    high_vol = [0.06, -0.04] * 10              # mean 0.01, high variance
    assert total(low_vol) > total(high_vol)


def test_active_dsr_tracks_excess_series():
    # portfolio exactly matches benchmark → active series is all zeros → DSR stays 0
    env = _StubEnv()
    vals = []
    for r in [0.01, -0.02, 0.03, 0.005]:
        env.push(r, b=r)
        vals.append(reward_active_dsr(env))
    assert all(v == 0.0 for v in vals)         # no edge over benchmark ⇒ no reward (can't fake it)


# ─────────────────────────────────────────── corrected portfolio_stats

def test_portfolio_stats_basic_quantities():
    r = [0.02, -0.01, 0.03, 0.0, 0.015]
    s = portfolio_stats(r, periods_per_year=252)
    assert s["n_bars"] == 5
    assert s["total_return"] == pytest.approx(np.prod([1 + x for x in r]) - 1.0)
    assert np.isfinite(s["sharpe"]) and np.isfinite(s["sortino"])
    assert s["max_drawdown"] <= 0.0


def test_annualization_matches_bar_clock():
    """Sharpe must scale with √periods_per_year — the fix for √252-on-1h-bars."""
    r = [0.01, -0.005, 0.012, -0.003, 0.008, 0.002]
    s_daily = portfolio_stats(r, periods_per_year=252)
    s_quart = portfolio_stats(r, periods_per_year=63)
    assert s_daily["sharpe"] / s_quart["sharpe"] == pytest.approx(math.sqrt(252 / 63), rel=1e-9)


def test_active_sharpe_undefined_when_matching_benchmark():
    """Tracking the equal-weight benchmark exactly yields no active edge (active≡0 → not a
    finite Sharpe) — de-risking / index-hugging can't manufacture a positive score."""
    r = [0.02, -0.01, 0.03, 0.01]
    s = portfolio_stats(r, bench_returns=r, periods_per_year=252)
    assert not np.isfinite(s["active_sharpe"])
    assert s["excess_total"] == pytest.approx(0.0, abs=1e-12)


def test_active_sharpe_positive_when_beating_benchmark():
    r = [0.02, 0.01, 0.03, 0.005]
    b = [0.01, 0.005, 0.01, 0.0]               # portfolio beats benchmark every bar
    s = portfolio_stats(r, bench_returns=b, periods_per_year=252)
    assert s["active_sharpe"] > 0
    assert s["excess_total"] > 0
