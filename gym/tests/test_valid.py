"""T3.6 — valid.py: MCPT p≈0.5 for random, p≈0 for strong edge; runs test IID non-sig."""
from __future__ import annotations

import numpy as np
import pytest

from gym.valid import MCPTResult, RunsTestResult, mcpt, runs_test


# ─── MCPT

def test_mcpt_random_excess_p_near_half():
    """Exactly zero-mean excess (equal +/- values) → p-value near 0.5 (no edge)."""
    # Construct exactly zero-mean data: 500 positive and 500 negative returns.
    # Sharpe = 0; sign-flip permutations should beat the observed ~50% of the time.
    n = 500
    excess = np.concatenate([np.full(n, 0.01), np.full(n, -0.01)])
    np.random.default_rng(0).shuffle(excess)
    result = mcpt(excess, n_perms=2000, seed=0)
    assert isinstance(result, MCPTResult)
    # observed Sharpe ≈ 0; p should be near 0.5 (no edge)
    assert result.p_value > 0.25, f"Expected p>0.25 for zero-mean data, got {result.p_value:.4f}"
    assert not result.significant


def test_mcpt_strong_positive_edge_p_near_zero():
    """Consistently positive excess → observed Sharpe hard to beat → p near 0."""
    rng = np.random.default_rng(1)
    excess = 0.02 + rng.normal(0, 0.001, size=500)  # strongly positive
    result = mcpt(excess, n_perms=2000, seed=1)
    assert result.p_value < 0.05
    assert result.significant


def test_mcpt_returns_finite_sharpe():
    rng = np.random.default_rng(2)
    excess = rng.normal(0.001, 0.01, 200)
    result = mcpt(excess, n_perms=500)
    assert np.isfinite(result.observed_sharpe)
    assert np.isfinite(result.p_value)
    assert 0.0 <= result.p_value <= 1.0


def test_mcpt_short_series_returns_nan():
    result = mcpt(np.array([0.01]))
    assert not result.significant


# ─── Runs test

def test_runs_test_iid_non_significant():
    """IID returns → runs test should NOT reject H0 (no serial dependence)."""
    rng = np.random.default_rng(3)
    rets = rng.normal(0, 0.01, 200)
    result = runs_test(rets)
    assert isinstance(result, RunsTestResult)
    # with true IID data, p should typically be > 0.05 (non-significant)
    # we use a generous threshold since this is a finite-sample test
    assert result.p_value > 0.01, f"IID data unexpectedly significant: p={result.p_value:.4f}"
    assert not result.significant


def test_runs_test_alternating_series_significant():
    """Perfectly alternating +/- series has too few runs → should reject H0."""
    n = 200
    rets = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
    result = runs_test(rets)
    assert result.significant, f"Alternating series should be significant, p={result.p_value:.4f}"


def test_runs_test_short_series_returns_gracefully():
    result = runs_test(np.array([0.01, -0.01, 0.01]))
    assert isinstance(result, RunsTestResult)


def test_runs_test_all_positive_degenerate():
    result = runs_test(np.ones(50) * 0.01)
    # all same sign → degenerate; should not raise
    assert isinstance(result, RunsTestResult)
