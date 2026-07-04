"""T6 — out-of-sample significance gate (sign-flip MCPT, block MCPT, runs test)."""
from __future__ import annotations

import numpy as np

from signif import default_block_len, gate, mcpt_sharpe, runs_test


def test_block_mcpt_single_block_is_coin_flip():
    """With one block the only permutations are ±the whole series, so any positive-Sharpe
    series gets p ≈ 0.5 — the mechanics sanity check."""
    r = np.linspace(0.005, 0.015, 40)
    _, p = mcpt_sharpe(r, n_perms=4000, seed=0, block_len=len(r))
    assert 0.40 < p < 0.60


def test_block_mcpt_matches_iid_at_block_one():
    rng = np.random.default_rng(3)
    r = rng.normal(0.002, 0.01, 120)
    _, p_iid = mcpt_sharpe(r, n_perms=4000, seed=7)
    _, p_blk = mcpt_sharpe(r, n_perms=4000, seed=7, block_len=1)
    assert p_blk == p_iid                              # same code path, same draws


def test_block_mcpt_is_harder_on_streaky_series():
    """A streaky series (long same-sign runs) fakes a decent Sharpe that per-bar flips
    call significant; block flips keep the streak structure in the null and don't."""
    r = np.concatenate([np.full(20, 0.004), np.full(10, -0.003)] * 4)
    _, p_iid = mcpt_sharpe(r, n_perms=3000, seed=1)
    _, p_blk = mcpt_sharpe(r, n_perms=3000, seed=1, block_len=20)
    assert p_blk > p_iid                               # dependence-robust null is stricter


def test_gate_reports_block_p_and_requires_it():
    r = np.linspace(0.005, 0.015, 60)                  # genuinely consistent edge
    g = gate(r, n_perms=2000, seed=0)
    assert np.isfinite(g.block_mcpt_p)
    assert "block p=" in g.summary()
    assert default_block_len(len(r)) >= 5


def test_mcpt_flags_strong_consistent_edge():
    r = np.linspace(0.005, 0.015, 40)             # all-positive, varied → very high Sharpe
    obs, p = mcpt_sharpe(r, n_perms=2000, seed=0)
    assert obs > 0
    assert p < 0.05                                # a real directional edge


def test_mcpt_finds_no_edge_in_symmetric_series():
    r = np.array([0.01, -0.01] * 30)               # zero mean → Sharpe ~ 0
    obs, p = mcpt_sharpe(r, n_perms=2000, seed=0)
    assert abs(obs) < 1e-9
    assert p > 0.05                                # not significant


def test_runs_test_detects_alternating_dependence():
    r = np.array([0.01, -0.01] * 25)               # perfectly alternating → max runs
    z, p = runs_test(r)
    assert np.isfinite(z) and p < 0.05             # serial dependence detected


def test_runs_test_undefined_for_one_sided():
    z, p = runs_test(np.array([0.01, 0.02, 0.03]))  # no negatives → undefined
    assert np.isnan(z) and np.isnan(p)


def test_gate_bundles_verdict():
    r = np.linspace(0.004, 0.016, 50)
    g = gate(r, n_perms=2000, seed=0)
    assert g.n == 50
    assert g.edge_significant is True
    assert "MCPT p=" in g.summary()


def test_gate_negative_series_not_significant():
    r = np.array([0.01, -0.01] * 30)
    g = gate(r, n_perms=2000, seed=0)
    assert g.edge_significant is False
