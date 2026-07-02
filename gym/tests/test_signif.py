"""T6 — out-of-sample significance gate (sign-flip MCPT + Wald-Wolfowitz runs test)."""
from __future__ import annotations

import numpy as np

from signif import gate, mcpt_sharpe, runs_test


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
