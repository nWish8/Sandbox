"""rule_policies.py — baseline rules are causal simplex policies; the market-permutation
MCPT flags structure-exploiting rules on trending tape and not on shuffled noise."""
from __future__ import annotations

import numpy as np
import pytest

from rule_policies import (RULES, inverse_vol, market_permutation_pvalue, momentum_topk,
                           permute_market)


def _panel(T=300, N=4, seed=0, drift=None):
    rng = np.random.default_rng(seed)
    drift = drift if drift is not None else np.zeros(N)
    rets = rng.normal(0.0005, 0.01, (T - 1, N)) + drift
    return np.vstack([np.full(N, 100.0), 100.0 * np.exp(np.cumsum(rets, axis=0))])


def test_rules_emit_valid_simplex_weights():
    closes = _panel()
    for name, fn in RULES.items():
        w = fn(closes)
        assert w.shape == closes.shape, name
        assert (w >= -1e-12).all(), name
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9, err_msg=name)


def test_rules_are_causal():
    closes = _panel()
    cut = 200
    mutated = closes.copy()
    mutated[cut:] *= 3.0                              # rewrite the future
    for name, fn in RULES.items():
        np.testing.assert_allclose(fn(closes)[:cut], fn(mutated)[:cut],
                                   err_msg=f"{name} looked ahead")


def test_momentum_concentrates_on_winners():
    # assets 0,1 trend up hard; 2,3 flat → top-2 momentum should hold only 0 and 1
    closes = _panel(drift=np.array([0.004, 0.003, 0.0, 0.0]), seed=1)
    w = momentum_topk(closes, k=2, lookback=60, rebalance=21)
    late = w[150:]
    assert late[:, :2].sum(axis=1).mean() > 0.95
    np.testing.assert_allclose(late.sum(axis=1), 1.0, atol=1e-9)


def test_inverse_vol_prefers_the_quiet_asset():
    rng = np.random.default_rng(2)
    T, N = 250, 2
    rets = np.column_stack([rng.normal(0, 0.005, T - 1),      # quiet
                            rng.normal(0, 0.02, T - 1)])      # 4x noisier
    closes = np.vstack([np.full(N, 100.0), 100.0 * np.exp(np.cumsum(rets, axis=0))])
    w = inverse_vol(closes, window=60, rebalance=21)
    assert w[150:, 0].mean() > 0.7                            # quiet asset dominates


def test_permute_market_preserves_marginals_destroys_order():
    closes = _panel(T=120, N=3, seed=3)
    rng = np.random.default_rng(0)
    perm = permute_market(closes, rng)
    assert perm.shape == closes.shape
    np.testing.assert_allclose(perm[0], closes[0])            # anchored first bar
    # same multiset of panel log-returns (rows reordered, columns intact)
    lr = np.sort(np.log(closes[1:] / closes[:-1]), axis=0)
    lp = np.sort(np.log(perm[1:] / perm[:-1]), axis=0)
    np.testing.assert_allclose(lr, lp, atol=1e-12)
    assert not np.allclose(perm, closes)                      # order actually changed


def test_market_mcpt_detects_serial_structure():
    """On a market with persistent per-asset trends momentum exploits real serial
    structure → shuffling the bars kills it → small p. On iid noise → unremarkable p."""
    trending = _panel(T=400, N=4, seed=4, drift=np.array([0.004, -0.002, 0.0, 0.001]))
    res_t = market_permutation_pvalue(lambda c: momentum_topk(c, k=1, lookback=40,
                                                              rebalance=10),
                                      trending, n_perms=99, seed=0)
    assert res_t["p_value"] < 0.10

    noise = _panel(T=400, N=4, seed=5)
    res_n = market_permutation_pvalue(lambda c: momentum_topk(c, k=1, lookback=40,
                                                              rebalance=10),
                                      noise, n_perms=99, seed=0)
    assert res_n["p_value"] > 0.10
