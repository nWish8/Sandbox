"""Tests for evo.py — EvoPolicy MLP + the generational ES loop."""
from __future__ import annotations

import numpy as np
import pytest

from gym import features as F
from gym.evo import (
    EvoConfig,
    EvoPolicy,
    EvoResult,
    evolve,
    load_champion,
    save_champion,
)

DAY = 86_400


def _make_ticker(write, symbol, closes):
    ts = [1_700_000_000 + i * DAY for i in range(len(closes))]
    write(symbol, ts, list(closes))


def _build(cfg, write, symbols, seed=0):
    cfg.lookback = 5
    rng = np.random.default_rng(seed)
    for s in symbols:
        _make_ticker(write, s, 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, 60)))
    vocab = F.build_vocab(symbols, cfg)
    dfs = []
    for s in symbols:
        df = F.build_ticker_frame(s, vocab, cfg)
        feat = F.feature_columns(df, cfg)
        df[feat] = df[feat].fillna(0.0)
        dfs.append(df)
    return dfs


@pytest.fixture
def universe(write_fixture):
    cfg, write = write_fixture
    train = _build(cfg, write, ["TR:A", "TR:B", "TR:C"], seed=1)
    val = _build(cfg, write, ["VA:A", "VA:B"], seed=2)
    test = _build(cfg, write, ["TE:A"], seed=3)
    return cfg, train, val, test


# ─────────────────────────────────── EvoPolicy

def test_policy_param_count_last_mode():
    p = EvoPolicy(n_feat=10, hidden=4, window_mode="last")
    # W1(10*4) + b1(4) + W2(4*1) + b2(1)
    assert p.n_params == 10 * 4 + 4 + 4 + 1
    assert p.n_in == 10


def test_policy_param_count_flatten_mode():
    p = EvoPolicy(n_feat=10, hidden=4, window_mode="flatten", lookback=3)
    assert p.n_in == 30
    assert p.n_params == 30 * 4 + 4 + 4 + 1


def test_act_in_unit_interval():
    rng = np.random.default_rng(0)
    p = EvoPolicy(n_feat=8, hidden=4, rng=rng)
    obs = rng.normal(0, 1, (5, 8))
    for _ in range(20):
        w = p.act(rng.normal(0, 3, (5, 8)))
        assert 0.0 <= w <= 1.0


def test_genome_roundtrip_is_deterministic():
    p = EvoPolicy(n_feat=6, hidden=3)
    g = p.genome.copy()
    obs = np.random.default_rng(1).normal(0, 1, (5, 6))
    w1 = p.act(obs)
    p.set_genome(g)                         # same genome -> same output
    assert p.act(obs) == pytest.approx(w1)


def test_different_genomes_differ():
    obs = np.random.default_rng(2).normal(0, 1, (5, 6))
    a = EvoPolicy(n_feat=6, hidden=3, rng=np.random.default_rng(10))
    b = EvoPolicy(n_feat=6, hidden=3, rng=np.random.default_rng(11))
    assert a.act(obs) != b.act(obs)


def test_from_genome_rejects_wrong_size():
    p = EvoPolicy(n_feat=6, hidden=3)
    with pytest.raises(ValueError):
        p.set_genome(np.zeros(p.n_params + 1))


# ─────────────────────────────────── evolve loop

def test_evolve_returns_structured_result(universe):
    cfg, train, val, test = universe
    evo_cfg = EvoConfig(pop_size=8, n_generations=3, hidden=4,
                        early_stop_patience=10, seed=7)
    res = evolve(train, val, test, cfg, evo_cfg)
    assert isinstance(res, EvoResult)
    assert len(res.generations) >= 1
    assert 0 <= res.best_generation_idx < len(res.generations)
    assert res.champion_genome.size == EvoPolicy(
        res.n_feat, hidden=4, window_mode=res.window_mode, lookback=res.lookback
    ).n_params
    assert res.test_stats is not None
    # every generation recorded its population survival count
    for gr in res.generations:
        assert 0 <= gr.n_survivors <= evo_cfg.pop_size


def test_evolve_is_reproducible(universe):
    cfg, train, val, test = universe
    evo_cfg = EvoConfig(pop_size=8, n_generations=3, hidden=4,
                        early_stop_patience=10, seed=99)
    r1 = evolve(train, val, None, cfg, evo_cfg)
    r2 = evolve(train, val, None, cfg, evo_cfg)
    np.testing.assert_allclose(r1.champion_genome, r2.champion_genome)
    assert r1.val_fitness == pytest.approx(r2.val_fitness, nan_ok=True)


def test_champion_save_load_roundtrip(universe, tmp_path):
    cfg, train, val, test = universe
    evo_cfg = EvoConfig(pop_size=6, n_generations=2, hidden=4, seed=5)
    res = evolve(train, val, None, cfg, evo_cfg)
    path = tmp_path / "champ.npz"
    save_champion(res, path)
    pol = load_champion(path)
    assert isinstance(pol, EvoPolicy)
    np.testing.assert_allclose(pol.genome, res.champion_genome)


def test_load_champion_missing_returns_none(tmp_path):
    assert load_champion(tmp_path / "nope.npz") is None
