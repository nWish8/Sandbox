"""Tests for evo_replay.py — the headless data layer + persistence (not the finplot draw)."""
from __future__ import annotations

import numpy as np
import pytest

from gym import features as F
from gym.baselines import BuyAndHoldPolicy
from gym.evo import EvoPolicy
from gym.evo_replay import GenerationReplay, build_world_frame, rebase_world_ohlc
from gym.population import run_population

DAY = 86_400


class FlatPolicy:
    def act(self, obs):  # noqa: ARG002
        return 0.0
    def reset(self):
        pass


def _build(cfg, write, symbols, seed=0, n=60):
    cfg.lookback = 5
    rng = np.random.default_rng(seed)
    for s in symbols:
        ts = [1_700_000_000 + i * DAY for i in range(n)]
        write(s, ts, list(100 * np.cumprod(1 + rng.normal(0.0008, 0.012, n))))
    vocab = F.build_vocab(symbols, cfg)
    dfs = []
    for s in symbols:
        df = F.build_ticker_frame(s, vocab, cfg)
        feat = F.feature_columns(df, cfg)
        df[feat] = df[feat].fillna(0.0)
        dfs.append(df)
    return dfs


@pytest.fixture
def gen_and_dfs(write_fixture):
    cfg, write = write_fixture
    dfs = _build(cfg, write, ["EV:A", "EV:B", "EV:C"], seed=1)
    n_feat = len(F.feature_columns(dfs[0], cfg))
    policies = [
        BuyAndHoldPolicy(),
        FlatPolicy(),
        EvoPolicy(n_feat, hidden=4, window_mode="last", lookback=5,
                  rng=np.random.default_rng(3)),
        EvoPolicy(n_feat, hidden=4, window_mode="last", lookback=5,
                  rng=np.random.default_rng(4)),
    ]
    results = run_population(policies, dfs, cfg, objective="timing_sortino")
    gen = GenerationReplay.from_results(results, ["EV:A", "EV:B", "EV:C"], "timing_sortino")
    return cfg, gen, dfs, results


# ─────────────────────────────────── construction + shapes

def test_from_results_shapes(gen_and_dfs):
    cfg, gen, dfs, results = gen_and_dfs
    assert gen.n_agents == 4
    assert gen.equity.shape == gen.weight.shape
    assert gen.bh_equity.shape[0] == gen.n_bars
    assert len(gen.names) == 4
    assert 0 <= gen.champion_idx < 4


def test_race_frame_has_agents_plus_bh(gen_and_dfs):
    cfg, gen, dfs, results = gen_and_dfs
    rf = gen.race_frame()
    assert "bh" in rf.columns
    for name in gen.names:
        assert name in rf.columns
    assert len(rf) == gen.n_bars


def test_world_frame_length_matches_race(gen_and_dfs):
    """T = sum_i (n_i - lookback): world frame and equity race must align bar-for-bar."""
    cfg, gen, dfs, results = gen_and_dfs
    world = build_world_frame(dfs, cfg.lookback)
    assert len(world) == gen.n_bars
    expected = sum(len(df) - cfg.lookback for df in dfs)
    assert len(world) == expected


def test_rebase_world_each_segment_starts_at_one(gen_and_dfs):
    """Every ticker segment is rebased to open=1.0 so wildly different price levels share one
    axis; this is the fix for the unreadable stitched world chart."""
    cfg, gen, dfs, results = gen_and_dfs
    world = build_world_frame(dfs, cfg.lookback)
    o = rebase_world_ohlc(world, gen.ticker_bounds)
    assert o.shape == (gen.n_bars, 4)
    for _tk, s, e in gen.ticker_bounds:
        assert o[s, 0] == pytest.approx(1.0)              # first open of each segment == 1.0
    # whole rebased frame lives on a sane ~O(1) scale, not raw price levels
    assert np.nanmax(o) < 100.0


def test_flat_agent_equity_is_constant(gen_and_dfs):
    cfg, gen, dfs, results = gen_and_dfs
    # agent 01 is the FlatPolicy -> equity stays at 1.0 across the pass
    flat_row = gen.equity[1]
    np.testing.assert_allclose(flat_row, 1.0, atol=1e-9)


# ─────────────────────────────────── leaderboard

def test_leaderboard_ranked_and_complete(gen_and_dfs):
    cfg, gen, dfs, results = gen_and_dfs
    board = gen.leaderboard()
    assert len(board) == gen.n_agents
    fits = [r["fitness"] for r in board]
    assert fits == sorted(fits, reverse=True)          # descending by fitness
    assert board[0]["rank"] == 1
    champs = [r for r in board if r["is_champion"]]
    assert len(champs) == 1


def test_leaderboard_champion_is_top_fitness(gen_and_dfs):
    cfg, gen, dfs, results = gen_and_dfs
    board = gen.leaderboard()
    # champion (best finite fitness) should not be outranked by a finite-fitness agent
    champ_rank = next(r["rank"] for r in board if r["is_champion"])
    assert champ_rank == 1


# ─────────────────────────────────── persistence

def test_save_load_roundtrip(gen_and_dfs, tmp_path):
    cfg, gen, dfs, results = gen_and_dfs
    path = tmp_path / "gen.npz"
    gen.save(path)
    loaded = GenerationReplay.load(path)
    np.testing.assert_allclose(loaded.equity, gen.equity, equal_nan=True)
    np.testing.assert_allclose(loaded.bh_equity, gen.bh_equity, equal_nan=True)
    assert loaded.names == gen.names
    assert loaded.champion_idx == gen.champion_idx
    assert loaded.objective == gen.objective
    assert loaded.ordered_slugs == gen.ordered_slugs


def test_ruined_agent_truncated_with_nan(write_fixture):
    cfg, write = write_fixture
    cfg.lookback = 5
    crash = np.linspace(100, 25, 50)               # -75% -> breaches ruin line
    ts = [1_700_000_000 + i * DAY for i in range(50)]
    write("EV:CRASH", ts, list(crash))
    vocab = F.build_vocab(["EV:CRASH"], cfg)
    df = F.build_ticker_frame("EV:CRASH", vocab, cfg)
    feat = F.feature_columns(df, cfg); df[feat] = df[feat].fillna(0.0)
    results = run_population([BuyAndHoldPolicy()], [df], cfg, ruin_frac=0.45,
                            objective="timing_sortino")
    gen = GenerationReplay.from_results(results, ["EV:CRASH"], "timing_sortino")
    assert gen.ruined[0]
    assert gen.ruin_bar[0] >= 0
    # equity is NaN after the ruin bar
    assert np.isnan(gen.equity[0, gen.ruin_bar[0] + 1:]).all() or gen.ruin_bar[0] == gen.n_bars - 1
