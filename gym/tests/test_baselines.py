"""T3.1 — baselines: BuyAndHoldPolicy and RulePolicy (A3, A5b, PCB)."""
from __future__ import annotations

import numpy as np
import pytest

from gym.baselines import BuyAndHoldPolicy, RulePolicy, make_rule_policies


# ─── minimal feature column list for testing

FEAT_COLS = [
    "ret_1", "vol_20",
    "Buy Signal_within_3",
    "Buy Signal_bars_since",
    "Sell Signal_within_3",
    "Sell Signal_bars_since",
    "bull_active_count",
    "bear_active_count",
]
N_FEAT = len(FEAT_COLS)
COL_IDX = {c: i for i, c in enumerate(FEAT_COLS)}


def _obs(values: dict, lookback: int = 3) -> np.ndarray:
    """Build a (lookback, N_FEAT) obs with last row set per `values`."""
    obs = np.zeros((lookback, N_FEAT), dtype=np.float32)
    for col, val in values.items():
        obs[-1, COL_IDX[col]] = val
    return obs


# ─── BuyAndHoldPolicy

def test_bah_always_one():
    pol = BuyAndHoldPolicy()
    for _ in range(5):
        obs = np.random.randn(3, N_FEAT).astype(np.float32)
        assert pol.act(obs) == 1.0


# ─── A3 rule

class TestA3:
    def _pol(self):
        return RulePolicy(FEAT_COLS, rule="A3")

    def test_long_when_bull_within3(self):
        pol = self._pol()
        obs = _obs({"Buy Signal_within_3": 1})
        assert pol.act(obs) == 1.0

    def test_flat_when_no_bull_within3(self):
        pol = self._pol()
        obs = _obs({"Buy Signal_within_3": 0})
        assert pol.act(obs) == 0.0

    def test_sell_within3_does_not_trigger(self):
        pol = self._pol()
        # Sell signals are bear-directional — A3 should NOT fire on them
        obs = _obs({"Sell Signal_within_3": 1, "Buy Signal_within_3": 0})
        assert pol.act(obs) == 0.0


# ─── A5b rule

class TestA5b:
    def _pol(self):
        return RulePolicy(FEAT_COLS, rule="A5b")

    def test_long_when_bull_active_no_bear(self):
        pol = self._pol()
        obs = _obs({"Buy Signal_bars_since": 3, "bear_active_count": 0})
        assert pol.act(obs) == 1.0

    def test_flat_when_bear_active(self):
        pol = self._pol()
        obs = _obs({"Buy Signal_bars_since": 2, "bear_active_count": 1})
        assert pol.act(obs) == 0.0

    def test_flat_when_bars_since_gte5(self):
        pol = self._pol()
        obs = _obs({"Buy Signal_bars_since": 5, "bear_active_count": 0})
        assert pol.act(obs) == 0.0

    def test_long_at_bars_since_4(self):
        pol = self._pol()
        obs = _obs({"Buy Signal_bars_since": 4, "bear_active_count": 0})
        assert pol.act(obs) == 1.0


# ─── PCB rule

class TestPCB:
    def _pol(self):
        return RulePolicy(FEAT_COLS, rule="PCB")

    def test_long_when_bull_count_ge1(self):
        pol = self._pol()
        obs = _obs({"bull_active_count": 2})
        assert pol.act(obs) == 1.0

    def test_flat_when_bull_count_zero(self):
        pol = self._pol()
        obs = _obs({"bull_active_count": 0})
        assert pol.act(obs) == 0.0


# ─── make_rule_policies factory

def test_make_rule_policies_returns_all():
    pols = make_rule_policies(FEAT_COLS)
    assert set(pols.keys()) == {"A3", "A5b", "PCB"}
    for rule, pol in pols.items():
        assert isinstance(pol, RulePolicy)
        assert pol.rule == rule


# ─── invalid rule

def test_invalid_rule_raises():
    with pytest.raises(ValueError, match="rule must be one of"):
        RulePolicy(FEAT_COLS, rule="bad")
