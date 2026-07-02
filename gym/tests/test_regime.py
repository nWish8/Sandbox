"""T5 — causal regime labelling + per-regime evaluation."""
from __future__ import annotations

import numpy as np

from regime import REGIMES, format_regime_table, label_regimes, stats_by_regime


def test_labels_trends_correctly():
    up = np.full(80, 0.004)                        # steady climb → bull
    down = np.full(80, -0.004)                     # steady fall → bear
    flat = np.zeros(80)                            # no trend → choppy
    assert label_regimes(up)[-1] == "bull"
    assert label_regimes(down)[-1] == "bear"
    assert label_regimes(flat)[-1] == "choppy"


def test_labelling_is_causal():
    """Perturbing the final bar's benchmark return cannot change any earlier bar's label."""
    rng = np.random.default_rng(0)
    b = rng.normal(0.0005, 0.01, 100)
    base = label_regimes(b)
    pert = b.copy(); pert[-1] += 0.5               # shock only the last bar
    shocked = label_regimes(pert)
    assert list(base[:-1]) == list(shocked[:-1])


def test_stats_by_regime_partitions_bars():
    # first half bull, second half bear
    b = np.concatenate([np.full(60, 0.004), np.full(60, -0.004)])
    r = b + 0.001                                  # portfolio slightly beats benchmark
    by = stats_by_regime(r, b)
    assert set(by).issubset(set(REGIMES))
    total = sum(s["n_bars"] for s in by.values())
    assert total == len(r)                         # every bar assigned to exactly one regime


def test_format_regime_table_renders():
    b = np.concatenate([np.full(60, 0.004), np.full(60, -0.004)])
    table = format_regime_table(stats_by_regime(b + 0.001, b))
    assert "regime" in table and "active_sharpe" in table
