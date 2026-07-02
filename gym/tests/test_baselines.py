"""baselines: BuyAndHoldPolicy (the rule-based A3/A5b/PCB policies were removed with the
Prophets signal features they depended on in the rev-4 refactor)."""
from __future__ import annotations

import numpy as np

from gym.baselines import BuyAndHoldPolicy


def test_bah_always_one():
    pol = BuyAndHoldPolicy()
    for _ in range(5):
        obs = np.random.randn(3, 8).astype(np.float32)
        assert pol.act(obs) == 1.0
