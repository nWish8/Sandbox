"""baselines.py — benchmark policy for the Signal Gym.

After the rev-4 refactor the only baseline is buy-and-hold: the passive reference curve
shown (orange dashed) in the population race. The rule-based policies (A3/A5b/PCB) were
removed with the Prophets signal features they depended on.
"""
from __future__ import annotations

import numpy as np


class BuyAndHoldPolicy:
    """Always hold 100%. The passive benchmark / reference curve."""

    def act(self, obs: np.ndarray) -> float:  # noqa: ARG002
        return 1.0

    def reset(self) -> None:
        pass
