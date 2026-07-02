"""Shared pytest path setup for Signal Gym tests.

Ensures the repo root is importable so ``import gym.<module>`` resolves to this package
regardless of where pytest is invoked from.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # .../Sandbox  (enables `import gym.<module>`)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The FinRL-side modules (pipeline, finrl_patch, portfolio, …) use flat imports
# (`from pipeline import …`), matching how they run as scripts. Put gym/ on the
# path so those resolve under pytest too.
GYM = Path(__file__).resolve().parents[1]    # .../Sandbox/gym
if str(GYM) not in sys.path:
    sys.path.insert(0, str(GYM))
