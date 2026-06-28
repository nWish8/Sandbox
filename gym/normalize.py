"""normalize.py — train-fit per-feature observation normalization.

Standardises each feature column with TRAIN-fit (mean, std) so the policy net isn't
dominated by raw-scale features. The evolutionary policies (and any neural learner) need
this — unnormalised observations collapse the policy to a constant action. Relocated here
from the removed agent_rl.py (rev 4) so it has no PPO/SB3 dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gym.config import Config, DEFAULT_CONFIG


def fit_obs_stats(train_dfs: list[pd.DataFrame], cfg: Config = DEFAULT_CONFIG
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-feature (mean, std) over pooled TRAINING bars only. Zero/degenerate std
    -> 1 so standardization is a safe no-op for constant features. Returns float32
    arrays aligned to feature_columns(df, cfg)."""
    from gym.features import feature_columns

    feat_cols = feature_columns(train_dfs[0], cfg)
    stacked = np.vstack([df[feat_cols].to_numpy(dtype=np.float64) for df in train_dfs])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def save_obs_stats(path, mean: np.ndarray, std: np.ndarray) -> None:
    np.savez(str(path), mean=mean, std=std)


def load_obs_stats(path) -> tuple[np.ndarray, np.ndarray] | None:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    d = np.load(str(p))
    return d["mean"], d["std"]
