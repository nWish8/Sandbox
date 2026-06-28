"""env_finrl.py — the FinRL interoperability seam.

The data layer is FinRL-standard (``date, tic, open, high, low, close, volume`` +
tech_indicators), so this module is the single place where the gym meets FinRL:

  * ``make_env(...)``        — the shared env factory the population/evo loops use; today
                              it returns the gym's own ``SignalGymEnv`` (one ticker per
                              episode), the right shape for the evolutionary race.
  * ``to_finrl_df(...)``     — concatenate per-ticker frames into FinRL's canonical long
                              DataFrame (sorted by date,tic; integer day index).
  * ``build_finrl_stock_env`` — OPTIONAL, guarded: instantiate FinRL's multi-stock
                              ``StockTradingEnv`` over our basket. Only imports ``finrl``
                              if it's installed (no hard dependency), so we can drop in
                              FinRL's DRL agents later without a data transform.
  * ``finrl_account_curve`` — map a FinRL env's post-rollout ``account_value`` onto the
                              gym's equity-curve shape, so FinRL runs can be rendered in
                              our pyqtgraph/finplot viz.

Why the gym env, not FinRL's, for the race: FinRL's ``StockTradingEnv`` manages the whole
basket as one simultaneous portfolio (action = an allocation vector over all tickers).
The evolutionary race is one-ticker-per-episode with capital threaded across a sequence —
a different (complementary) paradigm. Both produce equity curves, so the viz layer can
render either.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gym.config import Config, DEFAULT_CONFIG
from gym.env import SignalGymEnv
from gym.features import feature_columns, load_ticker


# ─────────────────────────────────────────── shared factory

def make_env(source, cfg: Config = DEFAULT_CONFIG,
             obs_stats: tuple[np.ndarray, np.ndarray] | None = None) -> SignalGymEnv:
    """Build a single-ticker gym env from a DataFrame or a ticker slug/symbol.

    The common entry point so callers don't hard-code ``SignalGymEnv``; a future FinRL
    env path can be selected here behind the same signature.
    """
    df = source if isinstance(source, pd.DataFrame) else load_ticker(_slugify(source), cfg)
    return SignalGymEnv(df, cfg, obs_stats=obs_stats)


def _slugify(s: str) -> str:
    from gym.config import slug
    return slug(s) if any(ch in s for ch in "=^.:") else s


# ─────────────────────────────────────────── FinRL long-format converter

def to_finrl_df(dfs: list[pd.DataFrame], cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Concatenate per-ticker frames into FinRL's canonical long DataFrame.

    Columns: date, tic, open, high, low, close, volume, <tech_indicators>. Sorted by
    (date, tic); the integer index is the day ordinal FinRL's envs expect
    (``df.index = df.date.factorize()[0]``).
    """
    tech = list(cfg.tech_indicators)
    keep = ["date", "tic", "open", "high", "low", "close", "volume", *tech]
    parts = []
    for df in dfs:
        d = df.copy()
        if "tic" not in d.columns and "ticker" in d.columns:
            d["tic"] = d["ticker"]
        parts.append(d[[c for c in keep if c in d.columns]])
    long = pd.concat(parts, ignore_index=True)
    long = long.sort_values(["date", "tic"]).reset_index(drop=True)
    long.index = long["date"].factorize()[0]
    return long


# ─────────────────────────────────────────── optional FinRL env (guarded)

def finrl_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("finrl") is not None


def build_finrl_stock_env(dfs: list[pd.DataFrame], cfg: Config = DEFAULT_CONFIG,
                          initial_amount: float = 1_000_000.0, hmax: int = 100,
                          reward_scaling: float = 1e-4):
    """Instantiate FinRL's multi-stock ``StockTradingEnv`` over our basket.

    Optional path — requires ``pip install finrl``. This is the seam for plugging in
    FinRL's DRL agents (PPO/SAC/TD3/…) on the *same* data later. Raises a clear
    ImportError if FinRL isn't installed.
    """
    if not finrl_available():
        raise ImportError(
            "FinRL is not installed. `pip install finrl` to use the multi-stock "
            "StockTradingEnv path; the evolutionary race does not need it."
        )
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    long = to_finrl_df(dfs, cfg)
    tech = [c for c in cfg.tech_indicators if c in long.columns]
    stock_dim = long["tic"].nunique()
    state_space = 1 + 2 * stock_dim + len(tech) * stock_dim
    cost = cfg.commission_bps / 1e4
    return StockTradingEnv(
        df=long,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=initial_amount,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[cost] * stock_dim,
        sell_cost_pct=[cost] * stock_dim,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=tech,
    )


def finrl_account_curve(finrl_env) -> pd.DataFrame:
    """Map a FinRL env's post-rollout account history onto the gym's equity-curve shape
    (columns: date, equity), so a FinRL run can feed the existing viz layer."""
    acct = finrl_env.save_asset_memory()            # FinRL: date + account_value
    acct = acct.rename(columns={"account_value": "equity"})
    acct["equity"] = acct["equity"] / float(acct["equity"].iloc[0])   # rebase to 1.0
    return acct
