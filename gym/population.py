"""population.py — run a policy (or a whole population) through a pass of the universe.

A *pass* (one generation's evaluation) plays a policy through a shared, randomly-ordered
sequence of tickers, threading **capital across the whole pass**: equity is one continuous
curve, so a bad early ticker can ruin an agent. Long-only spot can't reach zero, so "goes
broke" is a configurable **ruin line**: when threaded equity drops below `ruin_frac`, the
agent is dead and its curve ends there.

`run_pass`        — one policy through the ordered tickers -> PassResult (threaded curve + stats).
`run_population`  — many policies through the SAME ordered tickers -> list[PassResult].
`select_survivors`— rank non-ruined agents by a chosen objective (val-gated upstream).
`shuffled_order`  — a re-rollable shared ticker order for a generation.

The per-ticker step loop is `backtest.run_backtest` over `SignalGymEnv`; this module only
adds the cross-ticker threading, the ruin rule, and selection.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gym.backtest import run_backtest
from gym.config import Config, DEFAULT_CONFIG
from gym.env import SignalGymEnv
from gym.stats import compute_stats, objective_value

DEFAULT_RUIN_FRAC = 0.45


@dataclass
class PassResult:
    """One policy's threaded run through a pass."""
    ec: pd.DataFrame                                   # threaded equity curve
    ticker_bounds: list[tuple[str, int, int]]          # (ticker, start_row, end_row) per segment
    trades: list[dict]                                 # bar_idx offset into the threaded curve
    ruined: bool
    ruin_bar: int | None
    stats: dict = field(default_factory=dict)


def shuffled_order(dfs: list[pd.DataFrame], seed: int) -> list[pd.DataFrame]:
    """A shared, re-rollable ticker order for a generation (same seed -> same order)."""
    order = list(dfs)
    random.Random(seed).shuffle(order)
    return order


def _ticker_name(df: pd.DataFrame, fallback: str) -> str:
    return str(df["ticker"].iloc[0]) if "ticker" in df.columns else fallback


def run_pass(policy, ordered_dfs: list[pd.DataFrame], cfg: Config = DEFAULT_CONFIG,
             ruin_frac: float = DEFAULT_RUIN_FRAC, obs_stats: tuple | None = None,
             periods_per_year: int = 252, objective: str | None = None) -> PassResult:
    """Thread *policy* through *ordered_dfs*, carrying capital across tickers.

    Equity (and a parallel B&H reference) compound across the whole pass. Once threaded
    equity falls below `ruin_frac`, the agent is marked ruined and the curve is truncated
    (NaN thereafter); stats are computed on the realised pre-ruin curve.
    """
    if hasattr(policy, "reset"):
        policy.reset()

    capital = 1.0
    bh_capital = 1.0
    frames: list[pd.DataFrame] = []
    bounds: list[tuple[str, int, int]] = []
    trades: list[dict] = []
    row_offset = 0

    for k, df in enumerate(ordered_dfs):
        env = SignalGymEnv(df, cfg, obs_stats=obs_stats)
        res = run_backtest(env, policy)
        seg = res.equity_curve.copy()
        if seg.empty:
            continue

        # thread: scale this ticker's curve (which restarts at 1.0) by carried capital
        seg["equity"] = capital * seg["equity"].to_numpy(dtype=np.float64)
        seg["bh_equity"] = bh_capital * seg["bh_equity"].to_numpy(dtype=np.float64)
        seg["ticker"] = _ticker_name(df, f"T{k}")

        start = row_offset
        end = row_offset + len(seg) - 1
        bounds.append((seg["ticker"].iloc[0], start, end))
        for t in res.trades:
            trades.append({**t, "bar_idx": t["bar_idx"] + row_offset, "ticker": seg["ticker"].iloc[0]})

        frames.append(seg)
        capital = float(seg["equity"].iloc[-1])
        bh_capital = float(seg["bh_equity"].iloc[-1])
        row_offset = end + 1

    if not frames:
        return PassResult(pd.DataFrame(), [], [], ruined=False, ruin_bar=None, stats={})

    ec = pd.concat(frames, ignore_index=True)

    # ── ruin detection on the threaded curve
    eq = ec["equity"].to_numpy(dtype=np.float64)
    below = np.where(eq < ruin_frac)[0]
    ruined = below.size > 0
    ruin_bar = int(below[0]) if ruined else None

    if ruined:
        # truncate display curve after death; stats use the realised pre-ruin slice
        stats = compute_stats(ec.iloc[: ruin_bar + 1], periods_per_year=periods_per_year)
        ec.loc[ruin_bar + 1:, ["equity", "weight"]] = np.nan
    else:
        stats = compute_stats(ec, periods_per_year=periods_per_year)

    stats["ruined"] = ruined
    if objective is not None:
        stats["objective"] = objective
        stats["fitness"] = float("-inf") if ruined else objective_value(stats, objective)

    return PassResult(ec=ec, ticker_bounds=bounds, trades=trades,
                      ruined=ruined, ruin_bar=ruin_bar, stats=stats)


def run_population(policies: list, ordered_dfs: list[pd.DataFrame],
                   cfg: Config = DEFAULT_CONFIG, ruin_frac: float = DEFAULT_RUIN_FRAC,
                   obs_stats: tuple | None = None, periods_per_year: int = 252,
                   objective: str | None = None) -> list[PassResult]:
    """Run every policy through the SAME ordered pass (so their curves share a time axis)."""
    return [
        run_pass(p, ordered_dfs, cfg, ruin_frac=ruin_frac, obs_stats=obs_stats,
                 periods_per_year=periods_per_year, objective=objective)
        for p in policies
    ]


def select_survivors(results: list[PassResult], objective: str, n_select: int,
                     min_trades: int = 3) -> list[int]:
    """Indices of the top `n_select` non-ruined agents, ranked by `objective` (desc).

    Ruined agents are never selected. Ties and degenerate metrics are handled by
    `stats.objective_value` (degenerate -> -inf). Returns fewer than `n_select` indices if
    not enough agents survive.
    """
    scored: list[tuple[float, int]] = []
    for i, r in enumerate(results):
        if r.ruined or not r.stats:
            continue
        scored.append((objective_value(r.stats, objective, min_trades=min_trades), i))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [i for score, i in scored[:n_select] if np.isfinite(score)]
