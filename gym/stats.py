"""stats.py — backtesting.py-style performance stats + a pluggable objective registry.

`compute_stats` takes an equity curve (the same DataFrame shape produced by
`backtest.BacktestResult.equity_curve` and by `population.run_pass`) and returns a flat
dict of metrics modelled on kernc/backtesting.py's stats block: returns, risk-adjusted
ratios (Sharpe / Sortino / Calmar), drawdowns, and trade statistics.

The agent sets a continuous target weight in [0, 1], so there are no discrete round-trip
trades. We define a **trade as a maximal run of nonzero exposure** (entry when weight goes
0 -> >0, exit when it returns to 0); the trade's return is the compounded portfolio return
over that run. Win Rate / Profit Factor / Expectancy / SQN / Kelly are computed on those
holding-period returns.

The OBJECTIVES registry maps an objective name to a scalar where **higher is better**, used
by population selection. NaN/undefined objectives map to -inf so degenerate agents sort last.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Objectives whose value is meaningless without enough trades; guarded by `min_trades`.
_TRADE_OBJECTIVES = frozenset(
    {"profit_factor", "expectancy", "sqn", "win_rate", "kelly", "avg_trade"}
)


# ─────────────────────────────────────────── trade derivation

def holding_period_returns(weight: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """Return one compounded return per maximal run of nonzero exposure.

    weight[i] is the target weight realised over bar i; ret[i] is that bar's portfolio
    return. A trade spans a contiguous block where weight > 0. A position still open at the
    final bar is counted (closed at the last bar).
    """
    w = np.asarray(weight, dtype=np.float64)
    r = np.asarray(ret, dtype=np.float64)
    n = len(w)
    in_pos = w > 0
    out: list[float] = []
    i = 0
    while i < n:
        if in_pos[i]:
            j = i
            while j < n and in_pos[j]:
                j += 1
            out.append(float(np.prod(1.0 + r[i:j]) - 1.0))
            i = j
        else:
            i += 1
    return np.asarray(out, dtype=np.float64)


# ─────────────────────────────────────────── drawdown

def _drawdown(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0          # <= 0


def _avg_drawdown(equity: np.ndarray) -> float:
    """Mean depth of each distinct underwater episode (peak -> recovery)."""
    dd = _drawdown(equity)
    troughs: list[float] = []
    cur = 0.0
    underwater = False
    for d in dd:
        if d < 0:
            underwater = True
            cur = min(cur, d)
        elif underwater:                # surfaced -> close the episode
            troughs.append(cur)
            cur = 0.0
            underwater = False
    if underwater:
        troughs.append(cur)
    return float(np.mean(troughs)) if troughs else 0.0


# ─────────────────────────────────────────── ratios

def _sharpe(rets: np.ndarray, ppy: int) -> float:
    if len(rets) < 2:
        return float("nan")
    std = np.std(rets, ddof=1)
    if std == 0:
        return float("nan")
    return float(np.mean(rets) / std * np.sqrt(ppy))


def _sortino(rets: np.ndarray, ppy: int) -> float:
    if len(rets) < 2:
        return float("nan")
    downside = np.minimum(rets, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return float("nan")
    return float(np.mean(rets) / dd * np.sqrt(ppy))


def _timing_ratio(active: np.ndarray, ppy: int, downside: bool) -> float:
    """Risk-adjusted ratio of the *active* (timing-excess) return series.

    A constant-exposure agent has active==0 every bar -> returns 0.0 (neutral), NOT NaN,
    so a non-timer sits at zero rather than being treated as undefined/worst. A real timer
    (lighter into drops, heavier into rallies than its own average) scores positive; a bad
    timer scores negative. `downside=True` uses a Sortino-style denominator.
    """
    if len(active) < 2:
        return 0.0
    if np.allclose(active, 0.0):          # constant-exposure agent: it IS its twin
        return 0.0
    if downside:
        d = np.sqrt(np.mean(np.minimum(active, 0.0) ** 2))
        if d == 0:                        # never underperformed the twin -> use full spread
            d = np.std(active, ddof=1)
    else:
        d = np.std(active, ddof=1)
    if d == 0:
        return 0.0
    return float(np.mean(active) / d * np.sqrt(ppy))


# ─────────────────────────────────────────── main entry

def compute_stats(ec: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR,
                  initial_equity: float = 1.0) -> dict:
    """Compute the full stats block from an equity-curve DataFrame.

    Required columns: equity, weight, ret. Optional: bh_equity, bh_ret (for B&H reference
    + beta/alpha). `equity` is assumed to start near `initial_equity`.
    """
    out: dict[str, float] = {}
    if ec is None or len(ec) == 0:
        return out

    equity = ec["equity"].to_numpy(dtype=np.float64)
    rets = ec["ret"].to_numpy(dtype=np.float64)
    weight = ec["weight"].to_numpy(dtype=np.float64)
    n = len(rets)
    ppy = periods_per_year

    final = float(equity[-1])
    out["n_bars"] = n
    out["equity_final"] = final
    out["equity_peak"] = float(np.max(equity))
    out["total_return"] = final / initial_equity - 1.0
    out["exposure_pct"] = float(np.mean(weight > 0) * 100.0)
    out["mean_weight"] = float(np.mean(weight))
    out["weight_std"] = float(np.std(weight))      # exposure variation: 0 = constant hold

    # activity for a continuous sizer (n_trades undercounts when it never reaches cash):
    # turnover = total |Δweight| (incl. the initial entry from flat); n_reweights = # of bars
    # where exposure actually changed. These read the real re-sizing, not just round-trips.
    dw = np.abs(np.diff(weight, prepend=0.0))
    out["turnover"] = float(np.sum(dw))
    out["n_reweights"] = int(np.sum(dw > 1e-6))

    # annualised return (CAGR) + volatility
    if n > 0 and final > 0:
        out["return_ann"] = float((final / initial_equity) ** (ppy / n) - 1.0)
    else:
        out["return_ann"] = float("nan")
    out["volatility_ann"] = float(np.std(rets, ddof=1) * np.sqrt(ppy)) if n >= 2 else float("nan")

    out["sharpe"] = _sharpe(rets, ppy)
    out["sortino"] = _sortino(rets, ppy)

    dd = _drawdown(equity)
    out["max_drawdown"] = float(np.min(dd)) if n else 0.0
    out["avg_drawdown"] = _avg_drawdown(equity)
    out["calmar"] = (out["return_ann"] / abs(out["max_drawdown"])
                     if out["max_drawdown"] < 0 and np.isfinite(out["return_ann"])
                     else float("nan"))

    # B&H reference + alpha/beta
    if "bh_equity" in ec.columns:
        bh = ec["bh_equity"].to_numpy(dtype=np.float64)
        out["bh_return"] = float(bh[-1] / initial_equity - 1.0)
    if "bh_ret" in ec.columns and n >= 2:
        bh_r = ec["bh_ret"].to_numpy(dtype=np.float64)
        var_bh = np.var(bh_r, ddof=1)
        beta = float(np.cov(rets, bh_r, ddof=1)[0, 1] / var_bh) if var_bh > 0 else float("nan")
        out["beta"] = beta
        if np.isfinite(beta) and "bh_return" in out:
            out["alpha"] = out["total_return"] - beta * out["bh_return"]

        # ── timing skill vs the agent's OWN average-exposure twin (drives selection).
        # active_t = agent_ret_t - mean_weight * market_ret_t. A constant-exposure agent IS
        # its twin (active==0 -> score 0); only feature-timed exposure changes earn a
        # positive ratio. Isolates timing — de-risking can't fake it (the twin de-risks too).
        wbar = float(np.mean(weight))
        active = rets - wbar * bh_r
        out["active_return"] = float(np.sum(active))
        out["timing_sharpe"] = _timing_ratio(active, ppy, downside=False)
        out["timing_sortino"] = _timing_ratio(active, ppy, downside=True)

    # ── trade stats (holding-period returns)
    tr = holding_period_returns(weight, rets)
    out["n_trades"] = int(len(tr))
    if len(tr) > 0:
        wins = tr[tr > 0]
        losses = tr[tr < 0]
        out["win_rate"] = float(len(wins) / len(tr) * 100.0)
        out["best_trade"] = float(np.max(tr))
        out["worst_trade"] = float(np.min(tr))
        out["avg_trade"] = float(np.mean(tr))
        out["expectancy"] = out["avg_trade"]
        gross_profit = float(np.sum(wins))
        gross_loss = float(-np.sum(losses))
        out["profit_factor"] = (gross_profit / gross_loss if gross_loss > 0
                                else (float("inf") if gross_profit > 0 else float("nan")))
        out["sqn"] = (float(np.sqrt(len(tr)) * np.mean(tr) / np.std(tr, ddof=1))
                      if len(tr) >= 2 and np.std(tr, ddof=1) > 0 else float("nan"))
        if len(wins) and len(losses):
            payoff = np.mean(wins) / abs(np.mean(losses))
            p = len(wins) / len(tr)
            out["kelly"] = float(p - (1.0 - p) / payoff) if payoff > 0 else float("nan")
        else:
            out["kelly"] = float("nan")
    else:
        for k in ("win_rate", "best_trade", "worst_trade", "avg_trade", "expectancy",
                  "profit_factor", "sqn", "kelly"):
            out[k] = float("nan")

    return out


# ─────────────────────────────────────────── objective registry

#: name -> callable(stats_dict) -> float, higher is better. Used for population selection.
OBJECTIVES: dict[str, callable] = {
    "sortino":        lambda s: s.get("sortino"),
    "sharpe":         lambda s: s.get("sharpe"),
    "calmar":         lambda s: s.get("calmar"),
    "return":         lambda s: s.get("total_return"),
    "cagr":           lambda s: s.get("return_ann"),
    "profit_factor":  lambda s: s.get("profit_factor"),
    "sqn":            lambda s: s.get("sqn"),
    "expectancy":     lambda s: s.get("expectancy"),
    "win_rate":       lambda s: s.get("win_rate"),
    "kelly":          lambda s: s.get("kelly"),
    # risk-aware composite: reward return, punish drawdown depth
    "return_over_dd": lambda s: (s.get("total_return") / abs(s["max_drawdown"])
                                 if s.get("max_drawdown", 0) < 0 else float("nan")),
    # timing skill vs the agent's own average-exposure twin — rewards ONLY genuine timing
    # (constant exposure scores 0). The objective for "learn to time, not to hold less".
    "timing_sortino": lambda s: s.get("timing_sortino"),
    "timing_sharpe":  lambda s: s.get("timing_sharpe"),
}
DEFAULT_OBJECTIVE = "sortino"


def objective_value(stats: dict, name: str = DEFAULT_OBJECTIVE,
                    min_trades: int = 3) -> float:
    """Scalar fitness for selection (higher = better).

    Returns -inf when the objective is undefined (NaN/inf) or when a trade-based objective
    is asked for but the agent took fewer than `min_trades` trades — so degenerate agents
    (never trade, one lucky trade, no losers) sort to the bottom rather than winning by
    a vacuous metric.
    """
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}; choices: {sorted(OBJECTIVES)}")
    if name in _TRADE_OBJECTIVES and stats.get("n_trades", 0) < min_trades:
        return float("-inf")
    v = OBJECTIVES[name](stats)
    if v is None or not np.isfinite(v):
        return float("-inf")
    return float(v)
