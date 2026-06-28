"""backtest.py — run any policy over a SignalGymEnv episode and compute metrics.

Two entry points:
  run_backtest(env, policy)         — step through env, collect everything, return result
  fast_vectorized(weights, df, cfg) — vectorized equity/metrics given pre-computed weights

`fast_vectorized` uses the same two-term fill accounting as env.step so the equity curve
is numerically identical to the env loop (verified by T3.2 test).

BacktestResult carries: trades list, equity_curve DataFrame, metrics dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gym.config import Config, DEFAULT_CONFIG

TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────── result container

@dataclass
class BacktestResult:
    trades: list[dict]
    equity_curve: pd.DataFrame          # columns: timestamp, equity, bh_equity, weight, reward
    metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.metrics:
            self.metrics = _compute_metrics(self.equity_curve)


# ─────────────────────────────────────────── env-based loop

def run_backtest(env, policy) -> BacktestResult:
    """Step through *env* using *policy.act(obs)* and collect full history.

    Policy interface: act(obs: np.ndarray) -> float (target weight in [0,1]).
    Optional policy.reset() is called before the episode starts.
    """
    if hasattr(policy, "reset"):
        policy.reset()

    obs, _ = env.reset()
    trades: list[dict] = []
    rows: list[dict] = []
    done = False

    while not done:
        w = float(np.clip(policy.act(obs), 0.0, 1.0))
        obs, reward, done, _, info = env.step(np.array([w], dtype=np.float32))
        rows.append({
            "timestamp": info["timestamp"],
            "equity":    info["equity"],
            "bh_equity": info["bh_equity"],
            "weight":    info["weight"],
            "ret":       info["ret"],
            "bh_ret":    info["bh_ret"],
            "excess":    info["excess"],
            "reward":    reward,
        })
        if info["trade"] is not None:
            trades.append(info["trade"])

    ec = pd.DataFrame(rows)
    return BacktestResult(trades=trades, equity_curve=ec, metrics=_compute_metrics(ec))


# ─────────────────────────────────────────── vectorized fast-path

def fast_vectorized(weights: np.ndarray, df: pd.DataFrame,
                    cfg: Config = DEFAULT_CONFIG) -> BacktestResult:
    """Compute equity from a pre-determined weight vector *without* stepping the env.

    Parameters
    ----------
    weights : np.ndarray shape (N,)
        Target weight at each bar (same length as df). Index 0 = first bar.
        w[i] is realised over the i→i+1 gap (filled at open[i+1]).
    df : pd.DataFrame
        Must contain 'close', 'open', 'timestamp' columns.
    cfg : Config

    Returns a BacktestResult with equity_curve and metrics.
    The result is numerically identical to stepping SignalGymEnv with the same weights.
    """
    c = df["close"].to_numpy(dtype=np.float64)
    o = df["open"].to_numpy(dtype=np.float64)
    ts = df["timestamp"].to_numpy()
    n = len(c)
    commission = cfg.commission_bps / 1e4

    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != n:
        raise ValueError(f"weights length {w.shape[0]} != df rows {n}")

    rows: list[dict] = []
    trades: list[dict] = []
    equity = 1.0
    bh_equity = 1.0
    w_prev = 0.0

    for i in range(n - 1):
        w_t = float(np.clip(w[i], 0.0, 1.0))
        c_t, o_n, c_n = c[i], o[i + 1], c[i + 1]

        gap   = w_prev * (o_n / c_t - 1.0)
        intra = w_t    * (c_n / o_n - 1.0)
        turn  = commission * abs(w_t - w_prev)
        r     = gap + intra - turn
        bh_r  = c_n / c_t - 1.0
        excess = r - bh_r

        equity    *= (1.0 + r)
        bh_equity *= (1.0 + bh_r)

        rows.append({
            "timestamp": int(ts[i]),
            "equity":    equity,
            "bh_equity": bh_equity,
            "weight":    w_t,
            "ret":       r,
            "bh_ret":    bh_r,
            "excess":    excess,
            "reward":    0.0,   # reward not computed in vectorised path
        })
        if w_t != w_prev:
            trades.append({"bar_idx": i, "from": w_prev, "to": w_t, "price": o_n})
        w_prev = w_t

    ec = pd.DataFrame(rows)
    return BacktestResult(trades=trades, equity_curve=ec, metrics=_compute_metrics(ec))


# ─────────────────────────────────────────── metrics

def _compute_metrics(ec: pd.DataFrame) -> dict:
    if ec.empty:
        return {}

    rets = ec["ret"].to_numpy(dtype=np.float64)
    excess = ec["excess"].to_numpy(dtype=np.float64)

    total_return = float(ec["equity"].iloc[-1] - 1.0)
    bh_return    = float(ec["bh_equity"].iloc[-1] - 1.0)

    sharpe    = _sharpe(rets)
    exc_sharpe = _sharpe(excess)

    eq = ec["equity"].to_numpy(dtype=np.float64)
    max_dd = float(_max_drawdown(eq))

    beat_rate = float(np.mean(excess > 0)) if len(excess) else float("nan")

    # trades
    n_trades = len(ec[ec["weight"].diff().fillna(0) != 0]) if "weight" in ec.columns else 0

    return {
        "total_return":   total_return,
        "bh_return":      bh_return,
        "excess_return":  total_return - bh_return,
        "sharpe":         sharpe,
        "excess_sharpe":  exc_sharpe,
        "max_drawdown":   max_dd,
        "beat_rate":      beat_rate,
        "n_bars":         len(ec),
        "n_trades":       n_trades,
    }


def _sharpe(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return float("nan")
    mu  = np.mean(rets)
    std = np.std(rets, ddof=1)
    if std == 0:
        return float("nan")
    return float(mu / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    return float(np.min(dd))


# ─────────────────────────────────────────── convenience

def compare_policies(policies: dict, env_factory, n_episodes: int = 1) -> pd.DataFrame:
    """Run each policy through `n_episodes` freshly-reset envs and return a summary DataFrame.

    Parameters
    ----------
    policies : dict[str, policy]
        Named policy objects (each with .act(obs) -> float).
    env_factory : callable() -> SignalGymEnv
        Called once per episode per policy.
    n_episodes : int
        Number of independent episodes to average over.
    """
    records = []
    for name, pol in policies.items():
        for _ in range(n_episodes):
            env = env_factory()
            res = run_backtest(env, pol)
            records.append({"policy": name, **res.metrics})
    df = pd.DataFrame(records)
    if "policy" in df.columns:
        return df.groupby("policy").mean(numeric_only=True).reset_index()
    return df
