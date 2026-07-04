"""rule_policies.py — classic long-only allocation rules as honest baselines.

Adapted from the spirit of je-suis-tm/quant-trading's rule collection into **weight-vector
form**, so every rule runs through the exact same ``strategy_eval`` accounting (costs,
slippage, fills) as the RL agents. If an agent can't beat these dumb rules under identical
frictions, that belongs in the report.

Row convention matches ``PortfolioEnv.portfolio_history``: the weight row k is held over
bar k−1→k and may only use closes up to k−1 (causal — tested by future-perturbation).

Also here: ``market_permutation_pvalue`` — a neurotrader888/mcpt-style **data permutation**
test for rules. Instead of permuting the rule's returns, it permutes the *market* (panel
log-returns shuffled with one shared order across assets, first price anchored — marginal
return distributions and cross-sectional correlation preserved, serial order destroyed)
and re-runs the rule on each synthetic market. H0: the rule's edge does not come from real
temporal structure. Affordable for rules because they re-run in microseconds; the RL
agents keep the return-level sign-flip gate in ``signif.py`` (retraining per permutation
is not).
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────── rules (closes (T,N) → weights (T,N))

def momentum_topk(closes: np.ndarray, k: int = 3, lookback: int = 90,
                  rebalance: int = 21, skip: int = 0) -> np.ndarray:
    """Cross-sectional momentum rotation: every ``rebalance`` bars, hold the ``k`` assets
    with the highest trailing ``lookback``-bar return, equal-weighted. ``skip`` excludes
    the most recent bars from the signal — the classic 12-1 construction (per the GS
    strategy notes / academic momentum literature) skips the last month to sidestep
    short-term reversal. Equal-weight during warm-up."""
    T, N = closes.shape
    k = min(k, N)
    warmup = lookback + skip
    eq = np.full(N, 1.0 / N)
    w = np.tile(eq, (T, 1))
    current = eq
    for t in range(T):
        if t <= warmup:
            current = eq
        elif (t - warmup - 1) % rebalance == 0:
            ref = t - 1 - skip                                       # data ≤ t−1 only
            mom = closes[ref] / closes[ref - lookback] - 1.0
            top = np.argsort(mom)[-k:]
            current = np.zeros(N)
            current[top] = 1.0 / k
        w[t] = current
    return w


def inverse_vol(closes: np.ndarray, window: int = 60, rebalance: int = 21) -> np.ndarray:
    """Risk-parity-lite: weights ∝ 1/vol of trailing ``window`` log returns, renormalised
    to the simplex, rebalanced every ``rebalance`` bars. Equal-weight during warm-up."""
    T, N = closes.shape
    eq = np.full(N, 1.0 / N)
    w = np.tile(eq, (T, 1))
    logret = np.zeros_like(closes)
    logret[1:] = np.log(closes[1:] / closes[:-1])
    current = eq
    for t in range(T):
        if t <= window:
            current = eq
        elif (t - window - 1) % rebalance == 0:
            vol = logret[t - window:t].std(axis=0, ddof=1)           # bars ≤ t−1
            inv = np.where(vol > 0, 1.0 / vol, 0.0)
            current = inv / inv.sum() if inv.sum() > 0 else eq
        w[t] = current
    return w


RULES = {
    "momentum_top3": lambda closes: momentum_topk(closes, k=3),
    "momentum_12_1": lambda closes: momentum_topk(closes, k=3, lookback=231, skip=21),
    "inverse_vol": inverse_vol,
}


# ─────────────────────────────────────────── market-permutation significance for rules

def permute_market(closes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthetic market: shuffle the panel's log-return ROWS with one shared permutation
    (cross-sectional correlation per bar survives; serial order does not), then rebuild
    prices from the anchored first row."""
    logret = np.log(closes[1:] / closes[:-1])
    perm = rng.permutation(len(logret))
    return np.vstack([closes[[0]],
                      closes[[0]] * np.exp(np.cumsum(logret[perm], axis=0))])


def market_permutation_pvalue(rule_fn, closes: np.ndarray, n_perms: int = 200,
                              seed: int = 42, cost=None) -> dict:
    """p-value for a rule's active Sharpe against markets with the serial order destroyed.

    Small p ⇒ the rule's edge needs the real market's temporal structure (it dies on the
    shuffled ones). Uses the same cost model / accounting as every other evaluation.
    """
    from stats import portfolio_stats
    from strategy_eval import CostModel, replay_weights

    cost = cost if cost is not None else CostModel(slippage_bps=0.0)

    def active_sharpe(px: np.ndarray) -> float:
        rets = np.zeros_like(px)
        rets[1:] = px[1:] / px[:-1] - 1.0
        res = replay_weights(rule_fn(px), rets, cost)
        r = res["ret"].to_numpy()[1:]
        b = rets.mean(axis=1)[1:]
        return portfolio_stats(r, b).get("active_sharpe", float("nan"))

    obs = active_sharpe(closes)
    if not np.isfinite(obs):
        return {"observed_active_sharpe": obs, "p_value": float("nan"), "n_perms": 0}
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perms):
        null = active_sharpe(permute_market(closes, rng))
        if np.isfinite(null) and null >= obs:
            hits += 1
    return {"observed_active_sharpe": float(obs),
            "p_value": (1 + hits) / (1 + n_perms), "n_perms": n_perms}
