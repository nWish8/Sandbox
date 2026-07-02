"""rewards.py — the per-step training-reward registry for the reward investigation (v2).

These are the candidate objectives the agent is *trained* on — the thing the policy gradient
optimises. They are deliberately **online and causal**: each reads only the latest realised bar
(and an EMA of the past), never the whole episode. This is the correction over the retired
``timing_sortino``, which normalised against a twin built from the *full-window* mean exposure
(a look-ahead) and annualised a 1-hour series with √252. Annualisation belongs in *evaluation*
(``stats.portfolio_stats``), not in the per-step reward, so these stay scale-consistent.

Each reward is ``fn(env) -> float`` and may keep state in ``env.reward_state`` (reset per
episode). The basket return for bar t is ``env.returns_memory[-1]`` (net of turnover cost) and
the equal-weight benchmark return is ``env.bench_returns_memory[-1]``.

Registry (≥4 candidates, compared head-to-head in ``investigate.py``):

  * ``return``      — raw net bar return. The blunt baseline.
  * ``logret``      — net log return. Compounding-consistent; the env default.
  * ``diff_sharpe`` — Moody & Saffell (1998) Differential Sharpe Ratio: an online estimate of
                      ∂Sharpe/∂t. Rewards improving risk-adjusted return, per step.
  * ``active``      — net return minus the equal-weight benchmark return. Rewards beating a
                      naive diversified hold *causally* (external benchmark, not a self-twin).
  * ``active_dsr``  — Differential Sharpe of the *active* (excess-over-equal-weight) series.
                      The principled successor to ``timing_sortino``: risk-adjusted edge over an
                      honest benchmark, computed online with no full-window leakage.
"""
from __future__ import annotations

import math

ETA = 0.04          # EMA decay for the differential-Sharpe moments (Moody & Saffell)


def _dsr(value: float, state: dict, key: str) -> float:
    """One online Differential Sharpe Ratio update for a scalar ``value``.

    Maintains EMAs A=E[r], B=E[r²] under ``state[key]`` and returns the differential
    D_t = (B·ΔA − ½·A·ΔB) / (B − A²)^{3/2}. Returns 0 until variance is defined (first bars),
    so a degenerate constant series scores 0 rather than NaN.
    """
    a_key, b_key = f"{key}_A", f"{key}_B"
    A = state.get(a_key, 0.0)
    B = state.get(b_key, 0.0)
    dA = value - A
    dB = value * value - B
    var = B - A * A
    D = (B * dA - 0.5 * A * dB) / (var ** 1.5) if var > 1e-12 else 0.0
    state[a_key] = A + ETA * dA
    state[b_key] = B + ETA * dB
    return float(D)


def reward_return(env) -> float:
    return float(env.returns_memory[-1])


def reward_logret(env) -> float:
    return math.log(max(1e-9, 1.0 + env.returns_memory[-1]))


def reward_diff_sharpe(env) -> float:
    return _dsr(env.returns_memory[-1], env.reward_state, "own")


def reward_active(env) -> float:
    return float(env.returns_memory[-1] - env.bench_returns_memory[-1])


def reward_active_dsr(env) -> float:
    active = env.returns_memory[-1] - env.bench_returns_memory[-1]
    return _dsr(active, env.reward_state, "active")


#: name -> per-step reward callable. The investigation harness sweeps these.
REWARDS: dict[str, callable] = {
    "return": reward_return,
    "logret": reward_logret,
    "diff_sharpe": reward_diff_sharpe,
    "active": reward_active,
    "active_dsr": reward_active_dsr,
}
DEFAULT_REWARD = "logret"


def get_reward(name: str):
    """Look up a reward callable by name (raises with the valid set on a typo)."""
    if name not in REWARDS:
        raise KeyError(f"unknown reward {name!r}; choices: {sorted(REWARDS)}")
    return REWARDS[name]
