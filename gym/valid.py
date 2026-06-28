"""valid.py — statistical validation of OOS results.

Two tests:
  mcpt(excess_returns, ...)   — Monte-Carlo Permutation Test (sign-flip).
                                 H0: excess returns are symmetric around 0.
                                 p-value: fraction of permutations beating observed Sharpe.

  runs_test(trade_returns)    — Wald–Wolfowitz runs test for serial independence.
                                 H0: returns are IID (no autocorrelation / streaks).
                                 Returns z-statistic and p-value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from gym.backtest import _sharpe


# ─────────────────────────────────────────── result types

@dataclass
class MCPTResult:
    observed_sharpe: float
    p_value: float              # fraction of permutations ≥ observed (one-tailed)
    n_perms: int
    significant: bool           # p_value < alpha
    alpha: float = 0.05


@dataclass
class RunsTestResult:
    n_runs: int
    expected_runs: float
    z_stat: float
    p_value: float              # two-tailed
    significant: bool           # p < alpha (serial dependence detected)
    alpha: float = 0.05


# ─────────────────────────────────────────── MCPT

def mcpt(
    excess_returns: np.ndarray,
    n_perms: int = 10_000,
    metric_fn=None,
    seed: int = 42,
    alpha: float = 0.05,
) -> MCPTResult:
    """Monte-Carlo Permutation Test on pooled OOS excess returns.

    Sign-flip permutation: each permutation randomly flips the sign of each bar's
    excess return (H0: excess is symmetric / zero-mean), recomputes the Sharpe,
    and counts how often it exceeds the observed value.

    Parameters
    ----------
    excess_returns : 1-D array of per-bar (strategy_return − bh_return) values.
    n_perms        : number of sign-flip permutations.
    metric_fn      : callable(returns) -> float; defaults to annualised Sharpe.
    seed           : RNG seed for reproducibility.
    alpha          : significance level for the `significant` flag.
    """
    if metric_fn is None:
        metric_fn = _sharpe

    x = np.asarray(excess_returns, dtype=np.float64).ravel()
    if len(x) < 2:
        return MCPTResult(observed_sharpe=float("nan"), p_value=float("nan"),
                          n_perms=n_perms, significant=False, alpha=alpha)

    observed = metric_fn(x)
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_perms):
        signs  = rng.choice([-1.0, 1.0], size=len(x))
        perm_s = metric_fn(x * signs)
        if perm_s >= observed:
            count_ge += 1

    p = count_ge / n_perms
    return MCPTResult(
        observed_sharpe=float(observed),
        p_value=float(p),
        n_perms=n_perms,
        significant=(p < alpha),
        alpha=alpha,
    )


# ─────────────────────────────────────────── Runs test (Wald–Wolfowitz)

def runs_test(
    trade_returns: np.ndarray,
    alpha: float = 0.05,
) -> RunsTestResult:
    """Wald–Wolfowitz runs test for serial randomness.

    H0: the sequence of positive/negative returns is IID (no streaks / dependence).
    Classifies each return as + or − relative to the median, counts runs, computes
    the Z approximation.

    Parameters
    ----------
    trade_returns : 1-D array of per-trade (or per-bar) return values.
    alpha         : significance level for the `significant` flag.
    """
    x = np.asarray(trade_returns, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return RunsTestResult(n_runs=0, expected_runs=float("nan"), z_stat=float("nan"),
                              p_value=float("nan"), significant=False, alpha=alpha)

    median = np.median(x)
    signs  = np.where(x > median, 1, -1)   # ties ignored via count below
    # count n+ (n1) and n- (n2)
    n1 = int(np.sum(signs == 1))
    n2 = int(np.sum(signs == -1))
    n  = n1 + n2

    if n1 == 0 or n2 == 0:
        # degenerate: all same sign
        return RunsTestResult(n_runs=1, expected_runs=float("nan"), z_stat=float("nan"),
                              p_value=float("nan"), significant=False, alpha=alpha)

    # count runs
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))

    # expected runs and variance under H0
    mu_r  = (2.0 * n1 * n2) / n + 1.0
    var_r = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n * n * (n - 1))

    if var_r <= 0:
        return RunsTestResult(n_runs=runs, expected_runs=mu_r, z_stat=float("nan"),
                              p_value=float("nan"), significant=False, alpha=alpha)

    z = (runs - mu_r) / np.sqrt(var_r)
    p = 2.0 * scipy_stats.norm.sf(abs(z))   # two-tailed

    return RunsTestResult(
        n_runs=runs,
        expected_runs=float(mu_r),
        z_stat=float(z),
        p_value=float(p),
        significant=(p < alpha),
        alpha=alpha,
    )
