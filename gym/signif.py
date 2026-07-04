"""signif.py — the out-of-sample significance gate for the multi-asset portfolio (PRD FR-11).

Self-contained (numpy + scipy only) so it works from every entry point (CLI, control panel,
harness) without the package/flat import dance. ``gym/valid.py`` holds the canonical
single-ticker-equity versions; this is the portfolio-return-series counterpart.

Two complementary tests, run on the champion's out-of-sample **active** return series
(portfolio return minus the equal-weight benchmark, per bar):

  * ``mcpt_sharpe`` — Monte-Carlo Permutation Test by sign-flip. H0: active returns are
    symmetric about 0 (no directional edge). p = fraction of sign-flipped permutations whose
    Sharpe meets or beats the observed one. A small p means the realised edge is unlikely under
    "no edge".
  * ``mcpt_sharpe`` with ``block_len > 1`` — **block** sign-flip: one Rademacher sign per
    contiguous block instead of per bar. When returns are serially dependent, per-bar flips
    build an unrealistically easy null (they destroy the dependence the real series has);
    flipping whole blocks preserves within-block autocorrelation, so the null is harder to
    beat and the p-value honest under dependence. The gate runs both and requires both.
  * ``runs_test`` — Wald-Wolfowitz runs test for serial independence. H0: the sign sequence is
    IID. A significant result means streakiness (the bars aren't independent), which is exactly
    when the block MCPT matters more than the per-bar one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


def _sharpe(r: np.ndarray, ppy: int = 252) -> float:
    r = np.asarray(r, dtype=np.float64)
    if len(r) < 2:
        return float("nan")
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(ppy)) if sd > 0 else float("nan")


@dataclass
class GateResult:
    n: int
    observed_sharpe: float
    mcpt_p: float
    runs_z: float
    runs_p: float
    alpha: float = 0.05
    block_mcpt_p: float = float("nan")

    @property
    def edge_significant(self) -> bool:
        """A directional edge that clears alpha on the per-bar MCPT AND (when available)
        the dependence-robust block MCPT — the stricter of the two nulls decides."""
        iid_ok = np.isfinite(self.mcpt_p) and self.mcpt_p < self.alpha
        block_ok = (not np.isfinite(self.block_mcpt_p)) or self.block_mcpt_p < self.alpha
        return iid_ok and block_ok and self.observed_sharpe > 0

    @property
    def serially_dependent(self) -> bool:
        return np.isfinite(self.runs_p) and self.runs_p < self.alpha

    def summary(self) -> str:
        if self.n < 2:
            return "significance gate: too few bars."
        verdict = "SIGNIFICANT edge" if self.edge_significant else "not significant"
        dep = " (serially dependent — block p is the binding one)" if self.serially_dependent \
            else ""
        blk = f"  block p={self.block_mcpt_p:.4f}" if np.isfinite(self.block_mcpt_p) else ""
        return (f"gate: Sharpe={self.observed_sharpe:+.3f}  MCPT p={self.mcpt_p:.4f}{blk}  "
                f"runs p={self.runs_p:.4f} -> {verdict}{dep}")


def mcpt_sharpe(active_returns, n_perms: int = 10_000, seed: int = 42, ppy: int = 252,
                block_len: int = 1) -> tuple[float, float]:
    """Sign-flip permutation test on the Sharpe of an active-return series.

    ``block_len=1`` flips each bar independently (H0: symmetric IID noise). ``block_len>1``
    draws ONE sign per contiguous block, preserving within-block serial dependence in the
    null (the honest test when the runs test says the series is streaky).

    Returns (observed_sharpe, p_value). p = (1 + #{perm Sharpe >= observed}) / (1 + n_perms),
    the unbiased permutation-test estimator (never 0).
    """
    r = np.asarray(active_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    obs = _sharpe(r, ppy)
    if not np.isfinite(obs) or len(r) < 2:
        return obs, float("nan")
    rng = np.random.default_rng(seed)
    n = len(r)
    if block_len <= 1:
        signs = rng.choice([-1.0, 1.0], size=(n_perms, n))
    else:
        n_blocks = -(-n // block_len)                  # ceil
        block_signs = rng.choice([-1.0, 1.0], size=(n_perms, n_blocks))
        signs = np.repeat(block_signs, block_len, axis=1)[:, :n]
    perm = signs * r
    mean = perm.mean(axis=1)
    sd = perm.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        perm_sharpe = np.where(sd > 0, mean / sd * np.sqrt(ppy), -np.inf)
    p = (1 + int(np.sum(perm_sharpe >= obs))) / (1 + n_perms)
    return obs, float(p)


def default_block_len(n: int) -> int:
    """Rule-of-thumb block length for the dependence-robust MCPT: ~n^(1/3), min 5."""
    return max(5, int(round(n ** (1.0 / 3.0))))


def runs_test(returns) -> tuple[float, float]:
    """Wald-Wolfowitz runs test on the sign sequence of `returns`. Returns (z, two-tailed p)."""
    r = np.asarray(returns, dtype=np.float64)
    signs = np.sign(r[r != 0])
    n_pos = int(np.sum(signs > 0))
    n_neg = int(np.sum(signs < 0))
    n = n_pos + n_neg
    if n_pos == 0 or n_neg == 0 or n < 2:
        return float("nan"), float("nan")
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    exp = 1 + (2 * n_pos * n_neg) / n
    var = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n * n * (n - 1))
    if var <= 0:
        return float("nan"), float("nan")
    z = (runs - exp) / np.sqrt(var)
    p = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
    return float(z), p


def gate(active_returns, n_perms: int = 10_000, seed: int = 42, alpha: float = 0.05,
         ppy: int = 252) -> GateResult:
    """Run all three tests on an active-return series and bundle the verdict."""
    r = np.asarray(active_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    obs, p = mcpt_sharpe(r, n_perms=n_perms, seed=seed, ppy=ppy)
    _, bp = mcpt_sharpe(r, n_perms=n_perms, seed=seed + 1, ppy=ppy,
                        block_len=default_block_len(len(r)))
    z, rp = runs_test(r)
    return GateResult(n=len(r), observed_sharpe=obs, mcpt_p=p, runs_z=z, runs_p=rp,
                      alpha=alpha, block_mcpt_p=bp)
