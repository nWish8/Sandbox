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
  * ``runs_test`` — Wald-Wolfowitz runs test for serial independence. H0: the sign sequence is
    IID. A significant result means streakiness (the bars aren't independent), which tempers how
    much to trust the MCPT p-value.
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

    @property
    def edge_significant(self) -> bool:
        """A directional edge that clears alpha (one-tailed sign-flip MCPT)."""
        return np.isfinite(self.mcpt_p) and self.mcpt_p < self.alpha and self.observed_sharpe > 0

    @property
    def serially_dependent(self) -> bool:
        return np.isfinite(self.runs_p) and self.runs_p < self.alpha

    def summary(self) -> str:
        if self.n < 2:
            return "significance gate: too few bars."
        verdict = "SIGNIFICANT edge" if self.edge_significant else "not significant"
        dep = " (serially dependent — treat p with caution)" if self.serially_dependent else ""
        return (f"gate: Sharpe={self.observed_sharpe:+.3f}  MCPT p={self.mcpt_p:.4f}  "
                f"runs p={self.runs_p:.4f} -> {verdict}{dep}")


def mcpt_sharpe(active_returns, n_perms: int = 10_000, seed: int = 42, ppy: int = 252
                ) -> tuple[float, float]:
    """Sign-flip permutation test on the Sharpe of an active-return series.

    Returns (observed_sharpe, p_value). p = (1 + #{perm Sharpe >= observed}) / (1 + n_perms),
    the unbiased permutation-test estimator (never 0).
    """
    r = np.asarray(active_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    obs = _sharpe(r, ppy)
    if not np.isfinite(obs) or len(r) < 2:
        return obs, float("nan")
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perms, len(r)))
    perm = signs * r                                   # sign-flip each bar
    mean = perm.mean(axis=1)
    sd = perm.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        perm_sharpe = np.where(sd > 0, mean / sd * np.sqrt(ppy), -np.inf)
    p = (1 + int(np.sum(perm_sharpe >= obs))) / (1 + n_perms)
    return obs, float(p)


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
    """Run both tests on an active-return series and bundle the verdict."""
    r = np.asarray(active_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    obs, p = mcpt_sharpe(r, n_perms=n_perms, seed=seed, ppy=ppy)
    z, rp = runs_test(r)
    return GateResult(n=len(r), observed_sharpe=obs, mcpt_p=p, runs_z=z, runs_p=rp, alpha=alpha)
