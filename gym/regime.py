"""regime.py — causal market-regime labelling + per-regime evaluation (PRD FR-6).

The PRD's #1 lever is regime coverage: the previous "no edge" result came from a single
(bull) tape, where timing adds nothing. This module lets us *label* each held-out bar as
**bull / bear / choppy** and break the agent's out-of-sample performance down by regime — so a
verdict like "no edge" can be qualified ("…on bull bars; positive on bear bars") instead of
hiding a regime effect.

Labelling is **causal**: a bar's regime is decided from a trailing window of the equal-weight
benchmark only (no future bars). The benchmark is the same equal-weight index the
``PortfolioEnv`` scores against, so labels align 1:1 with the evaluation return series.

Definition: fit a least-squares line to log-benchmark over the trailing window; annualise its
slope. ``slope_ann >  trend_thresh`` → bull; ``< -trend_thresh`` → bear; otherwise → choppy.
Simple and transparent — the labelling rule is fixed before any results are seen.
"""
from __future__ import annotations

import numpy as np

from stats import portfolio_stats

TRADING_DAYS = 252
REGIMES = ("bull", "bear", "choppy")


def label_regimes(bench_returns, window: int = 60, trend_thresh: float = 0.10,
                  periods_per_year: int = TRADING_DAYS) -> np.ndarray:
    """Label each bar bull/bear/choppy from a trailing window of the benchmark (causal).

    ``bench_returns`` is the per-bar equal-weight benchmark return. Bars before ``min_seg`` of
    history are labelled 'choppy' (insufficient trend evidence). Returns an object array of
    labels aligned to ``bench_returns``.
    """
    b = np.asarray(bench_returns, dtype=np.float64)
    logp = np.log(np.cumprod(1.0 + b))
    labels = np.full(len(b), "choppy", dtype=object)
    min_seg = 5
    for t in range(len(b)):
        lo = max(0, t - window + 1)
        seg = logp[lo:t + 1]
        if len(seg) < min_seg:
            continue
        slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0])
        ann = slope * periods_per_year
        labels[t] = "bull" if ann > trend_thresh else "bear" if ann < -trend_thresh else "choppy"
    return labels


def stats_by_regime(returns, bench_returns, labels=None, *, window: int = 60,
                    trend_thresh: float = 0.10, periods_per_year: int = TRADING_DAYS
                    ) -> dict[str, dict]:
    """Split a portfolio return series by regime and compute ``portfolio_stats`` for each.

    If ``labels`` is None they're derived causally from ``bench_returns``. Regimes with no bars
    are omitted. Each value also carries ``n_bars`` so thin regimes can be discounted.
    """
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(bench_returns, dtype=np.float64)
    if labels is None:
        labels = label_regimes(b, window=window, trend_thresh=trend_thresh,
                               periods_per_year=periods_per_year)
    labels = np.asarray(labels, dtype=object)
    out: dict[str, dict] = {}
    for reg in REGIMES:
        mask = labels == reg
        if not mask.any():
            continue
        out[reg] = portfolio_stats(r[mask], b[mask], periods_per_year=periods_per_year)
    return out


def format_regime_table(by_regime: dict[str, dict],
                        metrics: tuple[str, ...] = ("n_bars", "active_sharpe", "sharpe",
                                                    "total_return")) -> str:
    """Render the per-regime breakdown as a compact table."""
    hdr = f"{'regime':>8} | " + " | ".join(f"{m:>13}" for m in metrics)
    lines = [hdr, "-" * len(hdr)]
    for reg in REGIMES:
        if reg not in by_regime:
            continue
        s = by_regime[reg]
        cells = []
        for m in metrics:
            v = s.get(m, float("nan"))
            cells.append(f"{int(v):>13}" if m == "n_bars" and np.isfinite(v)
                         else (f"{v:>13.3f}" if v is not None and np.isfinite(v) else f"{'n/a':>13}"))
        lines.append(f"{reg:>8} | " + " | ".join(cells))
    return "\n".join(lines)
