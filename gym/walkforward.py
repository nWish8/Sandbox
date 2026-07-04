"""walkforward.py — rolling retrain → one stitched out-of-sample record.

The workflow shape is qlib's Rolling Retraining loop, adapted to this project: instead of
one train/test split (a single draw), the out-of-sample period is cut into ``n_folds``
consecutive segments; for each fold the agent is retrained on **all data before the
segment** (anchored, expanding window) and rolled deterministically over the segment
only. The per-fold OOS returns are stitched into one continuous active-return series and
judged by the same significance gate as everything else.

Why this is stricter than the single split: every OOS bar is predicted by a model that
never saw it OR anything after it, the OOS record covers the whole trade period rather
than its tail, and fold-to-fold dispersion shows whether a result is one lucky segment.

Fold boundaries reset the book to equal weight (each fold's rollout starts fresh); the
reset row is dropped from the stitched series so no fold contributes a free zero bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import FinRLConfig
from portfolio import add_cov_features, run_portfolio, train_portfolio
from stats import portfolio_stats

TRADING_DAYS = 252


# ─────────────────────────────────────────── folds

def make_folds(n_days: int, n_folds: int, min_train_days: int) -> list[tuple[int, int]]:
    """[(train_end, test_end), ...] over day ordinals: fold k trains on days
    [0, train_end) and tests on [train_end, test_end). Segments tile
    [min_train_days, n_days) evenly; every fold honours the minimum training window."""
    if n_days <= min_train_days:
        raise ValueError(f"need > {min_train_days} days, got {n_days}")
    n_folds = max(1, int(n_folds))
    cuts = np.linspace(min_train_days, n_days, n_folds + 1).round().astype(int)
    return [(int(cuts[k]), int(cuts[k + 1]))
            for k in range(n_folds) if cuts[k + 1] > cuts[k]]


def _slice_days(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    sub = df[(df.index >= lo) & (df.index < hi)].copy()
    sub.index = sub["date"].factorize()[0]
    return sub


# ─────────────────────────────────────────── the rolling investigation

def walkforward_run(cfg: FinRLConfig, *, reward: str = "active_dsr", algo: str = "ppo",
                    timesteps: int = 20_000, n_folds: int = 4, lookback: int = 60,
                    min_train_frac: float = 0.5, seed: int = 42, device: str = "cpu",
                    data: pd.DataFrame | None = None, stop=None, log=print) -> dict:
    """Retrain per fold, stitch the OOS record, gate it. Returns the full report.

    ``data`` (a cov-augmented, day-ordinal-indexed long df) bypasses the download —
    used by tests and by callers that already prepared the panel."""
    from signif import gate

    if data is not None:
        full = data
    else:
        from pipeline import prepare_data
        train_df, trade_df = prepare_data(cfg, log=log)
        full = pd.concat([train_df, trade_df], ignore_index=True)
        full = full.sort_values(["date", "tic"])
        full.index = full["date"].factorize()[0]
        full = add_cov_features(full, lookback=lookback)

    n_days = full.index.nunique()
    folds = make_folds(n_days, n_folds, min_train_days=int(n_days * min_train_frac))
    log(f"[walkforward] {len(folds)} folds over {n_days} days "
        f"(algo={algo} reward={reward} steps={timesteps}/fold)")

    stitched_ret: list[np.ndarray] = []
    stitched_bench: list[np.ndarray] = []
    fold_rows: list[dict] = []
    for k, (tr_end, te_end) in enumerate(folds):
        if stop is not None and stop():
            log("[walkforward] stopped")
            break
        tr = _slice_days(full, 0, tr_end)
        te = _slice_days(full, tr_end, te_end)
        model = train_portfolio(tr, cfg, reward=reward, algo=algo, timesteps=timesteps,
                                seed=seed + k, lookback=lookback, device=device,
                                stop=stop, log=log)
        hist = run_portfolio(model, te, cfg, reward=reward, lookback=lookback)
        r = hist["ret"].to_numpy()[1:]                # drop the fold's reset row
        b = hist["bench_ret"].to_numpy()[1:]
        stitched_ret.append(r)
        stitched_bench.append(b)
        s = portfolio_stats(r, b, periods_per_year=TRADING_DAYS)
        fold_rows.append({"fold": k, "train_days": tr.index.nunique(),
                          "test_days": te.index.nunique(),
                          "test_start": str(te["date"].iloc[0]),
                          "test_end": str(te["date"].iloc[-1]), **{
                              m: s.get(m, float("nan"))
                              for m in ("active_sharpe", "sharpe", "total_return",
                                        "max_drawdown")}})
        log(f"[walkforward] fold {k}: {fold_rows[-1]['test_start'][:10]}"
            f"..{fold_rows[-1]['test_end'][:10]}  "
            f"active_sharpe={fold_rows[-1]['active_sharpe']:+.3f}")

    ret = np.concatenate(stitched_ret) if stitched_ret else np.array([])
    bench = np.concatenate(stitched_bench) if stitched_bench else np.array([])
    stitched_stats = portfolio_stats(ret, bench, periods_per_year=TRADING_DAYS)
    g = gate(ret - bench) if len(ret) > 2 else None
    return {"folds": fold_rows, "stitched": stitched_stats, "gate": g,
            "n_oos_bars": int(len(ret)),
            "ret": ret, "bench": bench,
            "config": {"reward": reward, "algo": algo, "timesteps": timesteps,
                       "n_folds": len(fold_rows), "lookback": lookback, "seed": seed}}


def format_walkforward(report: dict) -> str:
    lines = [f"{'fold':>4} {'test window':<24} {'active_sh':>9} {'sharpe':>8}"
             f" {'total':>8} {'maxDD':>8}"]
    lines.append("-" * len(lines[0]))
    for f in report["folds"]:
        lines.append(f"{f['fold']:>4} {f['test_start'][:10]}..{f['test_end'][:10]:<12}"
                     f" {f['active_sharpe']:>+9.3f} {f['sharpe']:>+8.3f}"
                     f" {f['total_return']:>+8.3f} {f['max_drawdown']:>+8.3f}")
    s = report["stitched"]
    lines.append(f"\nstitched OOS ({report['n_oos_bars']} bars): "
                 f"active_sharpe={s.get('active_sharpe', float('nan')):+.3f}  "
                 f"sharpe={s.get('sharpe', float('nan')):+.3f}  "
                 f"total={s.get('total_return', float('nan')):+.3f}  "
                 f"maxDD={s.get('max_drawdown', float('nan')):+.3f}")
    if report["gate"] is not None:
        lines.append("stitched " + report["gate"].summary())
    per_fold = [f["active_sharpe"] for f in report["folds"]
                if np.isfinite(f["active_sharpe"])]
    if len(per_fold) >= 2:
        pos = sum(1 for a in per_fold if a > 0)
        lines.append(f"fold dispersion: {pos}/{len(per_fold)} folds positive, "
                     f"spread [{min(per_fold):+.3f}, {max(per_fold):+.3f}]")
    return "\n".join(lines)
