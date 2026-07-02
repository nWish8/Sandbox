"""investigate.py — the reward investigation (PRD FR-4).

The central v2 experiment: *which training reward actually selects for generalizable edge?*
For each candidate reward in ``rewards.REWARDS`` we train one PPO policy on the train split,
then evaluate it on a held-out **validation** and **test** slice of the out-of-sample period,
scoring with the corrected, leakage-free ``stats.portfolio_stats``. Candidates are ranked by
**validation** ``active_sharpe`` (information ratio vs the equal-weight benchmark); the test
column is reported but never used for selection — that is the whole anti-overfitting point.

The honest verdict lives in the table: if no reward yields a positive *test* ``active_sharpe``,
that is a rigorous negative, recorded as plainly as a positive would be.

    from pipeline import FinRLConfig
    from investigate import investigate, format_table, append_to_research_log
    rows = investigate(FinRLConfig(tickers=[...]), timesteps=20_000)
    print(format_table(rows))
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import FinRLConfig
from portfolio import prepare_portfolio_data, run_portfolio, train_portfolio
from regime import format_regime_table, stats_by_regime
from rewards import REWARDS
from stats import PORTFOLIO_OBJECTIVES, portfolio_stats

HERE = Path(__file__).resolve().parent
TRADING_DAYS = 252


# ─────────────────────────────────────────── chronological holdout split

def _slice_days(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Slice a day-ordinal-indexed long df to day range [lo, hi) and re-index from 0."""
    sub = df[(df.index >= lo) & (df.index < hi)].copy()
    sub.index = sub["date"].factorize()[0]
    return sub


def split_val_test(trade_df: pd.DataFrame, val_frac: float = 0.5
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the out-of-sample period chronologically into validation (earlier) and test
    (later) — no shuffling, no overlap, so selection on val can't peek at test."""
    days = trade_df.index.unique()
    cut = int(len(days) * val_frac)
    cut = min(max(cut, 1), len(days) - 1)
    return (_slice_days(trade_df, days[0], days[cut]),
            _slice_days(trade_df, days[cut], days[-1] + 1))


def _eval(model, df, cfg, reward, lookback) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return (stats, portfolio_returns, benchmark_returns) for a deterministic rollout."""
    hist = run_portfolio(model, df, cfg, reward=reward, lookback=lookback)
    ret = hist["ret"].to_numpy()[1:]
    bench = hist["bench_ret"].to_numpy()[1:]
    stats = portfolio_stats(ret, bench, periods_per_year=TRADING_DAYS)
    return stats, ret, bench


def _rank_key(stats: dict, metric: str = "active_sharpe") -> float:
    v = stats.get(metric, float("nan"))
    return v if (v is not None and np.isfinite(v)) else float("-inf")


# ─────────────────────────────────────────── the investigation

def investigate(cfg: FinRLConfig, reward_names: list[str] | None = None, *,
                timesteps: int = 20_000, seed: int = 42, lookback: int = 60,
                device: str = "cpu", val_frac: float = 0.5, stop=None, log=print) -> list[dict]:
    """Train one PPO agent per candidate reward; score each on held-out val + test.

    Returns rows ``[{reward, val: stats, test: stats}]`` sorted by validation ``active_sharpe``
    (best first). Selection uses validation only; test is reported, never selected on.
    """
    reward_names = reward_names or list(REWARDS)
    train_df, trade_df = prepare_portfolio_data(cfg, lookback=lookback, log=log)
    val_df, test_df = split_val_test(trade_df, val_frac)
    log(f"[investigate] train_days={train_df.index.nunique()} "
        f"val_days={val_df.index.nunique()} test_days={test_df.index.nunique()}")

    rows: list[dict] = []
    for name in reward_names:
        if stop is not None and stop():
            log("[investigate] stopped"); break
        model = train_portfolio(train_df, cfg, reward=name, timesteps=timesteps, seed=seed,
                                lookback=lookback, device=device, stop=stop, log=log)
        val_s, _, _ = _eval(model, val_df, cfg, name, lookback)
        test_s, test_ret, test_bench = _eval(model, test_df, cfg, name, lookback)
        log(f"[investigate] {name:>12}  val active_sharpe={_rank_key(val_s):+.3f}  "
            f"test active_sharpe={_rank_key(test_s):+.3f}")
        rows.append({"reward": name, "val": val_s, "test": test_s,
                     "test_ret": test_ret, "test_bench": test_bench})

    rows.sort(key=lambda r: _rank_key(r["val"]), reverse=True)
    return rows


def _test_active(row: dict) -> np.ndarray:
    return np.asarray(row["test_ret"]) - np.asarray(row["test_bench"])


def champion_gate(rows: list[dict], n_perms: int = 10_000, seed: int = 42):
    """Run the MCPT/runs-test significance gate on the val-selected champion's TEST active
    returns. Returns a ``signif.GateResult`` (or None if unavailable)."""
    from signif import gate
    if not rows or "test_ret" not in rows[0]:
        return None
    return gate(_test_active(rows[0]), n_perms=n_perms, seed=seed)


# ─────────────────────────────────────────── reporting

def format_table(rows: list[dict], metrics: tuple[str, ...] = PORTFOLIO_OBJECTIVES) -> str:
    """Render the investigation result as a fixed-width val/test comparison table."""
    hdr = f"{'reward':>12} | " + " | ".join(f"val_{m[:9]:>9} test_{m[:9]:>9}" for m in metrics)
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        cells = []
        for m in metrics:
            v = r["val"].get(m, float("nan"))
            t = r["test"].get(m, float("nan"))
            cells.append(f"{_fmt(v):>13} {_fmt(t):>14}")
        lines.append(f"{r['reward']:>12} | " + " | ".join(cells))
    return "\n".join(lines)


def _fmt(x) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.3f}"


def verdict(rows: list[dict]) -> str:
    """One-line honest read: did the val-selected winner show positive test edge?"""
    if not rows:
        return "no results."
    best = rows[0]
    test_edge = best["test"].get("active_sharpe", float("nan"))
    name = best["reward"]
    if np.isfinite(test_edge) and test_edge > 0:
        return (f"val-selected reward '{name}' shows POSITIVE test active_sharpe "
                f"{test_edge:+.3f} -> candidate edge (confirm with the significance gate).")
    return (f"val-selected reward '{name}' has test active_sharpe {_fmt(test_edge)} -> "
            f"no out-of-sample edge over equal-weight. Honest negative.")


def append_to_research_log(rows: list[dict], cfg: FinRLConfig, *, timesteps: int, seed: int,
                           path: Path | None = None) -> None:
    """Append a dated investigation entry to RESEARCH_LOG.md (newest entries go near the top
    of the log by convention, but we append a clearly-dated section to keep it simple)."""
    path = path or (HERE / "RESEARCH_LOG.md")
    g = champion_gate(rows)
    champ = rows[0] if rows else None
    by_regime = (stats_by_regime(champ["test_ret"], champ["test_bench"]) if champ else {})
    manifest = write_manifest(rows, cfg, timesteps=timesteps, seed=seed)
    entry = [
        f"\n## {date.today().isoformat()} — reward investigation",
        f"\nUniverse: {len(cfg.tickers)} tickers · train {cfg.train_start}..{cfg.train_end} · "
        f"trade {cfg.trade_start}..{cfg.trade_end} · PPO {timesteps} steps · seed {seed}.",
        "\nRanked by **validation** `active_sharpe` (selection never touches test):",
        "\n```",
        format_table(rows),
        "```",
        f"\n**Verdict:** {verdict(rows)}",
        f"\n**Significance gate** (champion test active returns): {g.summary() if g else 'n/a'}",
        "\n**Champion test performance by market regime** (causal bull/bear/choppy labelling):",
        "\n```",
        format_regime_table(by_regime) if by_regime else "n/a",
        "```",
        f"\nRun manifest (reproducibility): `{manifest.name}`\n" if manifest else "\n",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))


def write_manifest(rows: list[dict], cfg: FinRLConfig, *, timesteps: int, seed: int,
                   reports_dir: Path | None = None) -> Path | None:
    """Serialise the full run config + seed + reward ranking to a JSON manifest so any result
    is reproducible (PRD FR-9). Returns the written path."""
    reports_dir = reports_dir or (HERE / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    cfg_d = asdict(cfg)
    cfg_d["indicators"] = list(cfg_d.get("indicators", []))
    ranking = [{"reward": r["reward"],
                "val_active_sharpe": r["val"].get("active_sharpe"),
                "test_active_sharpe": r["test"].get("active_sharpe")} for r in rows]
    g = champion_gate(rows)
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": cfg_d, "timesteps": timesteps, "seed": seed,
        "ranking": ranking, "champion": rows[0]["reward"] if rows else None,
        "gate": ({"observed_sharpe": g.observed_sharpe, "mcpt_p": g.mcpt_p,
                  "runs_p": g.runs_p, "edge_significant": g.edge_significant} if g else None),
    }
    out = reports_dir / f"investigation_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    return out
