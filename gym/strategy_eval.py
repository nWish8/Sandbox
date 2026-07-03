"""strategy_eval.py — the formal backtester + strategy promotion flow (Vision M4).

Once a policy shows promise in the RL lab, "promotion" runs it through a stricter,
independent evaluation than the training env applies:

  * ``replay_weights`` — a **vectorized re-implementation** of the portfolio accounting
    (weights × next-bar returns − costs). With the same cost assumptions it must reproduce
    ``PortfolioEnv``'s equity to float precision (unit-tested), so the env and the
    backtester cross-validate each other.
  * ``CostModel`` — realistic frictions the training env doesn't model: commission per
    unit turnover, extra **slippage** bps on traded notional, and **execution latency**
    (weights decided at bar t take effect ``delay_bars`` later).
  * ``promote`` — loads a recorded run's final model (runlog), rolls it deterministically
    over the held-out **test** split, applies the cost model, and emits a formal report:
    Sharpe / Sortino / Calmar / active_sharpe, max drawdown, win rates, turnover, plus the
    equity and drawdown series. Saved to ``gym/reports/promotion_*.json``.

The module is deliberately independent of FinRL: numpy/pandas + stats.py only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stats import portfolio_stats

HERE = Path(__file__).resolve().parent
TRADING_DAYS = 252


@dataclass
class CostModel:
    cost_pct: float = 0.001        # commission per unit turnover (as used in training)
    slippage_bps: float = 5.0      # extra slippage on traded notional
    delay_bars: int = 0            # bars between deciding weights and holding them

    @property
    def friction(self) -> float:
        return self.cost_pct + self.slippage_bps / 1e4


# ─────────────────────────────────────────── vectorized accounting

def replay_weights(weights: np.ndarray, asset_rets: np.ndarray,
                   cost: CostModel | None = None) -> pd.DataFrame:
    """Re-run the portfolio accounting for a weight path over per-asset returns.

    Row convention (matches ``PortfolioEnv.portfolio_history``): row k's ``weights`` were
    held over bar k−1→k and its ``asset_rets`` are the per-asset returns of that same bar
    (row 0 is the reset row: zero returns, initial weights). ``delay_bars`` shifts the
    weight path so decisions take effect later; the initial (equal-weight) allocation
    fills the gap.
    """
    cost = cost if cost is not None else CostModel()
    w = np.asarray(weights, dtype=np.float64)
    a = np.asarray(asset_rets, dtype=np.float64)
    if w.shape != a.shape:
        raise ValueError(f"weights {w.shape} and asset_rets {a.shape} must match")
    T, N = w.shape

    if cost.delay_bars > 0:
        d = int(cost.delay_bars)
        w = np.vstack([np.repeat(w[[0]], d, axis=0), w[:-d]])

    gross = (w * a).sum(axis=1)
    prev = np.vstack([w[[0]], w[:-1]])
    turnover = np.abs(w - prev).sum(axis=1)
    net = gross - cost.friction * turnover
    net[0] = 0.0                                          # reset row earns nothing
    return pd.DataFrame({"ret": net, "gross": gross, "turnover": turnover,
                         "equity": np.cumprod(1.0 + net)})


def asset_returns_from_history(df: pd.DataFrame, tics: list[str]) -> np.ndarray:
    """(T, N) per-asset close-to-close returns aligned to history rows (row 0 = 0)."""
    closes = (df.pivot_table(index="date", columns="tic", values="close")
              .sort_index()[tics].to_numpy())
    rets = np.zeros_like(closes)
    rets[1:] = closes[1:] / closes[:-1] - 1.0
    return rets


# ─────────────────────────────────────────── the formal report

def evaluate(weights: np.ndarray, asset_rets: np.ndarray, cost: CostModel | None = None,
             periods_per_year: int = TRADING_DAYS) -> dict:
    """Formal backtest report for a weight path under a cost model."""
    cost = cost if cost is not None else CostModel()
    res = replay_weights(weights, asset_rets, cost)
    bench = asset_rets.mean(axis=1)                       # equal-weight, same bars
    bench[0] = 0.0
    r = res["ret"].to_numpy()[1:]
    b = bench[1:]
    stats = portfolio_stats(r, b, periods_per_year=periods_per_year)

    equity = res["equity"].to_numpy()
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    active = r - b
    report = {
        "cost_model": asdict(cost),
        "stats": stats,
        "win_rate": float((r > 0).mean()),
        "active_win_rate": float((active > 0).mean()),
        "avg_turnover": float(res["turnover"].to_numpy()[1:].mean()),
        "total_friction_paid": float((cost.friction * res["turnover"]).sum()),
        "equity": equity.tolist(),
        "drawdown": drawdown.tolist(),
        "bench_equity": np.cumprod(1.0 + bench).tolist(),
    }
    return report


def format_report(report: dict, title: str = "promotion") -> str:
    s = report["stats"]
    c = report["cost_model"]
    lines = [
        f"── formal backtest [{title}] "
        f"(commission {c['cost_pct']:.4f}/turnover, slippage {c['slippage_bps']:.0f}bps, "
        f"delay {c['delay_bars']} bar) " + "─" * 10,
        f"  bars {s.get('n_bars', 0)}   total return {s.get('total_return', float('nan')):+.3f}"
        f"   vs equal-weight {s.get('bench_total_return', float('nan')):+.3f}"
        f"   (excess {s.get('excess_total', float('nan')):+.3f})",
        f"  sharpe {s.get('sharpe', float('nan')):+.3f}   sortino {s.get('sortino', float('nan')):+.3f}"
        f"   calmar {s.get('calmar', float('nan')):+.3f}   ACTIVE sharpe "
        f"{s.get('active_sharpe', float('nan')):+.3f}",
        f"  max drawdown {s.get('max_drawdown', float('nan')):+.3f}"
        f"   win rate {report['win_rate']:.1%}   active win rate {report['active_win_rate']:.1%}",
        f"  avg turnover/bar {report['avg_turnover']:.4f}"
        f"   total friction paid {report['total_friction_paid']:.4f}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────── promotion from a recorded run

def promote(run_id: str, cost: CostModel | None = None, split: str = "test",
            runs_dir: Path | str | None = None, log=print) -> dict:
    """Load a recorded run's final model, roll it over the held-out split, apply the cost
    model, and return (and persist) the formal report."""
    import stable_baselines3 as sb3

    from investigate import split_val_test
    from pipeline import FinRLConfig
    from portfolio import prepare_portfolio_data, run_portfolio
    from runlog import RunRecord

    cost = cost if cost is not None else CostModel()
    rec = RunRecord.load(run_id, runs_dir=runs_dir)
    m = rec.manifest
    cfg_fields = {k: v for k, v in m["config"].items()
                  if k in FinRLConfig.__dataclass_fields__}
    cfg_fields["indicators"] = tuple(cfg_fields.get("indicators", ()))
    cfg = FinRLConfig(**cfg_fields)

    model_path = rec.model_path()
    if model_path is None:
        raise FileNotFoundError(f"run {run_id} has no saved model.zip")
    model = getattr(sb3, m["algo"].upper()).load(str(model_path), device="cpu")

    log(f"[promote] run={run_id} algo={m['algo']} reward={m['reward']} split={split}")
    train_df, trade_df = prepare_portfolio_data(cfg, lookback=m["lookback"], log=log)
    val_df, test_df = split_val_test(trade_df)
    eval_df = {"train": train_df, "val": val_df, "test": test_df}[split]

    hist = run_portfolio(model, eval_df, cfg, reward=m["reward"], lookback=m["lookback"])
    tics = sorted(eval_df["tic"].unique().tolist())
    weights = hist[[f"w_{t}" for t in tics]].to_numpy()
    asset_rets = asset_returns_from_history(eval_df, tics)

    report = evaluate(weights, asset_rets, cost)
    report["run_id"] = run_id
    report["split"] = split
    report["algo"], report["reward"] = m["algo"], m["reward"]
    report["dates"] = hist["date"].astype(str).tolist()

    out_dir = HERE / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"promotion_{run_id}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log(f"[promote] report -> {out}")
    return report


# ─────────────────────────────────────────── optional bt cross-check

def bt_crosscheck(weights: np.ndarray, asset_rets: np.ndarray, tics: list[str],
                  dates: pd.DatetimeIndex | None = None) -> dict | None:
    """Independent equity check through the `bt` engine (FinRL-X's backtester).

    Feeds the same weight path and a price series rebuilt from the asset returns into
    ``bt.algos.WeighTarget`` with zero commissions; returns bt's total return, or None if
    bt is unavailable. Convention note: bt applies a weight row at that date's close, so
    it earns the *next* bar — the same alignment as ``replay_weights`` after shifting our
    held-over-bar-k rows back one bar.
    """
    try:
        import bt
    except ImportError:
        return None
    T = len(weights)
    idx = dates if dates is not None else pd.date_range("2020-01-01", periods=T, freq="B")
    prices = pd.DataFrame(np.cumprod(1.0 + asset_rets, axis=0) * 100.0,
                          index=idx, columns=tics)
    # our row k weights were held over bar k-1→k ⇒ bt must set them at bar k-1's close
    w = pd.DataFrame(weights, index=idx, columns=tics).shift(-1).dropna()
    strat = bt.Strategy("check", [bt.algos.WeighTarget(w), bt.algos.Rebalance()])
    res = bt.run(bt.Backtest(strat, prices, commissions=lambda q, p: 0.0,
                             integer_positions=False))
    total = float(res.stats.loc["total_return"].iloc[0])
    return {"bt_total_return": total}
