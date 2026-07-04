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


def replay_weights_ohlc(weights: np.ndarray, opens: np.ndarray, closes: np.ndarray,
                        cost: CostModel | None = None) -> pd.DataFrame:
    """Open-fill accounting (backtrader-style execution realism).

    The decision made after bar k−1's close executes at bar k's **open**: from
    close_{k−1}→open_k the book still holds the old weights (they earn the overnight gap);
    from open_k→close_k it holds the new ones. Compounded per bar. Turnover cost is charged
    on |Δw| at the open (the pre-gap weight drift is ignored — a documented approximation,
    consistent with the close-fill accounting). With gapless data (open_k == close_{k−1})
    this reproduces ``replay_weights`` exactly (tested)."""
    cost = cost if cost is not None else CostModel()
    w = np.asarray(weights, dtype=np.float64)
    o = np.asarray(opens, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    if not (w.shape == o.shape == c.shape):
        raise ValueError(f"shapes must match: weights {w.shape}, opens {o.shape}, "
                         f"closes {c.shape}")
    if cost.delay_bars > 0:
        d = int(cost.delay_bars)
        w = np.vstack([np.repeat(w[[0]], d, axis=0), w[:-d]])
    prev_w = np.vstack([w[[0]], w[:-1]])
    gap = np.zeros_like(c)
    gap[1:] = o[1:] / c[:-1] - 1.0
    day = np.zeros_like(c)
    day[1:] = c[1:] / o[1:] - 1.0
    gross = (1.0 + (prev_w * gap).sum(axis=1)) * (1.0 + (w * day).sum(axis=1)) - 1.0
    turnover = np.abs(w - prev_w).sum(axis=1)
    net = gross - cost.friction * turnover
    net[0] = 0.0
    return pd.DataFrame({"ret": net, "gross": gross, "turnover": turnover,
                         "equity": np.cumprod(1.0 + net)})


def _pivot(df: pd.DataFrame, tics: list[str], col: str) -> np.ndarray:
    return (df.pivot_table(index="date", columns="tic", values=col)
            .sort_index()[tics].to_numpy())


def asset_returns_from_history(df: pd.DataFrame, tics: list[str]) -> np.ndarray:
    """(T, N) per-asset close-to-close returns aligned to history rows (row 0 = 0)."""
    closes = _pivot(df, tics, "close")
    rets = np.zeros_like(closes)
    rets[1:] = closes[1:] / closes[:-1] - 1.0
    return rets


# ─────────────────────────────────────────── the formal report

def evaluate(weights: np.ndarray, asset_rets: np.ndarray, cost: CostModel | None = None,
             periods_per_year: int = TRADING_DAYS,
             opens: np.ndarray | None = None, closes: np.ndarray | None = None) -> dict:
    """Formal backtest report for a weight path under a cost model.

    Pass ``opens`` + ``closes`` to switch to open-fill execution (``replay_weights_ohlc``);
    the benchmark stays equal-weight close-to-close either way so reports are comparable."""
    cost = cost if cost is not None else CostModel()
    if opens is not None and closes is not None:
        res = replay_weights_ohlc(weights, opens, closes, cost)
        fills = "open"
    else:
        res = replay_weights(weights, asset_rets, cost)
        fills = "close"
    bench = asset_rets.mean(axis=1)                       # equal-weight, same bars
    bench[0] = 0.0
    r = res["ret"].to_numpy()[1:]
    b = bench[1:]
    stats = portfolio_stats(r, b, periods_per_year=periods_per_year)

    equity = res["equity"].to_numpy()
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    active = r - b
    if len(active) and np.max(np.abs(active)) < 1e-12:
        stats["active_sharpe"] = 0.0     # identical to benchmark bar-for-bar: the Sharpe of
        # float rounding noise is meaningless and must not masquerade as (anti-)edge
    wins, losses = float(r[r > 0].sum()), float(-r[r < 0].sum())
    w_arr = np.asarray(weights, dtype=np.float64)

    # drawdown persistence (gs-quant-style risk measures): longest underwater spell
    under = drawdown < -1e-12
    max_dd_duration = 0
    if under.any():
        changes = np.flatnonzero(np.diff(under.astype(np.int8)) != 0) + 1
        bounds = np.concatenate(([0], changes, [len(under)]))
        run_lens, run_starts = np.diff(bounds), bounds[:-1]
        max_dd_duration = int(run_lens[under[run_starts]].max())

    # tail shape (cantaro86's non-normality emphasis): losses are not symmetric-Gaussian
    from scipy import stats as sps
    var_95 = float(np.percentile(r, 5)) if len(r) else float("nan")
    tail = r[r <= var_95] if len(r) else np.array([])
    report = {
        "cost_model": asdict(cost),
        "fills": fills,
        "stats": stats,
        "win_rate": float((r > 0).mean()),
        "active_win_rate": float((active > 0).mean()),
        "profit_factor": (wins / losses) if losses > 0 else float("nan"),
        "hhi_mean": float((w_arr ** 2).sum(axis=1).mean()),   # concentration (1/N = spread)
        "best_bar": float(r.max()) if len(r) else float("nan"),
        "worst_bar": float(r.min()) if len(r) else float("nan"),
        "max_dd_duration": max_dd_duration,
        "pct_time_underwater": float(under.mean()),
        "var_95": var_95,
        "cvar_95": float(tail.mean()) if len(tail) else float("nan"),
        "skew": float(sps.skew(r)) if len(r) > 2 else float("nan"),
        "excess_kurtosis": float(sps.kurtosis(r)) if len(r) > 3 else float("nan"),
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
        f"delay {c['delay_bars']} bar, {report.get('fills', 'close')} fills) " + "─" * 10,
        f"  bars {s.get('n_bars', 0)}   total return {s.get('total_return', float('nan')):+.3f}"
        f"   vs equal-weight {s.get('bench_total_return', float('nan')):+.3f}"
        f"   (excess {s.get('excess_total', float('nan')):+.3f})",
        f"  sharpe {s.get('sharpe', float('nan')):+.3f}   sortino {s.get('sortino', float('nan')):+.3f}"
        f"   calmar {s.get('calmar', float('nan')):+.3f}   ACTIVE sharpe "
        f"{s.get('active_sharpe', float('nan')):+.3f}",
        f"  max drawdown {s.get('max_drawdown', float('nan')):+.3f}"
        f"   win rate {report['win_rate']:.1%}   active win rate {report['active_win_rate']:.1%}"
        f"   profit factor {report.get('profit_factor', float('nan')):.2f}",
        f"  avg turnover/bar {report['avg_turnover']:.4f}"
        f"   concentration (HHI) {report.get('hhi_mean', float('nan')):.3f}"
        f"   total friction paid {report['total_friction_paid']:.4f}",
        f"  longest underwater spell {report.get('max_dd_duration', 0)} bars"
        f" ({report.get('pct_time_underwater', float('nan')):.0%} of time)"
        f"   VaR95 {report.get('var_95', float('nan')):+.4f}"
        f"   CVaR95 {report.get('cvar_95', float('nan')):+.4f}"
        f"   skew {report.get('skew', float('nan')):+.2f}"
        f"   ex-kurt {report.get('excess_kurtosis', float('nan')):+.2f}",
    ]
    if "baselines" in report:
        lines.append("  vs rule baselines (same costs, close fills):")
        lines.append(f"    {'policy':<16} {'active_sh':>9} {'sharpe':>8} {'total':>8} {'maxDD':>8}")
        for name, m in report["baselines"].items():
            lines.append(f"    {name:<16} {m['active_sharpe']:>+9.3f} {m['sharpe']:>+8.3f}"
                         f" {m['total_return']:>+8.3f} {m['max_drawdown']:>+8.3f}")
    return "\n".join(lines)


# ─────────────────────────────────────────── promotion from a recorded run

def promote(run_id: str, cost: CostModel | None = None, split: str = "test",
            fills: str = "close", baselines: bool = True,
            runs_dir: Path | str | None = None, log=print) -> dict:
    """Load a recorded run's final model, roll it over the held-out split, apply the cost
    model, and return (and persist) the formal report.

    ``fills="open"`` switches the agent's accounting to open-price execution.
    ``baselines=True`` adds classic rule policies (rule_policies.RULES) plus constant
    equal-weight, evaluated under the SAME cost model, so the agent has honest company."""
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
    closes = _pivot(eval_df, tics, "close")

    if fills == "open":
        report = evaluate(weights, asset_rets, cost,
                          opens=_pivot(eval_df, tics, "open"), closes=closes)
    else:
        report = evaluate(weights, asset_rets, cost)
    report["run_id"] = run_id
    report["split"] = split
    report["algo"], report["reward"] = m["algo"], m["reward"]
    report["dates"] = hist["date"].astype(str).tolist()

    if baselines:
        from rule_policies import RULES

        def _brief(rep: dict) -> dict:
            s = rep["stats"]
            return {k: s.get(k, float("nan"))
                    for k in ("active_sharpe", "sharpe", "total_return", "max_drawdown")}

        T, N = weights.shape
        comp = {"agent": _brief(report),
                "equal_weight": _brief(evaluate(np.tile(np.full(N, 1.0 / N), (T, 1)),
                                                asset_rets, cost))}
        skipped = []
        for name, fn in RULES.items():
            w_b = fn(closes)
            if np.allclose(w_b, 1.0 / N):             # never left warm-up in this window
                skipped.append(name)
                continue
            comp[name] = _brief(evaluate(w_b, asset_rets, cost))
        report["baselines"] = comp
        if skipped:
            report["baselines_skipped"] = skipped
            log(f"[promote] baselines skipped (warm-up exceeds window): {', '.join(skipped)}")

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
