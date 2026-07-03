"""run.py — CLI entry point for Signal Gym / Vision.

Usage:
  python -m gym.run investigate [--tickers SYM ...] [--timesteps N] [--lookback N]
                                [--device cpu|cuda] [--no-log]
  python -m gym.run screen      [--tickers SYM ...] [--end YYYY-MM-DD]

`investigate` trains one PPO policy per candidate reward on the FinRL multi-asset
portfolio env, scores each on a chronological validation/test split, ranks by
validation active_sharpe (selection never touches test), runs the significance gate
on the champion's test edge, and appends the dated result to RESEARCH_LOG.md.

`screen` prints the technical snapshot table (RSI/MACD/Bollinger/ATR/volume/trend)
for a basket, using the local OHLCV cache (fetches what's missing).
"""
from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s  %(message)s", level=level)
    for noisy in ("matplotlib", "PIL", "finplot", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_DEFAULT_PORTFOLIO = ["AAPL", "MSFT", "JPM", "XOM", "CAT", "PG", "JNJ", "WMT"]


def cmd_investigate(args):
    """v2 reward investigation: train PPO under each candidate reward on FinRL's multi-asset
    StockPortfolioEnv, score on a held-out val/test split, rank by validation active_sharpe, and
    run the significance gate on the champion's test edge."""
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))   # gym/ → flat finrl-side imports
    from pipeline import FinRLConfig
    from investigate import (append_to_research_log, champion_gate, format_table,
                             investigate, verdict)

    cfg = FinRLConfig(
        tickers=args.tickers or _DEFAULT_PORTFOLIO,
        train_start=args.train_start, train_end=args.train_end,
        trade_start=args.trade_start, trade_end=args.trade_end,
        agent="ppo", device=args.device,
    )
    rows = investigate(cfg, reward_names=args.rewards, algo=args.algo, timesteps=args.timesteps,
                       seed=args.seed, lookback=args.lookback, device=args.device, log=print)
    print("\n" + format_table(rows))
    print("\nVerdict: " + verdict(rows))
    g = champion_gate(rows, n_perms=args.perms)
    if g is not None:
        print("Champion " + g.summary())
    if not args.no_log:
        append_to_research_log(rows, cfg, timesteps=args.timesteps, seed=args.seed)
        print(f"\nAppended to {Path(__file__).resolve().parent / 'RESEARCH_LOG.md'}")


def cmd_project(args):
    """ML return projections with confidence bands + OOS coverage honesty check."""
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from projections import format_projection, project

    table, honesty = project(args.tickers or _DEFAULT_PORTFOLIO, horizon=args.horizon,
                             end=args.end, log=print)
    print(format_projection(table, honesty, args.horizon))


def cmd_screen(args):
    """Technical screen: one snapshot row per ticker from cached daily OHLCV."""
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))   # gym/ → flat finrl-side imports
    from screener import format_screen, screen

    table = screen(args.tickers or _DEFAULT_PORTFOLIO, end=args.end, log=print)
    print(format_screen(table))
    if args.json:
        table.to_json(args.json, orient="records", indent=2)
        print(f"\nWritten to {args.json}")


# ─────────────────────────────────────────── argument parser

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gym.run", description="Signal Gym CLI")
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    iv = sub.add_parser("investigate", help="multi-asset reward investigation")
    iv.add_argument("--tickers", nargs="*", default=None, help="Yahoo symbols (default: a "
                    "small diversified basket)")
    iv.add_argument("--rewards", nargs="*", default=None,
                    help="subset of reward names (default: all in rewards.REWARDS)")
    iv.add_argument("--train-start", default="2014-01-01")
    iv.add_argument("--train-end", default="2022-01-01")
    iv.add_argument("--trade-start", default="2022-01-01")
    iv.add_argument("--trade-end", default="2024-01-01")
    iv.add_argument("--algo", choices=["ppo", "sac", "a2c"], default="ppo")
    iv.add_argument("--timesteps", type=int, default=20_000)
    iv.add_argument("--lookback", type=int, default=60)
    iv.add_argument("--seed", type=int, default=42)
    iv.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    iv.add_argument("--perms", type=int, default=10_000)
    iv.add_argument("--no-log", action="store_true", help="don't append to RESEARCH_LOG.md")

    sc = sub.add_parser("screen", help="technical snapshot table for a basket")
    sc.add_argument("--tickers", nargs="*", default=None, help="Yahoo symbols (default: a "
                    "small diversified basket)")
    sc.add_argument("--end", default=None, help="as-of date YYYY-MM-DD (default: today)")
    sc.add_argument("--json", default=None, help="also write the table to this JSON path")

    pj = sub.add_parser("project", help="ML return projections with confidence bands")
    pj.add_argument("--tickers", nargs="*", default=None)
    pj.add_argument("--horizon", type=int, default=20, help="forward horizon (trading days)")
    pj.add_argument("--end", default=None, help="as-of date YYYY-MM-DD (default: today)")

    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    dispatch = {
        "investigate": cmd_investigate,
        "screen": cmd_screen,
        "project": cmd_project,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
