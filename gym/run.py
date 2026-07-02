"""run.py — CLI entry point for Signal Gym (rev 4 — evolutionary race + FinRL data).

Usage:
  python -m gym.run collect [--force] [SYM ...]   # download 1h OHLCV (yfinance) → raw parquets
  python -m gym.run build                          # raw → FinRL-schema features + splits
  python -m gym.run evolve [--objective timing_sortino] [--pop N] [--gens N] [--monitor]
  python -m gym.run evo-replay [--speed N]         # watch the recorded population race
  python -m gym.run validate [--scope test|validation] [--perms N]   # MCPT + runs test
  python -m gym.run investigate [--tickers ...] [--timesteps N]      # reward investigation (v2)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from gym.config import (
    DEFAULT_CONFIG,
    GYM_DATA,
    GYM_MODELS,
    SPLITS_FILE,
    Config,
    slug,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────── helpers

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


def _load_splits_and_dfs(cfg: Config):
    from gym.features import load_ticker
    if not SPLITS_FILE.exists():
        sys.exit(f"splits.json not found at {SPLITS_FILE}. Run `gym.run build` first.")
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    train_dfs = [load_ticker(slug(t), cfg) for t in splits["training_tickers"]]
    val_dfs   = [load_ticker(slug(t), cfg) for t in splits["validation_tickers"]]
    test_dfs  = [load_ticker(slug(t), cfg) for t in splits["test_tickers"]]
    return splits, train_dfs, val_dfs, test_dfs


# ─────────────────────────────────────────── subcommands

def cmd_collect(args):
    from gym.collect import collect
    syms = args.symbols or None
    done = collect(tickers=syms, force=args.force)
    print(f"Collected {len(done)} tickers → {DEFAULT_CONFIG.raw_data}")


def cmd_build(args):
    from gym.features import build_features
    frames = build_features(DEFAULT_CONFIG)
    print(f"Built {len(frames)} ticker frames. Parquets in {GYM_DATA}")


def cmd_evolve(args):
    from gym.evo import EvoConfig, evolve, save_champion

    cfg = DEFAULT_CONFIG
    splits, train_dfs, val_dfs, test_dfs = _load_splits_and_dfs(cfg)
    evo_cfg = EvoConfig(objective=args.objective, seed=args.seed)
    if args.pop is not None:
        evo_cfg.pop_size = args.pop
    if args.gens is not None:
        evo_cfg.n_generations = args.gens
    if args.hidden is not None:
        evo_cfg.hidden = args.hidden

    record = GYM_MODELS / "evo_generation.npz"
    holder: dict = {}

    def worker(monitor):
        try:
            holder["result"] = evolve(train_dfs, val_dfs, test_dfs, cfg, evo_cfg,
                                      monitor=monitor, record_path=record)
        except Exception as e:  # noqa: BLE001
            holder["error"] = e
        finally:
            if monitor is not None:
                monitor.emit(None)

    if args.monitor:
        from gym.evo_monitor import EvoMonitor
        mon = EvoMonitor()
        mon.run(lambda: worker(mon))               # blocks on main thread; worker in bg
    else:
        worker(None)

    if "error" in holder:
        raise holder["error"]
    result = holder["result"]
    save_champion(result)

    ts = result.test_stats or {}
    print(f"\nEvolution complete. objective={result.objective}")
    print(f"  best gen {result.best_generation_idx}  val_fitness={result.val_fitness:.4f}")
    if ts:
        from gym.stats import objective_value
        print(f"  test: fitness={objective_value(ts, result.objective):+.4f}  "
              f"weight_std={ts.get('weight_std', float('nan')):.3f} (0=constant hold)  "
              f"turnover={ts.get('turnover', float('nan')):.1f}  "
              f"return={ts.get('total_return', float('nan')):+.3f}  "
              f"maxDD={ts.get('max_drawdown', float('nan')):.3f}")
    print(f"  champion + final-generation race saved under {GYM_MODELS}")
    print(f"  watch it:  python -m gym.run evo-replay")


def cmd_evo_replay(args):
    from gym.evo_replay import GenerationReplay, replay_evolution
    from gym.features import load_ticker

    cfg = DEFAULT_CONFIG
    record = GYM_MODELS / "evo_generation.npz"
    if not record.exists():
        sys.exit("No recorded generation found. Run `python -m gym.run evolve` first.")
    gen = GenerationReplay.load(record)
    ordered_dfs = [load_ticker(slug(t), cfg) for t in gen.ordered_slugs]
    replay_evolution(gen, ordered_dfs, cfg.lookback, speed=args.speed)


def cmd_validate(args):
    import numpy as np

    from gym.backtest import run_backtest
    from gym.env import SignalGymEnv
    from gym.evo import load_champion
    from gym.normalize import fit_obs_stats
    from gym.valid import mcpt, runs_test

    cfg = DEFAULT_CONFIG
    splits, train_dfs, val_dfs, test_dfs = _load_splits_and_dfs(cfg)
    champ = load_champion()
    if champ is None:
        sys.exit("No evolved champion (evo_champion.npz). Run `gym.run evolve` first.")

    eval_dfs = test_dfs if args.scope == "test" else val_dfs
    if not eval_dfs:
        sys.exit(f"No tickers in the {args.scope} split.")
    obs_stats = fit_obs_stats(train_dfs, cfg)     # champion was evolved under this norm

    all_excess: list[float] = []
    all_trade_rets: list[float] = []
    for df in eval_dfs:
        env = SignalGymEnv(df, cfg, obs_stats=obs_stats)
        res = run_backtest(env, champ)
        ec = res.equity_curve
        all_excess.extend(ec["excess"].tolist())
        trade_mask = ec["weight"].diff().fillna(0) != 0
        all_trade_rets.extend(ec.loc[trade_mask, "ret"].tolist())

    mcpt_res = mcpt(np.array(all_excess), n_perms=args.perms)
    runs_res = runs_test(np.array(all_trade_rets))

    print(f"\nValidate champion ({args.scope}, {len(eval_dfs)} tickers):")
    print(f"  MCPT:      Sharpe={mcpt_res.observed_sharpe:.4f}  p={mcpt_res.p_value:.4f}  "
          f"significant={mcpt_res.significant}")
    print(f"  Runs test: n_runs={runs_res.n_runs}  z={runs_res.z_stat:.3f}  "
          f"p={runs_res.p_value:.4f}  significant={runs_res.significant}")


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
    rows = investigate(cfg, reward_names=args.rewards, timesteps=args.timesteps, seed=args.seed,
                       lookback=args.lookback, device=args.device, log=print)
    print("\n" + format_table(rows))
    print("\nVerdict: " + verdict(rows))
    g = champion_gate(rows, n_perms=args.perms)
    if g is not None:
        print("Champion " + g.summary())
    if not args.no_log:
        append_to_research_log(rows, cfg, timesteps=args.timesteps, seed=args.seed)
        print(f"\nAppended to {Path(__file__).resolve().parent / 'RESEARCH_LOG.md'}")


# ─────────────────────────────────────────── argument parser

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gym.run", description="Signal Gym CLI")
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    co = sub.add_parser("collect")
    co.add_argument("symbols", nargs="*", help="optional subset of Yahoo symbols")
    co.add_argument("--force", action="store_true", help="re-download existing tickers")

    sub.add_parser("build")

    ev_ = sub.add_parser("evolve")
    ev_.add_argument("--objective", default="timing_sortino",
                     help="selection objective (see gym.stats.OBJECTIVES)")
    ev_.add_argument("--pop", type=int, default=None, help="population size")
    ev_.add_argument("--gens", type=int, default=None, help="number of generations")
    ev_.add_argument("--hidden", type=int, default=None, help="policy MLP hidden units")
    ev_.add_argument("--seed", type=int, default=42)
    ev_.add_argument("--monitor", action="store_true",
                     help="live dashboard: population race + skill curve + leaderboard")

    er = sub.add_parser("evo-replay")
    er.add_argument("--speed", type=float, default=1.0)

    va = sub.add_parser("validate")
    va.add_argument("--scope", choices=["validation", "test"], default="test")
    va.add_argument("--perms", type=int, default=10_000)

    iv = sub.add_parser("investigate", help="v2 multi-asset reward investigation")
    iv.add_argument("--tickers", nargs="*", default=None, help="Yahoo symbols (default: a "
                    "small diversified basket)")
    iv.add_argument("--rewards", nargs="*", default=None,
                    help="subset of reward names (default: all in rewards.REWARDS)")
    iv.add_argument("--train-start", default="2014-01-01")
    iv.add_argument("--train-end", default="2022-01-01")
    iv.add_argument("--trade-start", default="2022-01-01")
    iv.add_argument("--trade-end", default="2024-01-01")
    iv.add_argument("--timesteps", type=int, default=20_000)
    iv.add_argument("--lookback", type=int, default=60)
    iv.add_argument("--seed", type=int, default=42)
    iv.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    iv.add_argument("--perms", type=int, default=10_000)
    iv.add_argument("--no-log", action="store_true", help="don't append to RESEARCH_LOG.md")

    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    dispatch = {
        "collect":    cmd_collect,
        "build":      cmd_build,
        "evolve":     cmd_evolve,
        "evo-replay": cmd_evo_replay,
        "validate":   cmd_validate,
        "investigate": cmd_investigate,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
