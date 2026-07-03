# Signal Gym

A local research framework for training reinforcement-learning agents to allocate a
**long-only, multi-asset spot portfolio** from **price-only (OHLCV) features**, and for
answering one question honestly:

> Is there generalizable, feature-driven allocation edge over a naive equal-weight portfolio —
> and which reward objective actually selects for it?

The reward objective is not assumed. Several candidate rewards are trained head-to-head and
judged on out-of-sample data with a significance gate, so the framework reports a defensible
*positive* result or a rigorous *negative* one — never a metric flattering itself.

## How it works

Each run is a pipeline:

1. **Data.** Daily OHLCV for a chosen universe is downloaded and enriched with standard
   technical indicators. A per-bar covariance matrix is added from a trailing window of past
   returns (strictly causal — no future bars enter any observation).
2. **Environment.** An agent observes the current bar (covariance + indicators) and emits a
   raw score per asset; a softmax turns it into portfolio weights — a long-only point on the
   simplex that sums to 1. The realized return is the weighted basket return over the *next*
   bar, so a decision only ever earns the move that follows it. Turnover is charged a
   transaction cost, so churn is never free.
3. **Reward.** The per-step training reward is pluggable (see [Rewards](#rewards)). Each
   candidate is online and causal — it reads only the latest bar and a running estimate of the
   past, never the whole episode.
4. **Training.** One PPO policy is trained per candidate reward.
5. **Evaluation.** The out-of-sample period is split chronologically into a **validation** and
   a **test** slice. Each agent is scored on both. Candidates are ranked by **validation**
   `active_sharpe` (the information ratio versus an equal-weight benchmark); the test column is
   reported but never used for selection.
6. **Significance gate.** The selected champion's test performance is run through a Monte-Carlo
   permutation test (sign-flip on the active-return Sharpe) and a Wald–Wolfowitz runs test, so
   an apparent edge has to clear a noise threshold before it counts.
7. **Regime breakdown.** Test bars are labeled bull / bear / choppy from a causal trailing
   trend of the benchmark, and performance is reported per regime — a flat verdict can hide a
   regime effect, so it is split out.

Every run serializes its full config and seed to a JSON manifest and appends a dated entry to
the research log.

## Design constraints

- **Long-only spot** — no shorting, leverage, derivatives, or options.
- **Research only** — backtest/evaluation; no broker execution and no real capital.
- **Price-only** — features derive from OHLCV; no news, fundamentals, or order-book data.
- **No lookahead** — at bar `t` only bars ≤ `t` are observable; fills happen on the next bar;
  the covariance and benchmark labels use trailing windows only.
- **Reproducible & leakage-free** — identical config + seed reproduce a run; selection touches
  validation only, and the test slice is scored once.

## Quick start

The framework runs in the project virtual environment (Python 3.11, with PyTorch, Stable-
Baselines3, FinRL, and PyQt5).

Run the reward investigation from the command line:

```bash
# default diversified basket, daily data
python -m gym.run investigate --timesteps 20000

# choose your own universe, dates, training budget, and device
python -m gym.run investigate \
    --tickers AAPL MSFT JPM XOM CAT PG JNJ WMT \
    --train-start 2014-01-01 --train-end 2022-01-01 \
    --trade-start 2022-01-01 --trade-end 2024-01-01 \
    --timesteps 50000 --lookback 60 --device cpu
```

This prints a ranked comparison table, the verdict, and the significance gate; writes a
reproducibility manifest to `reports/`; and appends a dated entry to
[`RESEARCH_LOG.md`](RESEARCH_LOG.md). Pass `--no-log` to skip the log/manifest.

Screen a basket's current technical state (uses the local OHLCV cache, fetching what's
missing):

```bash
python -m gym.run screen --tickers AAPL MSFT JPM XOM
```

Or drive it visually from the desktop control panel:

```bash
python gym/control_panel.py
```

Open the **Reward Investigation (multi-asset)** tab, set the universe and training budget, and
press **Run**. You get a live log, a bar chart of validation-vs-test `active_sharpe` per
reward, the champion's equity against the equal-weight benchmark, and the verdict + gate.

## Rewards

The candidate training rewards compared by the investigation:

| Name | What it optimizes |
|------|-------------------|
| `return` | the raw per-bar net return |
| `logret` | the per-bar net log return (compounding-consistent) |
| `diff_sharpe` | an online estimate of the Sharpe ratio's increment (rewards risk-adjusted return, per step) |
| `active` | the per-bar return in excess of the equal-weight benchmark |
| `active_dsr` | the online differential Sharpe of the *active* (excess-over-benchmark) series — risk-adjusted edge over an honest benchmark |

New rewards are added to the registry in [`rewards.py`](rewards.py); the investigation picks
them up automatically.

## How results are judged

The headline metric is **`active_sharpe`**: the annualized information ratio of the portfolio's
returns *minus the equal-weight benchmark's returns*, bar for bar. Because the benchmark is
external and computed from the same bars, an agent cannot earn a positive score by simply
tracking or de-risking toward the benchmark — only genuine relative skill scores. A positive
absolute Sharpe with a negative `active_sharpe` means the portfolio made money but did not beat
naive equal weighting. The significance gate then decides whether a positive `active_sharpe` is
distinguishable from luck.

## Modules

```
marketdata.py        provider-swappable daily OHLCV source (yfinance default) + parquet cache
screener.py          per-ticker technical snapshot: RSI, MACD, Bollinger, ATR, volume, trend
pipeline.py          data prep (via marketdata) + feature engineering + PPO train/backtest wrapper
portfolio.py         the multi-asset portfolio environment: softmax-weight allocation,
                     turnover cost, pluggable reward, causal covariance data-prep, PPO train/rollout
rewards.py           the per-step training-reward registry
stats.py             portfolio performance metrics (Sharpe/Sortino/Calmar, active_sharpe, drawdown)
signif.py            the out-of-sample significance gate (permutation + runs test)
regime.py            causal bull/bear/choppy labeling + per-regime evaluation
investigate.py       the reward investigation: train per reward, score on val/test, rank, gate, report
evo_portfolio.py     GPU-batched population evolution on the multi-stock portfolio
evo_replay_panel.py  3-pane bar-by-bar replay of a recorded population-evolution run
control_panel.py     PyQt desktop panel to configure, launch, watch, and stop runs
run.py               command-line entry point
```

## Outputs

- [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — a dated, append-only record of every investigation:
  the ranked table, verdict, significance gate, and per-regime breakdown.
- `reports/investigation_*.json` — a reproducibility manifest per run (full config, seed,
  ranking, and gate result).

## Tests

```bash
pytest gym/tests -q
```

Covering environment mechanics and causality, turnover cost, the reward functions, the
performance metrics and their leakage-free benchmark comparison, the significance gate, the
holdout split and reporting, the regime labeling, the market-data cache (contract, cache
hits, range extension, offline fallback), and the screener's indicators (values on known
series plus a future-perturbation causality test).
