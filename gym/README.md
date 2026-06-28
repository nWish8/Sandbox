# Signal Gym

A local research framework for discovering which Prophets indicator signals carry edge
in price action, using a long-only spot portfolio agent trained with PPO and validated
with a LightGBM supervised baseline. It also includes an **evolutionary visual training
mode** (a watchable population of agents) — see [`evo_spec.md`](evo_spec.md).

## Goals

1. **Train a general agent** that manages a long-only spot portfolio on any single ticker,
   learning online by playing through a training universe of tickers bar-by-bar.
2. **Discover signal edge** — which Prophets signals, conditions (regimes/contexts), and
   interactions (confluence, cross-TF agreement) actually predict price movement.
3. **Make training watchable** — evolve a *population* of agents through the universe and
   watch them race, get pruned at a ruin line, and be selected by a chosen objective. The
   reward selects for genuine *trade timing*, not de-risking ([`evo_spec.md`](evo_spec.md)).

## Quick start

```bash
# 1. Collect market data + Prophets signals via TradingView CDP
python -m gym.run collect

# 2. Build feature matrices and 3-way ticker splits
python -m gym.run build

# 3. Train (supervised baseline first — fast, interpretable)
python -m gym.run train --agent sup

# 4. Evaluate on validation set
python -m gym.run eval --scope validation --agent sup

# 5. Build signal-edge report (D-B)
python -m gym.run analyze --agent sup

# 6. Statistical validation (MCPT + runs test)
python -m gym.run validate --scope test --agent sup

# 7. Replay a trained policy on a single ticker
python -m gym.run replay NYSE:LMT --agent sup

# --- RL (after supervised baseline is validated) ---
python -m gym.run train --agent ppo --monitor
python -m gym.run eval  --scope test --agent ppo
python -m gym.run analyze --agent ppo

# --- Evolutionary visual mode (watchable population) ---
python -m gym.run evolve --objective timing_sortino --pop 64 --gens 40
python -m gym.run evo-replay        # world candles + signals / equity race / leaderboard
```

## Architecture

```
collect.py   → signal_study/results/*.json     scrape OHLCV + all indicator plots
features.py  → gym/data/*.parquet              per-bar causal feature matrices
env.py       → SignalGymEnv                    Gymnasium env, one ticker per episode
agent_sup.py → EdgePolicy (LightGBM)           supervised baseline + SHAP attribution
agent_rl.py  → RLPolicy (PPO)                  primary learner
train.py     → training_run()                  episodic loop, checkpoint/eval protocol
backtest.py  → run_backtest(), fast_vectorized  policy evaluation + metrics
analysis.py  → build_edge_report()             signal-edge report (D-B)
valid.py     → mcpt(), runs_test()             MCPT + Wald-Wolfowitz validation
baselines.py → BuyAndHoldPolicy, RulePolicy    B&H + A3/A5b/PCB benchmarks
monitor.py   → TrainingMonitor                 live pyqtgraph 4-panel dashboard
replay.py    → replay()                        finplot bar-stepped chart
run.py       → CLI                             all commands above

# --- evolutionary visual mode (see evo_spec.md) ---
stats.py     → compute_stats(), OBJECTIVES     backtesting.py-style metrics + objective registry
population.py→ run_pass(), select_survivors()  capital-threaded pass, ruin pruning, selection
evo.py       → EvoPolicy, evolve()             ES/neuroevolution population trainer
evo_replay.py→ GenerationReplay, replay_evolution  recorded race: world / equity-race / leaderboard
```

## Key design decisions

- **Long-only spot only** — no shorts, no leverage (hard constraint).
- **Three disjoint ticker splits** — train / validation / test. Generalisation to
  *unseen tickers* is the point; no time-based leakage allowed.
- **No-lookahead invariant** — observation at bar `t` contains only features from bars
  ≤ `t`; fills at `t+1` open; vol terciles and z-scores fitted on training tickers only.
- **Differential Sharpe reward** — Moeller differential Sharpe of excess-over-B&H,
  turnover-penalised. The RL training signal.
- **Baseline-first build order** — supervised LightGBM runs in seconds, produces the
  first signal-edge report, and validates the full pipeline before PPO is engaged.
- **Capture everything** — all Prophets signal plots (event fires and continuous values)
  are in the feature matrix. Feature selection / regularisation is the model's job.
- **Reward selects for timing, not de-risking** (evolutionary mode) — ratio objectives are
  scale-free and a static drawdown penalty just yields a smaller constant hold, so the default
  objective rewards an agent's active return over its *own average-exposure twin*
  (`timing_sortino`). A constant-exposure agent scores 0; only feature-driven timing scores.
  See [`evo_spec.md`](evo_spec.md) for the full rationale and findings.

## Feature vocabulary

Signal features (per event signal): `{sig}_fired`, `{sig}_bars_since`, `{sig}_within_3`, `{sig}_ever`  
Continuous signals: `{sig}_val` (z-scored, train-fit), `{sig}_slope_3`  
Conditions: `bull_active_count`, `bear_active_count`, `continuation_active`,
`cloud_state ∈ {-1, 0, 1}`, `background_active`, `points_active`, `vol_regime ∈ {0, 1, 2}`  
Interactions: `htf_bull_confirm_4h`, `htf_bear_confirm_4h`  
Context: `asset_class ∈ {0..3}` (no raw ticker identity)  
Price: `ret_1/3/6`, `vol_20`, `range_pos_20`, `drawdown_20`, `atr_14`  
Target (supervised): `fwd_edge_6 = fwd_ret_6 − baseline_long_6`

## Tests

```bash
pytest gym/tests -q
```

145 tests covering: features causality, no-lookahead invariants, env mechanics, env
reward sign, supervised agent importances, backtest equity math (vectorised = env),
MCPT and runs test, full pipeline integration, determinism, and the evolutionary mode
(stats/objectives incl. timing-vs-twin, capital-threaded passes + ruin pruning + selection,
the ES loop and EvoPolicy, and the replay data layer + persistence).

## Deliverables

- **D-A** `gym/models/ppo_{run}.zip` — trained PPO policy + checkpoints
- **D-B** `gym/reports/edge_report.json` — signal-edge report
  (solo edge, condition/interaction edge, GBT importances/SHAP, RL ablation deltas)
- **D-C** Gym itself — reusable framework; both PPO and supervised plug in unchanged
- **D-E** `gym/models/evo_champion.npz` — winning evolved genome + architecture
- **D-F** `gym/models/evo_generation.{npz,json}` — final-generation race recording (replayed
  by `evo-replay`)
