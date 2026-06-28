# Evolutionary Visual Gym — design & status

A population/selection training mode for Signal Gym, built to be **watched**. Where the
original trainer fits one PPO policy by gradient descent, this mode evolves a *population* of
agents through the ticker universe, prunes the ones that go broke, and selects survivors by a
chosen objective — so the training process is legible as a crowd of portfolios racing,
surviving, and being culled. Inspired by PokemonRedExperiments' watchable multi-agent training.

**Goal is not to beat buy-and-hold.** The study showed there's no long-only timing edge on
this universe. The goal is to **watch agents learn to manage a portfolio and understand why
they act** — a training-dynamics showcase and a diagnostic instrument. The reward is designed
so that *timing*, not de-risking, is what gets selected (see "Reward design" below).

## Status (2026-06-28)

| Phase | What | State |
|---|---|---|
| P1 | `stats.py` (metrics + objective registry) · `population.py` (threaded pass, ruin, selection) | **done, tested** |
| P2 | `evo.py` (EvoPolicy MLP + generational ES loop, val-gated) | **done, tested** |
| Reward | timing-vs-matched-constant objective; turnover/activity metrics | **done, tested** |
| P3 | `evo_replay.py` (recorded replay: world + race + leaderboard) + `evolve`/`evo-replay` CLI | **done; data layer tested, Qt window runs locally** |
| P4 | live attach — watch the population race *during* training (per-generation refresh) | **next** |

145 tests pass (`pytest gym/tests -q`).

## Locked decisions

| Decision | Choice |
|---|---|
| Optimizer | **ES / neuroevolution** — population of small MLP policies, no gradients; per generation: evaluate → prune ruined → select by objective → mutate/recombine (elitism + Gaussian mutation). PPO kept as a sibling learner in the same env. |
| The "world" | Randomly-ordered training tickers, stepped bar-by-bar; Prophets signals revealed only at bar close (existing causality invariant). |
| Capital model | **Carried across the whole pass** — one continuous equity curve threaded through all tickers. A bad early ticker can ruin an agent. |
| Ruin rule | Long-only spot can't hit zero, so "broke" = a **ruin line**: threaded equity < `ruin_frac` (default **0.45**) → agent dead, curve ends. |
| Ordering | **Shared** random order within a generation (all agents share one world chart / time axis), **re-rolled** between generations to average out order-luck. |
| Objective | **Pluggable** backtesting.py-style metric; default **`timing_sortino`** (see Reward design). Selectable in code now, Sandbox UI later. |
| Overfitting guard | Evolve/prune on **training**; the search is driven by train fitness, the champion is **val-gated** (val is never the breeding target); **test** touched once. |
| Rendering | Recorded replay first (3 panes: world / equity-race / leaderboard); live attach (P4) is per-generation. Desktop finplot + pyqtgraph, reusing the existing stack. |

## Architecture

```
stats.py        backtesting.py-style metrics from an equity curve; OBJECTIVES registry
                (name -> scalar, higher=better); timing metrics + turnover/activity.
population.py   run_pass(policy, ordered_dfs) -> threaded equity curve + ruin flag + stats;
                run_population(...) -> per-agent PassResult; select_survivors(...);
                shuffled_order(...) (re-rollable shared order).
evo.py          EvoPolicy (numpy MLP over the current-bar feature row; genome<->flat vector);
                EvoConfig; evolve() generational loop (val-gated, records final-gen race);
                save_champion / load_champion.
evo_replay.py   GenerationReplay (compact (A×T) equity/weight matrices, save/load); data layer
                (race_frame, leaderboard, build_world_frame); replay_evolution (finplot draw).
run.py          `evolve` (train) and `evo-replay` (watch) subcommands.
```

Reused unchanged: `env.SignalGymEnv` (per-ticker stepper; population.py threads capital on
top), `backtest.run_backtest`, `agent_rl.fit_obs_stats` (obs normalization — ES needs it too,
unnormalized obs collapse the policy), `features`, `config`, the split discipline, `replay`,
`monitor`.

## Reward design (the central decision)

The agent sets a continuous target weight in [0, 1]. Under return- or ratio-based objectives
(return, Sharpe, Sortino, Calmar) it **collapses to a constant partial-exposure hold** — a
de-risked buy-and-hold — and learns no timing. Two reasons:

1. **Ratio objectives are scale-free.** 100% vs 50% exposure gives nearly the same ratio, so
   they reward the *shape* of the curve, not the level. On a +19%/yr bull tape with no real
   timing edge, constant exposure is the easy attractor.
2. **A static drawdown penalty is satisfied by static de-risking.** `return − λ·drawdown` is
   cheapest to satisfy by holding a *smaller constant fraction* — less exposure, less
   drawdown, zero timing. (Confirmed earlier: the PPO `absolute_dd` run produced a de-risked
   B&H that a constant-exposure twin beat.)

**Solution — reward timing against the agent's own average-exposure twin.** Define per bar
`active_t = agent_ret_t − mean_weight · market_ret_t`, the active return over a strategy that
just holds the agent's *mean* exposure every bar. Fitness = the risk-adjusted ratio of
`active`:

- a constant-exposure agent **is** its twin → active ≡ 0 → score **0** (de-risking buys
  nothing),
- an agent lighter into drops / heavier into rallies (timing, from features) scores **> 0**,
- a bad timer scores **< 0**.

This isolates timing, can't be faked by de-risking, encodes drawdown-aversion only when it's
*timed*, and tells the agent nothing about *when* or *which* features — so the realization
stays the agent's. Implemented as `timing_sharpe` / `timing_sortino` in `stats.py`
(`_timing_ratio` falls back to full dispersion when an agent never underperforms its twin, so a
perfect timer scores high rather than 0/NaN). `timing_sortino` is the evo default.

**Honest caveat:** changing the objective changes what we *select for* — it cannot manufacture
edge the features don't contain. On this bull tape the likely outcome is timing fitness near 0
out-of-sample (see findings). The real lever for *guaranteed* timing edge is the **data**
(a bear/choppy period where timing pays); that's a separate collection task.

## Metrics & objectives

`stats.compute_stats(equity_curve)` returns a flat dict: returns / CAGR, Sharpe / Sortino /
Calmar, max & avg drawdown, exposure, alpha/beta, plus **timing** (`timing_sharpe`,
`timing_sortino`, `active_return`) and **activity** (`weight_std`, `turnover` = Σ|Δweight|,
`n_reweights`). A "trade" = a maximal run of nonzero exposure; trade stats (win rate, profit
factor, expectancy, SQN, Kelly) are derived from those holding periods. Because a continuous
sizer rarely reaches full cash, `n_trades` undercounts activity — **`weight_std` and
`turnover` are the honest activity signals** (`weight_std ≈ 0` = constant hold; `> 0` = timing).

`OBJECTIVES` maps a name to a scalar (higher = better); `objective_value` returns `-inf` for
undefined metrics and for trade-based objectives below `min_trades`, so degenerate agents sort
last. Registered: `sortino, sharpe, calmar, return, cagr, profit_factor, sqn, expectancy,
win_rate, kelly, return_over_dd, timing_sortino, timing_sharpe`.

## Key findings so far

- **Under return/Sortino:** champion collapses to constant ~partial-exposure hold (de-risked
  B&H, `weight_std ≈ 0`). Gradient-free ES independently reproduces the PPO negative result.
- **Under `timing_sortino`:** the collapse breaks — agents genuinely vary exposure
  (`weight_std ≈ 0.37`, mean ~0.45) and evolution improves train timing-fitness across
  generations. **But it overfits**: validation bounces and held-out **test timing-fitness is
  negative** — the timing doesn't generalize. So the features carry timing patterns that fit
  in-sample but no real out-of-sample edge, reconfirming the project's core finding through a
  new optimizer. This in-sample-good / out-of-sample-bad gap is itself the watchable result.

## How to run

```bash
# train a population (records the final-generation race for replay)
python -m gym.run evolve --objective timing_sortino --pop 64 --gens 40 --seed 42

# watch the recorded race: world candles + signals / equity race / leaderboard
python -m gym.run evo-replay [--speed N]
```

Artifacts: `gym/models/evo_champion.npz` (winning genome + arch),
`gym/models/evo_generation.{npz,json}` (final-generation race recording).

## P4 — live attach (next)

Watch the population race **during** training. Granularity = **per generation** (not per bar —
per-bar live rendering would throttle training). Each generation finishes → the live view
redraws: (1) population equity race (champion bold, ruined red, B&H dashed, ruin line),
(2) skill curve (best + mean fitness per generation), (3) leaderboard. `evolve()` emits a
per-generation snapshot (population equity downsampled to ~200 pts + ruined flags + fitnesses +
champion idx) over the existing thread-safe `monitor` queue; wired via `evolve --monitor`. The
world candle pane stays in recorded replay only (tickers sweep too fast to read live).

## Open directions

- **Data angle** — collect a bear/choppy period so timing genuinely pays; the only path to a
  *positive* out-of-sample timing result.
- **True in/out timing** — let exposure reach 0 (current sigmoid never fully de-risks), making
  drawdown-aversion literal.

## Non-goals
- No shorting / leverage / options (long-only spot).
- No claim of market-beating edge; B&H shown for reference only.
- No web/broadcast layer in v1 (desktop, reusing the existing stack).
