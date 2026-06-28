# Signal Gym — Walkthrough (the whole thing, in order)

> **Purpose of this file.** A plain-language, step-by-step tour of *everything* the gym
> does: what runs, in what order, what gets decided at each step, how the reinforcement
> learning actually works, and — the part you most want to shape — the exact sequence of
> visualizations and what each pane shows.
>
> **This is a working document. Edit it.** Wherever you see a `> ✏️ EDIT:` block, that's a
> decision or a visual we can change together. Add your own. When we agree, the change
> flows back into the code (the relevant `file:line` is named so we both know where it
> lives). Nothing here is locked.
>
> Companion docs: [`BRIEF.md`](BRIEF.md) (vision), [`spec.md`](spec.md) (the technical
> spec / decisions log), [`evo_spec.md`](evo_spec.md) (the evolutionary visual mode),
> [`README.md`](README.md) (quick commands).

---

## 0. The one-paragraph mental model

You have ~34 tickers, each with daily price bars **plus** the full Prophets indicator
signal suite aligned to those bars. The gym turns each ticker into a little **game**: step
through the bars one at a time; at each bar an **agent** looks at a window of recent
features (price + signals) and decides *how much of the ticker to hold* (a weight from 0 =
all cash to 1 = fully long — **long-only spot, no shorts, no leverage**). We score the
agent not on raw profit (buy-and-hold already makes +19%/yr on this universe — too easy to
just hold) but on **timing skill**: did it lighten up before drops and lean in before
rallies, beyond what its own average exposure would have done? Two kinds of agent learn
this game — a **PPO neural net** (gradient reinforcement learning) and an **evolved
population** of small nets (gradient-free, and built to be *watched* racing). The real
deliverable isn't the agent; it's the **knowledge**: a signal-edge report saying *which
Prophets signals and combinations actually predict price*, validated statistically so we
don't fool ourselves.

**Honest headline finding so far:** on this (bullish) data, there is **no long-only timing
edge that survives out-of-sample**. In-sample the agents find timing patterns; on held-out
tickers the timing fitness goes negative. That null result *is* a legitimate output — and
the watchable training is the instrument that makes it legible.

---

## 1. The whole pipeline at a glance

Everything is driven by one CLI: `python -m gym.run <command>`
([run.py](run.py)). Here is the full intended order, end to end:

| # | Command | What happens | Produces | Visual? |
|---|---------|--------------|----------|---------|
| 1 | `collect` | Scrape OHLCV + **every** indicator plot per ticker/TF from TradingView (CDP) | `signal_study/results/*.json` | no |
| 2 | `build` | Turn JSONs into per-bar causal feature matrices + 3-way ticker split | `gym/data/*.parquet`, `splits.json` | no |
| 3 | `train --agent sup` | Fit LightGBM baseline on pooled training rows (fast, interpretable) | `edge_model.lgbm`, run JSON | **Monitor** (optional) |
| 4 | `train --agent ppo --monitor` | Episodic PPO playthrough of training tickers | `ppo_best.zip` + checkpoints | **Monitor** |
| 5 | `eval --scope test/validation` | Score a trained policy on unseen tickers | printed metrics | no |
| 6 | `analyze` | Build the signal-edge report (solo edge + interactions + SHAP + RL ablation) | `reports/edge_report.json` | no |
| 7 | `validate --scope test` | MCPT (is the edge chance?) + runs test (serial dependence) | printed p-values | no |
| 8 | `replay TICKER` | Watch one trained agent trade one ticker bar-by-bar | finplot window | **Replay** |
| 9 | `evolve --objective timing_sortino --monitor` | Evolve a *population* through the universe; watch them race + get culled | `evo_champion.npz`, `evo_generation.*` | **Evo Monitor** (live) |
| 10 | `evo-replay` | Re-watch the recorded final-generation race | finplot window | **Evo Replay** (recorded) |
| 11 | `report` | Print the edge-report leaderboard | stdout | no |

> ✏️ **EDIT (order):** The spec leans "baseline-first" (steps 3 → 6 → 7 before PPO) to
> de-risk the pipeline, then PPO. You could instead go straight to PPO, or make the
> evolutionary mode (9–10) the primary experience. Tell me which track is the "main" one
> for you and I'll reorganize the docs/commands around it.

**Where we are right now** (artifacts on disk, 2026-06-28):
- ✅ 34 ticker parquets + `splits.json` built (step 2 done).
- ✅ Supervised baseline trained (`edge_model.lgbm`) + edge report written (steps 3, 6).
- ✅ PPO trained, 9 checkpoints + `ppo_best.zip` (step 4 done).
- ✅ Evolution run, champion + final-generation race recorded (step 9 done).
- ⏳ **P4 "live attach" for evolution is the next build item** (watch the race *during*
  training) — the live `EvoMonitor` exists and is wired, but the spec lists per-generation
  live attach as the next milestone to harden.

---

## 2. Part A — from raw data to a playable game

### Step 1 · `collect` — scrape everything ([collect.py](collect.py))

- **Input:** [`config/tickers.txt`](config/tickers.txt) — one `EXCHANGE:TICKER` per line.
  This is the *universe of record*. (Current universe: 6 indices, ~17 equities, 2
  commodity futures, 9 ETFs — defence/aero, agriculture, energy themes.)
- **What it does:** for each ticker, across **1H / 4H / 1D**, drives your local
  TradingView desktop over CDP (port 9222), waits for bars to settle, and extracts:
  - full **OHLCV** candles, and
  - **every** indicator plot the Prophets suite exposes — both *event* fires (sparse "this
    signal just fired" timestamps) **and** *continuous* values (oscillator levels, cloud
    distances, etc.).
- **Decision captured here (D5):** we keep **all** signals, even rare ones. Pruning is the
  *model's* job, not the collector's — because the whole point is to *measure* everything.
- **Output:** augmented `signal_study/results/effectiveness_<slug>.json`. Idempotent /
  resumable per ticker. This is the only step that touches the network.

> ✏️ **EDIT (universe/data):** The single biggest lever on results is **the data**. This
> universe is mostly a bull tape, which is *why* timing shows no edge out-of-sample. If we
> want timing to have a chance to pay, we'd collect a bear/choppy period or a wider, more
> regime-diverse universe. Want me to draft a second tickers list / date range?

### Step 2 · `build` — make the feature matrices + splits ([features.py](features.py))

This is where raw JSON becomes a clean, per-bar table the agent can read. **1D is the
clock** — one row per daily bar. Key things that happen, and the decisions baked in:

1. **Causal features only (the headline invariant).** Every column at bar *t* uses only
   data from bars ≤ *t*. Trailing returns, volatility, range-position, drawdown, ATR;
   per-signal `*_fired`, `*_bars_since`, `*_within_3`, `*_ever`; continuous `*_val`,
   `*_slope_3`; regime/condition counts (`bull_active_count`, `cloud_state`,
   `vol_regime`…); cross-TF confirmation (`htf_bull_confirm_4h` = a 4H same-side fire
   inside the current daily bar); and context (`asset_class` — **never raw ticker
   identity**, so the model can't memorize "this is LMT").
2. **The supervised target** (used only by the LightGBM baseline):
   `fwd_edge_6 = fwd_ret_6 − baseline_long_6` — the forward 6-bar return *minus* the
   ticker's own drift. It looks into the future *by design* and is NaN on the last 6 bars.
   The RL/evo agents never see this; they learn from reward instead.
3. **Train-only normalizers (D13).** Volatility terciles and continuous-value z-scores are
   fit on **training tickers only**, then applied everywhere. Fitting on all data would
   leak test information into training.
4. **The 3-way ticker split (D4)** — `splits.json`:
   - **training ≈ 70%** — the tickers the agent plays through to learn.
   - **validation ≈ 15%** — unseen tickers scored at every checkpoint to *pick the best
     model* / early-stop.
   - **test ≈ 15%** — unseen tickers touched **once**, at the very end, for the honest
     number.
   - Split is **over tickers** (not time) and **stratified by asset class** (each set spans
     indices/equities/commodities/ETFs), deterministic from `seed=42`.

> ✏️ **EDIT (features & splits):** lookback window `W`, target horizon `H`, the split
> ratios, and the stratification all live in [config.py](config.py)
> (`lookback=30`, `target_horizon=6`, `split_ratios=(0.70,0.15,0.15)`). The feature
> *vocabulary* (what counts as a signal, the condition columns) lives in features.py. If
> you want different context features (e.g. a market-regime flag), this is where to add it.

---

## 3. Part B — the environment (the actual "game") ([env.py](env.py))

One `SignalGymEnv` instance = **one ticker = one episode**. This single class is the spine:
the PPO trainer, the evolved population, the baselines, and the replay all step *the same
env*. Understanding one `step()` is understanding the whole gym.

**Setup (`reset`):** start at the first bar where a full lookback window exists
(`_first = lookback−1`), weight = 0 (all cash), equity = 1.0, and a B&H reference equity =
1.0 running in parallel.

**One `step(action)` — the heartbeat:**

```
              bar t (decision)         bar t+1 (fill happens here)
  obs ───────────►  agent picks  ───►  position rebalanced at t+1 OPEN
  (window of            w_t            then held to t+1 CLOSE
   features ≤ t)                       reward computed, equity updated
```

1. The agent has already seen `obs` = the feature window ending at bar *t* (rows
   `t−lookback+1 … t`). It outputs **w_t**, a target weight in [0, 1].
2. **No-lookahead fill (D2):** the weight you choose on bar *t* is realised over *t → t+1*
   using bar *t+1*'s prices — first the gap to next-bar **open** at the old weight, then
   the rest of the bar to next-bar **close** at the new weight, minus a turnover cost on
   the size of the change. You can never trade on a price you were shown.
   - `r = w_prev·(open_{t+1}/close_t − 1) + w_t·(close_{t+1}/open_{t+1} − 1) − cost·|w_t − w_prev|`
   - `cost = commission_bps/10000` (default 10 bps).
3. **Excess over buy-and-hold:** `excess = r − (close_{t+1}/close_t − 1)`.
4. **Reward = differential Sharpe of the excess** (see §4.2). The first `lookback` steps
   return reward 0 while the running mean/variance warm up.
5. `info` carries everything the visuals need: weight, equity, bh_equity, ret, excess,
   drawdown, reward, any trade, timestamp.

**Action space (a decision, D-/option):** continuous weight in [0,1] by default, or a
discrete `{0, 0.25, 0.5, 0.75, 1.0}` mode (`continuous_action` in config).

> ✏️ **EDIT (mechanics):** commission (`commission_bps=10`), the continuous-vs-discrete
> action, and the reward EMA decay (`reward_decay=0.02`, ~50-bar memory) are all in
> [config.py](config.py). The fill accounting is [env.py:112](env.py) (`step`). If you
> want, say, "only rebalance when a signal changes" (to cut turnover/noise) — that's an
> action-cadence change we'd make here. It's listed as open question #4 in the spec.

---

## 4. Deep dive — the reinforcement learning process

This is the part you asked to understand in depth. There are **three** learners, all
plugging into the same env. Two are "real" RL (learn from reward by playing); one is a
supervised sanity-baseline.

### 4.1 The three agents

| Agent | File | How it learns | Role |
|-------|------|---------------|------|
| **PPO** (neural net, gradients) | [agent_rl.py](agent_rl.py) + [train.py](train.py) | Reinforcement learning: policy-gradient updates from rollouts | **Primary** RL learner |
| **Evolved population** (small MLPs, no gradients) | [evo.py](evo.py) | Neuroevolution: mutate/select a *population* by fitness | **Watchable** RL — built to see learning happen |
| **LightGBM** (gradient-boosted trees) | [agent_sup.py](agent_sup.py) | Supervised one-shot fit on the `fwd_edge_6` target | Baseline + signal-attribution instrument |

All three expose the same tiny interface — `act(obs) → weight` — so the env, backtester,
evaluator, and replay don't care which one they're driving.

### 4.2 The reward — *why* it's shaped the way it is

Naïvely rewarding profit teaches "just hold" (B&H is +19%/yr). So:

- **Differential Sharpe** (Moeller): instead of summing returns, each bar's reward is that
  bar's *marginal contribution to a running Sharpe ratio* of the excess-over-B&H return.
  This rewards **consistent risk-adjusted** edge, not lucky spikes, and gives a smooth
  per-bar training signal. ([env.py `_diff_sharpe`](env.py))
- **Three reward modes** (config `reward_mode`, the RL training signal):
  - `excess` (default) — differential Sharpe of (agent − B&H). "Beat the market." Makes
    full-long the *safe* attractor; deviations must earn their keep.
  - `absolute` — differential Sharpe of the agent's *own* return. Full-long earns B&H's
    Sharpe; you beat it by dodging drawdowns.
  - `absolute_dd` — `absolute` minus a penalty on current drawdown.
- **The timing trap, and the fix (evolutionary mode's default).** Under plain ratio
  objectives the agent collapses to a *constant partial-exposure hold* (a de-risked B&H)
  and learns no timing — because ratios are scale-free and a drawdown penalty is cheapest
  to satisfy by simply holding *less, constantly*. The fix is **`timing_sortino`**
  ([stats.py `_timing_ratio`](stats.py)): score the agent against **its own
  average-exposure twin**.
  - `active_t = agent_ret_t − mean_weight · market_ret_t`
  - A constant-exposure agent **is** its twin → `active ≡ 0` → score **0**.
  - Lighter-into-drops / heavier-into-rallies (real, feature-driven timing) → score **> 0**.
  - Bad timing → score **< 0**.
  - This *cannot be faked by de-risking* (the twin de-risks too), which is exactly what we
    want to measure.

> ✏️ **EDIT (the objective):** this is the most important conceptual knob. The PPO loop
> selects on Sharpe-of-excess; the evolution defaults to `timing_sortino`. The full
> registry of selectable objectives is `OBJECTIVES` in [stats.py](stats.py) (sortino,
> sharpe, calmar, return, profit_factor, sqn, kelly, return_over_dd, timing_sharpe,
> timing_sortino…). Changing the objective changes *what we select for* — it can't conjure
> edge the data lacks, but it decides which behaviour wins.

### 4.3 PPO training loop — exact order ([train.py `_run_ppo`](train.py))

1. **Fit observation normalization** on training tickers only (per-feature mean/std),
   save to `ppo_obs_stats.npz`. *Without this the MLP ignores the raw-scale feature mix and
   collapses to a constant action* — a real, hard-won detail.
2. Build one PPO model (SB3 `MlpPolicy`, net `(64,64)`, entropy coef `0.01` to fight
   degenerate always-flat/always-100% policies on small data).
3. **The playthrough:** maintain a queue of training tickers. When empty, refill and
   **reshuffle** (each pass sees tickers in a new order). Pop a ticker → build its env →
   `model.learn(...)` for that episode's length → advance the timestep counter.
4. **Every `checkpoint_every` timesteps (default 20k):**
   - evaluate the current policy on **all validation tickers**, pooled → Sharpe-of-excess,
     beat-rate, abs-Sharpe vs B&H, mean max-drawdown;
   - **emit a checkpoint event to the live Monitor** (skill curve point);
   - save a checkpoint `ppo_ckpt_NNNN.zip`;
   - track the best-by-validation model; **early-stop** if validation hasn't improved for
     `early_stop_patience` (default 5) checkpoints.
5. **At the end:** reload the **best-by-validation** checkpoint, evaluate **once** on the
   **test** tickers → the honest generalization number. The CLI promotes that file to
   `ppo_best.zip`.

> RunConfig defaults: `total_timesteps=200_000`, `checkpoint_every=20_000`,
> `early_stop_patience=5`. PPO hyperparams in [config.py](config.py) `PPOParams`.

### 4.4 Evolution loop — exact order ([evo.py `evolve`](evo.py))

The watchable alternative. No gradients — a **population** of small MLP policies
(`EvoPolicy`, by default reading only the *current bar's* feature row) is improved by
selection. Per generation:

1. **Re-roll** a shared random ticker order (re-rolled each generation to average out
   order-luck).
2. **Run the whole population through that order** ([population.py `run_population`](population.py)),
   each agent **threading one continuous capital curve across all tickers** — a bad early
   ticker can sink you for the rest of the pass.
3. **Ruin rule:** long-only spot can't hit zero, so "broke" = threaded equity drops below
   `ruin_frac` (default **0.45**). A ruined agent's curve **ends there** and its fitness is
   `−∞` (sorted last).
4. **Fitness** = `objective_value(stats, "timing_sortino")` on the threaded curve.
5. **Select & breed:** keep the top `elite_frac` (25%) by *train* fitness (elitism); fill
   the rest with **mutated elites** (Gaussian noise, `mutation_sigma=0.1`).
6. **Val-gate:** evaluate the generation's *best* genome on a fixed validation order and
   record it. The search is driven by *train* fitness, but the **champion is the
   generation with the best *validation* fitness** — validation is never the breeding
   target (that would just overfit to it). Early-stop on val patience (8).
7. **Test once** on the champion at the very end.
8. **Record the final generation's race** to `evo_generation.{npz,json}` so `evo-replay`
   can render it later.

> EvoConfig defaults: `pop_size=64`, `n_generations=40`, `elite_frac=0.25`,
> `mutation_sigma=0.1`, `ruin_frac=0.45`, `objective="timing_sortino"`, `hidden=32`,
> `window_mode="last"`. All in [evo.py](evo.py) `EvoConfig`.

> ✏️ **EDIT (RL knobs):** population size, generations, mutation strength, the ruin line,
> the policy size, and whether the evolved net sees just the current bar (`"last"`) or the
> whole window (`"flatten"`) are all here. PPO depth/width/entropy are in PPOParams. Tell
> me what you want to experiment with and I'll expose it as a CLI flag.

---

## 5. Part C — evaluation, the edge report, and validation

These turn a trained agent into **knowledge** (the real deliverable).

### Step 5 · `eval` — score on unseen tickers ([train.py `eval_split`](train.py))
Runs the policy over every ticker in the chosen split, pools the per-bar excess returns,
and reports pooled Sharpe-of-excess, beat-rate (% of bars beating B&H), and drawdown.
`--scope validation` for the tuning number, `--scope test` for the honest one.

### Step 6 · `analyze` — the signal-edge report (D-B) ([analysis.py](analysis.py))
Assembles `reports/edge_report.json` from four lenses:
1. **Solo-signal edge** — each signal's forward edge vs the ticker's baseline, pooled over
   out-of-sample tickers, with sample size & significance.
2. **Condition / interaction edge** — edge of signals *gated by context*: confluence pairs
   (co-firing), regime buckets, cross-TF agreement, volatility regimes. Reports the *lift*
   over the better single signal.
3. **Model attribution (GBT)** — LightGBM gain importances + SHAP: which inputs the
   baseline leans on.
4. **RL ablation** — zero/shuffle one feature, re-evaluate the trained agent, measure the
   drop in Sharpe-of-excess: *which inputs the agent actually trades on.*

### Step 7 · `validate` — don't fool ourselves ([valid.py](valid.py))
- **MCPT** (Monte Carlo Permutation Test): shuffle the sign of excess returns thousands of
  times to get a **p-value** that the observed edge isn't chance.
- **Runs test** (Wald–Wolfowitz): is there serial dependence in trade outcomes?
- Both run on **pooled out-of-sample** data. This is where a null result gets stated
  honestly.

> ✏️ **EDIT (reporting):** what the edge report ranks and how the leaderboard prints
> ([run.py `cmd_report`](run.py)) is easy to reshape. If there's a specific "I want to see
> X signal vs Y condition" view, describe it and we'll add a section.

---

## 6. THE VISUALIZATION SEQUENCE  ← the part to amend

There are **four** distinct visual surfaces. For each: when it appears, its layout, what
every pane shows, where the data comes from, and what's currently rough. The ASCII sketches
are deliberately editable — redraw them and I'll rebuild the panes to match.

### 6.1 Visual #1 — Training Monitor (live, during `train`) ([monitor.py](monitor.py))

**When:** opens immediately when you pass `--monitor` to `train`; updates **once per
checkpoint** (PPO) or once at the end (sup). Runs on a background thread; the queue drains
every 200 ms. Closes when training ends.

**Layout — 2×2 grid:**

```
┌──────────────────────────────┬──────────────────────────────┐
│ A: Rollout Returns           │ B: Validation Sharpe (excess) │
│  bar chart, last 100 points  │  green line + dots, y=0 line  │
│  (per-rollout return)        │  THE "skill curve"            │
├──────────────────────────────┼──────────────────────────────┤
│ C: Equity vs B&H (last ep.)  │ D: Top Feature Importances    │
│  cyan = agent, orange = B&H  │  purple bars, re-sort as it    │
│  (dashed)                    │  learns                        │
└──────────────────────────────┴──────────────────────────────┘
```

- **A** — per-rollout return (bar). **B** — validation Sharpe-of-excess per checkpoint: the
  curve that should *rise* if the agent is learning. **C** — agent equity vs B&H on a
  representative validation ticker. **D** — top feature importances / ablation deltas.

> ⚠️ **Accuracy note (and a prime amendment target):** in the current code the **PPO**
> checkpoint only emits `val_sharpe` + `beat_rate`, so during PPO training **only panel B
> actually updates.** Panels A, C, D are wired but starved of data on the PPO path (the
> supervised path emits importances → D, but only once). So today the live PPO experience
> is essentially a single skill curve.
>
> ✏️ **EDIT:** I'd suggest we make `_run_ppo` emit (a) per-rollout return → A, (b) the
> best val ticker's equity/B&H arrays → C, and (c) periodic RL ablation deltas → D, so all
> four panels live up to the layout. Want me to wire that?

### 6.2 Visual #2 — Single-ticker Replay (`replay TICKER`) ([replay.py](replay.py))

**When:** you run `replay NYSE:LMT` after training. Opens a finplot window showing one
agent trading one ticker.

**Layout — 3 stacked panes:**

```
┌───────────────────────────────────────────────────────────┐
│ PRICE  candlesticks                                        │
│   ▲ green = bull signal fired    ▼ red = bear signal fired │
│   ▲ green entry arrow            ▼ red exit arrow          │
├───────────────────────────────────────────────────────────┤
│ EQUITY   cyan = agent      orange dashed = buy & hold      │
├───────────────────────────────────────────────────────────┤
│ POSITION   blue line + shaded fill, 0 = cash … 1 = full    │
│   (the heart of it: watch exposure rise and fall)         │
└───────────────────────────────────────────────────────────┘
```

> ⚠️ **Accuracy note:** despite the "bar-stepped / speed" wording, `replay()` currently
> **draws the entire chart at once** (you scrub/zoom it; it is *not* animated bar-by-bar).
> A `replay_stepped()` exists but is a naïve redraw loop and isn't wired to the CLI. So
> "watch it play out live" is, today, "scrub a fully-drawn chart."
>
> ✏️ **EDIT:** if you want a true **animated** playback (bars revealing left-to-right at N
> bars/sec, position pane filling as it goes, a live PnL/reward readout in the corner),
> that's a real feature to build — and it sounds like the experience you're picturing. Say
> the word and I'll spec it.

### 6.3 Visual #3 — Evolution Monitor (live, during `evolve --monitor`) ([evo_monitor.py](evo_monitor.py))

**When:** opens when you run `evolve --monitor`; redraws **once per generation** (per-bar
would throttle training). Qt runs on the main thread; evolution runs on a worker thread;
queue drains every 150 ms.

**Layout — population race on top, two diagnostics below:**

```
┌───────────────────────────────────────────────────────────┐
│ A: Population equity race   (log-y)                        │
│   thin grey  = each agent      red = ruined (curve ends)   │
│   BOLD blue  = champion        orange dashed = B&H         │
│   dark dashed = ruin line (0.45)                           │
├──────────────────────────────┬────────────────────────────┤
│ B: Skill curve / generation  │ C: Leaderboard             │
│   green = best train fitness  │  horizontal purple bars,   │
│   grey  = mean train fitness  │  top agents by objective   │
│   blue  = validation fitness  │                            │
└──────────────────────────────┴────────────────────────────┘
```

- **A** is the showpiece: a crowd of portfolios racing through the stitched universe,
  curves dying at the ruin line. **B** tells you whether the population is *improving* (and
  whether train and val agree — they diverging is the overfitting story). **C** ranks the
  current top agents.

> This is the most "alive" visual and the one closest to your goal of *watching learning
> happen*. It's wired and runs locally; the spec calls hardening this live attach the next
> milestone (P4).

### 6.4 Visual #4 — Evolution Replay (recorded, `evo-replay`) ([evo_replay.py](evo_replay.py))

**When:** after an `evolve` run recorded a generation. Re-renders the **final generation's**
race from the saved `.npz`. The leaderboard prints to the console; two panes draw:

```
console: === leaderboard (objective=timing_sortino) ===
          1  agent 07   fit=+0.42  wstd=0.37  turnover=14.2  <= champion
          …

┌───────────────────────────────────────────────────────────┐
│ WORLD  (log-y)  stitched candles of all tickers in pass    │
│   each ticker segment rebased to start at 1.0              │
│   ▲ bull signal     ▼ bear signal                         │
├───────────────────────────────────────────────────────────┤
│ RACE   (log-y)   same population race as Visual #3 panel A │
│   grey agents · red ruined · bold blue champion ·          │
│   orange dashed B&H · dark dashed ruin line                │
└───────────────────────────────────────────────────────────┘
```

- The **world** pane shows the *single shared price stream* every agent traded (24 tickers
  stitched end-to-end, each rebased to 1.0 so cheap grains and a 52,000 index share an
  axis), with Prophets signal markers — so you can line up *what the agents saw* against
  *how they did* in the race pane below.

> ⚠️ **Accuracy note:** `evo-replay` also **draws everything at once** (the `--speed`
> flag is accepted but not used to animate). Same situation as Visual #2.
>
> ✏️ **EDIT:** the obvious upgrade is a synchronized animation — world bars reveal
> left-to-right while the race curves draw in lockstep and ruined agents wink out at their
> death bar. Plus a per-generation scrubber to step gen 0 → final and watch the crowd
> improve. If the *watchable race* is the centerpiece for you, this is where I'd invest.

### 6.5 Visualization summary & open questions for you

| # | Visual | Trigger | Cadence | Animated today? | Biggest gap |
|---|--------|---------|---------|-----------------|-------------|
| 1 | Training Monitor | `train --monitor` | per checkpoint | live updates, but PPO feeds only panel B | wire panels A/C/D for PPO |
| 2 | Replay | `replay TICKER` | once | **no** (static, scrubbable) | true bar-by-bar animation + readout |
| 3 | Evo Monitor | `evolve --monitor` | per generation | **yes** (redraws each gen) | harden P4; add scrubber |
| 4 | Evo Replay | `evo-replay` | once | **no** (static, scrubbable) | synchronized animated race + gen scrubber |

> ✏️ **DECIDE WITH ME:** which of these is "the" visualization you care most about? My
> read is the **evolution race (3/4)** — a watchable crowd surviving and being culled — is
> the soul of the project, and an *animated, scrubbable* version is the thing to build. But
> if it's the **single-agent replay (2)** trading one chart in real time, that's a
> different (also great) build. Pick one and we'll design it in detail here before touching
> code.

---

## 7. Decisions & knobs — where to change what

Everything you might want to tune, and the file it lives in. (Defaults shown.)

| Knob | Default | Lives in | Effect |
|------|---------|----------|--------|
| `lookback` (obs window W) | 30 bars | [config.py](config.py) | how much history the agent sees |
| `target_horizon` (H) | 6 bars | [config.py](config.py) | forward window for the supervised target |
| `commission_bps` | 10 | [config.py](config.py) | turnover cost per unit weight change |
| `reward_decay` | 0.02 | [config.py](config.py) | differential-Sharpe memory (~50 bars) |
| `reward_mode` | `excess` | [config.py](config.py) | PPO training signal (excess / absolute / absolute_dd) |
| `continuous_action` | `True` | [config.py](config.py) | continuous weight vs discrete 5-step |
| `split_ratios` | 70/15/15 | [config.py](config.py) | train / validation / test sizes |
| `seed` | 42 | [config.py](config.py) | reproducible splits & runs |
| PPO net / entropy | (64,64) / 0.01 | [config.py](config.py) `PPOParams` | policy capacity & exploration |
| `total_timesteps` / `checkpoint_every` / `early_stop_patience` | 200k / 20k / 5 | [train.py](train.py) `RunConfig` | PPO training length & cadence |
| evolution `pop_size` / `n_generations` | 64 / 40 | [evo.py](evo.py) `EvoConfig` | population search budget |
| `elite_frac` / `mutation_sigma` | 0.25 / 0.1 | [evo.py](evo.py) `EvoConfig` | selection pressure & exploration |
| `ruin_frac` | 0.45 | [evo.py](evo.py) `EvoConfig` | the death line in the race |
| `objective` | `timing_sortino` | [evo.py](evo.py) / [stats.py](stats.py) | **what we select for** |
| `window_mode` | `last` | [evo.py](evo.py) | evolved net sees current bar vs whole window |
| asset-class map | per ticker | [config.py](config.py) `TICKER_ASSET_CLASS` | the only context feature |

---

## 8. The honest state of the science (so the visuals don't oversell)

From [evo_spec.md](evo_spec.md), reconfirmed by two independent optimizers (PPO and ES):

- Under **return/Sortino** objectives, the champion **collapses to a constant
  partial-exposure hold** — a de-risked buy-and-hold, `weight_std ≈ 0`, *no timing*.
- Under **`timing_sortino`**, the collapse breaks — agents genuinely vary exposure
  (`weight_std ≈ 0.37`) and **train** timing-fitness improves across generations — **but it
  overfits**: validation bounces and held-out **test timing-fitness is negative**.
- **Conclusion:** the features contain timing patterns that fit in-sample but carry **no
  real out-of-sample edge** on this bull universe. That in-sample-good / out-of-sample-bad
  gap *is the watchable result.*
- **The only path to a positive timing result is the data** — a bear/choppy regime where
  timing genuinely pays. That's a collection task, not a modelling trick.

> ✏️ **EDIT:** if you'd rather the project's framing be "find the regimes/conditions where
> a signal *does* predict" (a measurement goal) rather than "build a market-beating agent"
> (a performance goal), say so — it changes which numbers the visuals foreground.

---

## 9. Glossary (quick reference)

- **Weight / exposure** — fraction of capital in the ticker, 0…1. The agent's only lever.
- **Excess return** — agent return minus buy-and-hold return that bar.
- **Differential Sharpe** — per-bar marginal contribution to a running Sharpe ratio; the
  smooth reward signal.
- **Timing fitness (`timing_sortino`)** — risk-adjusted skill *vs the agent's own
  average-exposure twin*; 0 means "no timing, just held a constant amount."
- **Threaded capital** — in evolution, one equity curve compounded across *all* tickers in
  a pass (so early losses haunt you).
- **Ruin line** — equity level (0.45) below which an evolved agent is declared dead.
- **Val-gating** — choosing the champion by validation, never by the thing we optimized
  (train), to avoid overfitting the selection.
- **MCPT** — permutation test for "is this edge real or chance?"

---

*End of walkthrough. Edit freely — every `> ✏️ EDIT` is an invitation. When you've marked
up the visuals (or told me which one is "the" one), I'll turn the agreed changes into code.*
