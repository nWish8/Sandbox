# Technical Spec: Signal Gym

**Status:** Draft (rev 3 — RL/episodic reframe + signal-edge discovery as outcome)
**Linked PRD:** prd.md
**Last updated:** 2026-06-27

## 0. Revision notes

**Rev 3** realigns the spec with the project's actual plan and outcome (Nick's
direction). The previous rev treated v1 as a supervised batch-fit with RL deferred;
that mis-stated the plan. The real plan and the real *deliverable*:

- **The agent learns by playing through the market, not by fitting a matrix.** A
  *training run* is a randomly-ordered playthrough of the **training** ticker set; the
  agent manages a long-only spot portfolio bar-by-bar from strictly-live inputs;
  performance is evaluated at checkpoints on the **test** and **validation** ticker
  sets. This is a reinforcement-learning / online-policy loop and is now the **primary
  paradigm** (was "v2").
- **Three disjoint ticker sets — training / test / validation —** replace rev-2's
  "train + internal time-split + held-out". Splits are over *tickers* (generalization
  to unseen tickers is the point).
- **The outcome is knowledge, not just a P&L.** The project exists to *discover and
  record which Prophets signals, conditions, and signal-to-signal interactions carry
  edge with respect to price action and movement.* A **signal-edge report** is a
  first-class deliverable, not a side metric.
- **Capture the whole Prophets signal suite.** Rev-2 pruned rare signals from the data
  layer to fight overfitting; that conflicts with the goal of measuring *everything*.
  Rev 3 **collects every plot the indicators expose** — sparse event fires *and*
  continuous values (oscillator levels, cloud distances) — at each bar. Feature
  selection / regularization is the **model's** job, not the collector's; rare-signal
  edge is still *recorded* even when too sparse to trade.
- **The supervised edge-model is retained**, repositioned as (a) a fast, interpretable
  **baseline policy** and (b) the **signal-attribution instrument** feeding the
  edge report. It is not the primary learner. (This keeps a supervised-first build
  order available if Nick wants to validate the pipeline before standing up RL — see
  §11 D26.)
- **The env reward is now a real training signal** (the RL objective), not the
  "evaluation-only" metric of rev 2.

Superseded rev-2 decisions: D17 (data-layer pruning — reversed), D20 (reward
eval-only — reversed). Carried forward: causality/leak guards (D2, D13), parquet (D7),
finplot+pyqtgraph (D9), differential-Sharpe reward shape (D3).

## 1. Overview

Signal Gym is a Python-only, local research framework with two intertwined goals:

1. **Train a general agent** that manages a long-only spot portfolio on any single
   ticker, learning *online* by playing through a training universe of tickers and
   deciding, bar-by-bar from live Prophets-signal + price inputs, how much to hold.
2. **Discover and record the edge structure of the Prophets signal suite** — which
   signals, conditions (regimes/contexts), and interactions (confluence, cross-TF
   agreement) actually predict price movement — using both classical per-signal
   statistics and the trained agent's own attributions.

A *training run* shuffles the training tickers and plays each through as an episode;
at checkpoints the agent is evaluated on the test and validation ticker sets. A live
**training monitor** (pyqtgraph) shows the agent getting better as it learns, and a
**finplot replay** shows a trained agent trading a single ticker bar-by-bar. Every
reported result is statistically validated (MCPT, runs test) on pooled out-of-sample
data, and every run produces a **signal-edge report**.

Stack: Python 3.11+, pandas/numpy, Gymnasium + stable-baselines3 (PPO, the primary
learner), LightGBM (baseline + attribution), finplot + pyqtgraph (PyQt5), scipy,
shap. Runtime is offline; the only network step is the build-time data scrape.

### Primary deliverables
- **D-A. Trained general agent** (PPO policy) + checkpoints + metrics on test/validation.
- **D-B. Signal-edge report** (`gym/reports/edge_report.json` + printed summary):
  ranked solo-signal edge, condition/interaction edge, and model attributions
  (GBT importances/SHAP, RL feature-ablation deltas) — the answer to "what gives edge."
- **D-C. Reusable gym** that both the PPO agent and the supervised baseline plug into
  unchanged, plus the live monitor and finplot replay.

## 2. Architecture

```
gym/config/tickers.txt   (user-provided ticker list — the scrape input)
        │
        ▼
┌──────────────┐  TradingView CDP (build-time only)
│  collect.py  │  scrape OHLCV + ALL indicator plots (fires + continuous) per ticker/TF
└──────┬───────┘  → signal_study/results/*.json  (augmented: OHLCV + values)
        ▼
┌──────────────┐
│  features.py │  per-bar aligned matrices (all signals/values) + 3-way ticker splits
└──────┬───────┘  → gym/data/*.parquet, gym/data/splits.json
        ▼
┌──────────────┐
│   env.py     │  Gymnasium env over ONE ticker; obs = live feature window;
│              │  action = target weight [0,1]; next-bar-open fill + costs;
│              │  reward = differential-Sharpe excess-over-B&H − turnover
└──────┬───────┘
        │  (one env instance per episode = per ticker playthrough)
        ▼
┌───────────────────────────────────────────────┐
│              train.py  (the playthrough loop)   │
│  shuffle training tickers → episode rollouts →  │
│  policy update → checkpoint → eval test+val     │
└───┬───────────────┬───────────────┬─────────────┘
    ▼               ▼               ▼
┌────────┐   ┌──────────────┐  ┌──────────────┐
│agent_rl│   │ agent_sup.py │  │  monitor.py  │  live pyqtgraph dashboard
│  .py   │   │ (baseline +  │  │ (episodes/   │  (loss/return, val Sharpe,
│ (PPO)  │   │  attribution)│  │  timesteps)  │   live equity, importances)
└────────┘   └──────┬───────┘  └──────────────┘
                    ▼
            ┌──────────────┐   ┌──────────┐   ┌──────────────────┐
            │  analysis.py │   │ valid.py │   │    replay.py     │
            │ edge report  │   │ MCPT +   │   │ finplot bar-step │
            │ (D-B)        │   │ runs test│   │ candles+trades+eq│
            └──────────────┘   └──────────┘   └──────────────────┘
```

Orchestration: `gym/run.py` — CLI: `collect`, `build`, `train`, `eval`, `analyze`,
`validate`, `replay`, `report`.

### Components
- **collect.py** — Read `tickers.txt`; for each ticker drive TradingView (existing CDP
  path) across 1H/4H/1D; scrape OHLCV **and every indicator plot** — sparse fires and
  continuous values — into the augmented `results/*.json`. Build-time, network. Owns
  the scrape and the raw stored data.
- **features.py** — Build per-bar feature matrices from the augmented JSONs (all
  signals + values + conditions), compute the supervised target, fit train-only
  normalizers, write parquets + `splits.json` (3-way ticker split). Owns the feature
  schema and causality guarantees.
- **env.py** — Gymnasium `Env` over one ticker. Streams a live feature window, accepts
  a target weight, fills at next-bar open with cost, returns the differential-Sharpe
  excess reward. One instance per episode. Owns observation/action spaces, reward,
  episode state. Doubles as the replay stepper.
- **agent_rl.py** — PPO policy (stable-baselines3) — the **primary** learner. Wraps the
  env for vectorized rollouts; exposes `act(obs)` for eval/replay. Owns the policy net
  + checkpoints.
- **agent_sup.py** — LightGBM edge-model: trains on pooled labeled experience; serves
  as a baseline policy and as the attribution instrument (importances, SHAP). Owns the
  GBT artifact and edge→weight mapping.
- **train.py** — The playthrough loop (§5.3): shuffle training tickers, run episode
  rollouts, update the policy, checkpoint, evaluate on test+validation, stream
  `TrainingEvent`s to the monitor. Owns the training protocol + early-stopping on
  validation.
- **monitor.py** — Live pyqtgraph dashboard keyed on episodes/timesteps (§5.7).
- **analysis.py** — Builds the **signal-edge report** (D-B): solo edge, condition /
  interaction edge, GBT attributions, and RL ablation deltas. Owns `edge_report.json`.
- **baselines.py** — Buy-and-hold + logged rules (A3, A5b, Prophet Confluence Buy) as
  env-compatible policies (benchmarks the agent must beat).
- **backtest.py** — Run any policy over a ticker → trades, equity, metrics; plus a
  vectorized fast-path used by the monitor and by analysis ablations.
- **valid.py** — MCPT (sign-flip) + Wald–Wolfowitz runs test on pooled OOS results.
- **replay.py** — finplot bar-stepped chart of a trained policy on one ticker.
- **config.py** — One `Config` dataclass: tickers path, TFs, horizons, lookback,
  costs, reward decay, PPO hyperparams, GBT params, split ratios, seed.

## 3. Data model

### 3.1 Ticker list (input)
`gym/config/tickers.txt` — one `EXCHANGE:TICKER` per line (comments with `#`). This is
the scrape manifest and the universe of record; splits are derived from it
deterministically.

### 3.2 Collected data (collect.py → augmented `signal_study/results/*.json`)
Per timeframe, extend the existing format to capture **everything the indicators
expose**:
```python
{
  "label": "1D", "resolution": "D", "ohlc_bars": 467,
  "series": {"t":[...], "o":[...], "h":[...], "l":[...], "c":[...], "v":[...]},
  "baseline": {"long": {...}, "short": {...}},     # per-ticker drift (target only)
  "signals": {
    "<study>|<plot>": {
      "direction": "long|short|n/a",
      "kind": "event|continuous",                  # NEW: sparse fire vs continuous value
      "fires": [unix_ts, ...],                      # event plots
      "values": {"t":[...], "v":[...]} | null,      # NEW: continuous plots (causal, value@bar)
      "fires_total": int, "fires_in_window": int,
      "by_horizon": {...}, "edge_by_horizon": {...} # study stats, retained
    }, ...
  }
}
```
The current extractor already reads full OHLCV (it keeps only `c`) and already
enumerates every titled plot (it keeps only those firing <40% of bars as "signals").
collect.py keeps **all** of it: OHLCV, event fires, and continuous values; near-
continuous plots become `continuous` with their per-bar value.

### 3.3 Feature matrix (features.py → `gym/data/{slug}.parquet`)
1D clock, one row per bar. **All signals represented** (no data-layer pruning).
Dtypes float32 / int64 ts / float64 prices. The observation the env feeds the agent is
a `lookback`-bar window of the model-input columns (everything except meta/exec/target).

| Group | Columns | Notes |
|------|---------|-------|
| meta | `ticker, timestamp, open, high, low, close, volume` | raw bar |
| exec | `fill_price` = next-bar `open` (`open.shift(-1)`; else `close.shift(-1)`) | next-bar only |
| price | `ret_1,3,6`, `vol_20`, `range_pos_20`, `drawdown_20`, `atr_14` | trailing only |
| per event-signal (**all** event plots) | `{sig}_fired`, `{sig}_bars_since` (cap 50), `{sig}_ever`, `{sig}_within_3` | bars ≤ t |
| per continuous-signal (**all** continuous plots) | `{sig}_val` (z-scored, train-fit), `{sig}_slope_3` | value@bar t (causal) |
| conditions / regime | `bull_active_count`, `bear_active_count`, `continuation_active`, `cloud_state∈{-1,0,1}`, `background_active`, `points_active`, `vol_regime∈{0,1,2}` | trailing; terciles train-fit |
| interactions | `htf_bull_confirm_4h`, `htf_bear_confirm_4h` (4H same-side fires within 1 daily bar) | causal |
| context | `asset_class∈{0..3}` (index/equity/commodity/ETF) | static; **no raw ticker id** |
| target (supervised only) | `fwd_edge_6 = fwd_ret_6 − baseline_long_6` (NaN last 6 bars), `fwd_ret_6` | label (future by design) |

Notes:
- Capturing all signals raises dimensionality — that is intentional for *measurement*.
  Overfitting is controlled at the model (PPO regularization, GBT `feature_fraction`/
  `min_child_samples`/`lambda`), and the analysis ablations tell us which features the
  models actually rely on. Rare signals that can't be traded still get **recorded edge
  stats** in the report.
- Raw ticker identity is never a feature (PRD). `asset_class`, `vol_regime` are the
  only context. The per-ticker `baseline` is used only to define the supervised target.
- Continuous values are causal (indicator value at bar t uses only data ≤ t) and are
  z-scored with **train-set** statistics stored in `splits.json`.

### 3.4 Splits (features.py → `gym/data/splits.json`)
```python
{
  "training_tickers":   [...],   # ~70%  — episodes played during a training run
  "validation_tickers": [...],   # ~15%  — checkpoint eval; drives early-stop/selection
  "test_tickers":       [...],   # ~15%  — final, untouched; reported once
  "stratify": "asset_class",     # each set spans all asset classes (deterministic by seed)
  "normalizers": {"vol_tercile_edges":[lo,hi], "cont_zscore":{...}},  # TRAIN-fit only
  "seed": 42
}
```
Roles (standard ML meaning, clarifying Nick's "test and validation are used"):
**validation** = evaluated every checkpoint to pick the best model / early-stop;
**test** = touched once at the end for the honest generalization number. Both are
*unseen tickers* — the generalization test the PRD demands.

### 3.5 Artifacts
`gym/models/ppo_{run}.zip` (+ checkpoints), `gym/models/edge_model.lgbm`,
`gym/models/meta.json` (full config, feature list, split, metrics, git commit),
`gym/reports/edge_report.json` (D-B).

## 4. API / Interface contracts

### 4.1 collect.py
```python
def collect(tickers_file=TICKERS, timeframes=("60","240","D"), out=RESULTS) -> list[str]:
    """Scrape OHLCV + all indicator plots (fires + continuous values) per ticker/TF
    via the TradingView CDP path. Writes augmented results/*.json. Returns slugs."""
```

### 4.2 features.py
```python
def build_features(results_dir=RESULTS, out=GYM_DATA, config=CFG) -> dict[str,pd.DataFrame]
def make_splits(tickers, ratios=(0.70,0.15,0.15), stratify="asset_class", seed=42) -> dict
def load_ticker(slug, data_dir=GYM_DATA) -> pd.DataFrame
def feature_columns(df) -> list[str]
```

### 4.3 env.py — `SignalGymEnv(gymnasium.Env)`
```python
class SignalGymEnv(gymnasium.Env):
    def __init__(self, ticker_df, lookback=30, commission_bps=10.0,
                 reward_decay=0.02, continuous=True): ...
    observation_space: Box        # (lookback, n_features) float32 — LIVE window only
    action_space: Box | Discrete  # [0,1] or {0,.25,.5,.75,1}
    def reset(self, seed=None, options=None) -> (obs, info)
    def step(self, action) -> (obs, reward, terminated, truncated, info)
        # decide w_t on bar t; fill at t+1 open w/ cost; reward = differential-Sharpe
        # of excess-over-B&H − turnover (§5.4). info: position, weight, equity,
        # bh_equity, trade, reward, bar_idx, active_signals.
    def get_equity_curve(self) -> pd.DataFrame
```

### 4.4 agent_rl.py
```python
def make_vec_env(ticker_dfs, config) -> VecEnv          # episodes sampled across tickers
def train_ppo(train_dfs, val_dfs, config, monitor=None) -> PPO
    """Playthrough training (delegates the loop to train.py). Early-stops on val Sharpe."""
class RLPolicy:                                          # wraps PPO for eval/replay
    def act(self, obs) -> float
```

### 4.5 agent_sup.py
```python
def train_edge_model(train_df, feature_cols, target="fwd_edge_6", config=CFG, monitor=None) -> lgb.Booster
def calibrate_sizing(model, train_df, feature_cols, config) -> SizingParams   # TRAIN-only
def edge_to_weight(edge_pred, sizing) -> float
def feature_importances(model) -> pd.DataFrame
def shap_attributions(model, df, feature_cols) -> pd.DataFrame
class EdgePolicy:                                        # baseline policy
    def act(self, obs) -> float
```

### 4.6 train.py — the playthrough loop
```python
@dataclass
class RunConfig:
    total_timesteps: int = 1_000_000
    checkpoint_every: int = 20_000
    reshuffle_each_pass: bool = True
    early_stop_patience: int = 5     # checkpoints w/o val improvement
def training_run(train_dfs, val_dfs, test_dfs, config, monitor=None) -> RunResult:
    """Shuffle training tickers → episode rollouts → policy update → checkpoint →
    eval val (+ test at the end). Returns best policy + per-checkpoint history."""
```

### 4.7 analysis.py — signal-edge report (D-B)
```python
def build_edge_report(results, models, splits, out=REPORTS) -> dict:
    """Assemble: (1) solo-signal edge (study stats, pooled OOS);
    (2) condition/interaction edge (confluence pairs, regime-gated buckets);
    (3) GBT importances + SHAP; (4) RL feature-ablation deltas. Writes edge_report.json."""
def ablate_feature(policy, eval_dfs, feature_col, config) -> float
    """Zero/shuffle one feature, re-eval, return ΔSharpe-of-excess (the feature's value
    to the trained agent)."""
```

### 4.8 backtest.py / valid.py / replay.py / monitor.py
```python
def run_backtest(env, policy) -> BacktestResult        # trades, equity_curve, metrics
def fast_vectorized(policy_or_model, df, config) -> dict
def mcpt(excess_returns, n_perms=10_000, metric_fn=sharpe, seed=42) -> MCPTResult   # POOLED OOS
def runs_test(trade_returns) -> RunsTestResult
def replay(result, ticker_df, signal_cols, speed=1.0) -> None
class TrainingMonitor: ...     # emit(TrainingEvent); 4-panel pyqtgraph dashboard
```

### 4.9 run.py — CLI
```
python -m gym.run collect                       # scrape tickers.txt → results/*.json
python -m gym.run build                          # results → parquets + splits
python -m gym.run train [--agent ppo|sup] [--monitor] [--seed 42]
python -m gym.run eval  [--scope validation|test] [--agent ppo|sup]
python -m gym.run analyze                        # build edge_report.json (D-B)
python -m gym.run validate [--scope test] [--perms 10000]
python -m gym.run replay TICKER [--agent ppo] [--speed 2]
python -m gym.run report [--scope all|training|validation|test]
```

## 5. Key flows

### 5.1 Collect (the scrape)
Read `tickers.txt`; per ticker, per TF: switch symbol/timeframe, wait for bars to
settle (existing `stable_bars`), extract OHLCV + **all** plots (events + continuous),
write augmented JSON. Idempotent / resumable per ticker.

### 5.2 Build features + splits
Build all-signal feature matrices (trailing/causal), compute supervised target, fit
**train-only** normalizers (vol terciles, continuous z-scores), derive the 3-way
stratified ticker split, write parquets + `splits.json`.

### 5.3 Training run — the central loop (Stories 2,4,5; Nick's plan)
1. Load training/validation/test parquets. Build the env factory (one env per ticker).
2. **Shuffle the training tickers** (reshuffle each pass). Roll out episodes: the agent
   steps each ticker's bars, seeing only the live `lookback` window, choosing a target
   weight, receiving the differential-Sharpe excess reward. PPO updates on the
   collected rollouts.
3. **At each checkpoint** (`checkpoint_every` timesteps): evaluate the current policy on
   **all validation tickers** (pooled Sharpe-of-excess, beat-rate), emit a
   `TrainingEvent` to the monitor, save a checkpoint. Early-stop on validation patience.
4. **At the end**: load the best (by validation) checkpoint, evaluate **once** on the
   **test** tickers for the honest generalization number. Save policy + meta.
   (`--agent sup` swaps the PPO learner for the LightGBM baseline, evaluated under the
   same checkpoint/eval protocol — so the pipeline is exercisable before RL is tuned.)

### 5.4 Reward / fill accounting (env + vectorized, identical)
Weight `w_t` decided on bar t fills at bar t+1 open:
`r_{t+1} = w_{t-1}(open_{t+1}/close_t − 1) + w_t(close_{t+1}/open_{t+1} − 1) − c·|w_t−w_{t-1}|`,
`c = commission_bps/1e4` (no OHLC ⇒ `open:=prior close`). Excess
`x_{t+1}=r_{t+1}−(close_{t+1}/close_t−1)`. Reward = differential Sharpe of `x`:
`D_t=(B_{t-1}δA − ½A_{t-1}δB)/(B_{t-1}−A_{t-1}²)^{3/2}`, EMA updates with `η=reward_decay`,
denominator floor ε, `lookback`-bar warmup. This is the **RL training signal**.

### 5.5 Signal-edge discovery (D-B; Stories — the project outcome)
1. **Solo edge:** per-signal forward edge vs per-ticker baseline, pooled over OOS
   tickers (reuse/extend `signal_study` stats), with sample-n and significance.
2. **Condition / interaction edge:** edge of signals *gated by condition* — confluence
   pairs (co-fire within tol), regime buckets (e.g. divergence × continuation-active),
   cross-TF agreement, vol_regime buckets. Reports lift over the better solo leg.
3. **Model attribution:** GBT gain importances + SHAP (which inputs the edge-model
   leans on); **RL ablation** — zero/shuffle each feature, measure ΔSharpe-of-excess of
   the trained agent (which inputs it actually trades on).
4. Rank and write `edge_report.json` + a printed leaderboard: *"these signals /
   conditions / interactions carry edge; these don't."*

### 5.6 Validate
Pooled OOS excess/trade returns → `mcpt` (p-value the edge isn't chance) + `runs_test`
(serial dependence). Report per-set beat-rate vs B&H. Flag pass/fail vs PRD thresholds.

### 5.7 Watch training live + replay
Monitor (pyqtgraph, x-axis = timesteps/episodes): **A** rollout return / policy loss,
**B** validation Sharpe-of-excess per checkpoint (the skill curve), **C** live
equity-vs-B&H on a representative validation ticker, **D** top feature importances /
ablation deltas re-sorting as training proceeds. Replay (finplot): candles, signal
markers, position shading, trade arrows, equity-vs-B&H sub-pane, readout; ≥10 bars/sec.

## 6. External dependencies

| Library | Purpose | Version |
|---------|---------|---------|
| pandas, numpy, pyarrow | data, parquet | ≥2.0 / ≥1.24 / ≥14 |
| gymnasium | env API | ≥0.29 |
| stable-baselines3 | PPO (primary learner) | ≥2.3 |
| torch | SB3 backend (CPU) | ≥2.2 |
| lightgbm | baseline + attribution | ≥4.0 |
| shap | feature attribution | ≥0.45 |
| finplot, pyqtgraph, PyQt5 | replay + monitor | ≥1.9 / ≥0.13 / ≥5.15 |
| scipy | runs test, stats | ≥1.11 |

Runtime offline; the only network step is build-time `collect` (local TradingView CDP,
port 9222, via the existing Node CLI under `tradingview-mcp/`).

## 7. Security
Minimal threat model — single-user local app. No runtime network, no auth, no secrets,
no PII; inputs are a local ticker file + CLI args + pre-collected history. Paths from
constants; no `eval`/dynamic exec; no sockets/HTTP. Seeds explicit. The `collect`
step talks only to the user's own localhost TradingView over CDP.

## 8. Performance & scaling
Single user, offline, ~34 tickers × ~460 1D bars. Collect: minutes (UI-driven,
build-time). Feature build < 5 s. **PPO training is the heavy path** — CPU PPO over
~24 short episodes/pass for ~1e6 timesteps is minutes-to-low-hours; checkpointed and
monitorable, early-stopped on validation. GBT baseline trains in < 2 s. Per-ticker
eval < 0.1 s; MCPT 10k perms ≈ 2 s; ablation = one fast eval per feature. Monitor and
replay are interactive-smooth (pyqtgraph/finplot). Data fits in memory many times over;
no caching/parallelism needed beyond SB3's vectorized envs.

## 9. Observability
`logging` (`--verbose`); stdout for load counts, feature shape, per-checkpoint val
metrics, final test metrics, edge-report leaderboard, validation p-values. `meta.json`
records full config + feature list + metrics + git commit. The training monitor is the
live observability surface; `edge_report.json` is the persistent research output. Fail
fast on data issues; the only documented fallback is close-only when OHLC is absent.

## 10. Testing strategy

| Layer | Type | What |
|-------|------|------|
| collect | unit | augmented JSON shape; events vs continuous classified right; OHLCV captured |
| features | unit | known JSON → expected features; `bars_since`/counts/`cloud_state`; continuous z-score |
| features | **leakage** | (a) feature@t uses bars ≤ t only; (b) normalizers fit on train only — perturbing test values can't change train rows; (c) target NaN last H bars |
| env | unit | known actions → expected weight/equity/reward; **fill at t+1**; cost on \|Δw\|; warmup |
| env | **leakage** | obs@t excludes t+1; env loop == `fast_vectorized` |
| train | unit | training run shuffles training tickers; val drives early-stop; **test touched once**; determinism given seed |
| agent_rl | smoke | PPO learns on a planted-edge synthetic ticker (val Sharpe rises) |
| agent_sup | unit | planted-edge synthetic → that feature tops importance; `edge_to_weight` monotone |
| analysis | unit | ablation ΔSharpe sign correct on planted feature; report schema valid |
| valid | unit | MCPT random→p≈0.5, strong→p≈0; runs test IID→non-sig |
| pipeline | integration | collect(mock)→build→train(sup, short)→analyze→validate on 2–3 tickers |
| replay/monitor | manual | visual: trades align, equity tracks, skill curve updates, controls work |

**No-lookahead is the single most critical property** — guarded by feature causality,
train-only normalizer fitting, t+1 fills, and the live-window observation contract.
Coverage ≥ 90% on features.py and env.py.

## 11. Decisions log
> **D1.** PPO (stable-baselines3) is the **primary learner**; the env's reset/step
> playthrough is the training spine. Online episodic learning matches the plan (agent
> manages a portfolio from live inputs, learning by playing through tickers). Pure
> supervised batch-fit rejected as the primary path — it doesn't "play through" or
> manage a sequential portfolio.

> **D2 (carried).** Fill at next-bar open (else next-bar close), explicit accounting
> (§5.4). Deciding and filling on the same close is lookahead — forbidden.

> **D3 (carried).** Reward = differential-Sharpe of excess-over-B&H, turnover-
> penalized — rewards timing/sizing edge, not riding the +19%/yr drift. **Now the RL
> training signal** (supersedes rev-2 D20).

> **D4.** Three disjoint **ticker** sets — training/validation/test — stratified by
> asset class. Validation drives checkpoint selection/early-stop; test is touched once.
> Generalization to unseen tickers is the goal (PRD). Time-split-within-train (rev 2)
> dropped as primary.

> **D5.** **Capture the whole signal suite** — events *and* continuous values for every
> indicator plot (supersedes rev-2 D17 pruning). Measuring all signals is the point;
> selection/regularization is the model's job; rare signals still get recorded edge.

> **D6.** Supervised LightGBM **retained as baseline + attribution instrument**, not the
> primary learner. It is the most interpretable lens on "which signals give edge" and
> lets the pipeline be validated before RL is tuned.

> **D7 (carried).** Parquet for features (dtypes, columnar, compression).

> **D8.** **Signal-edge report is a first-class deliverable** (D-B), assembled from
> study stats + condition/interaction buckets + GBT SHAP + RL ablation. The project's
> outcome is this knowledge, not only the agent.

> **D9 (carried).** finplot (replay) + pyqtgraph (monitor) on one PyQt5 stack.

> **D10.** Sigmoid sizing for the supervised baseline (`w=sigmoid(scale·(edge−thr))`,
> thr=0, scale train-calibrated). PPO outputs the weight directly.

> **D11.** Build-time `collect` captures OHLCV (real candles + realistic fills); close-
> only is a graceful fallback. Reuses the existing CDP extractor.

> **D12–D16 (carried from rev 2):** train-only normalizers/terciles (D13); cross-TF =
> 4H→1D aggregated within 1 bar (D14); stratified ticker holdout (D15); pooled-OOS MCPT
> + runs test (D16); LightGBM regularization + determinism for the baseline (D18).

> **D17 (new).** Continuous indicator values stored per bar as causal features
> (`{sig}_val`, `{sig}_slope_3`), z-scored on train stats — exploits "everything we can
> get from the indicators," not just binary fires, while staying leak-free.

> **D18 (new).** **RL feature-ablation** (zero/shuffle a feature, re-eval ΔSharpe) is
> the agent-side edge attribution, complementing GBT SHAP — answers "what does the
> trained agent actually trade on."

> **D19 (new).** Reward is the RL signal *and* a monitor/replay readout. (rev-2's
> "eval-only" framing is void now that PPO is primary.)

## 12. Open questions
1. **Does the edge survive at all?** (PRD killer risk.) First experiment once the
   pipeline runs: pooled-OOS MCPT on test tickers + the solo/interaction edge report. A
   null result invalidates the premise and must be reported honestly.
2. **PPO on small data.** ~24 short episodes/pass is little for stable policy gradients;
   risk of degenerate policies (always-flat / always-100%). Mitigations: heavy reward
   shaping (the turnover-penalized differential Sharpe), small policy net, many
   reshuffled passes, entropy regularization, and the supervised baseline as a sanity
   floor. May need a contextual-bandit simplification if PPO won't stabilize.
3. **Episode shape.** Whole-history episode per ticker vs sliding sub-episodes
   (random start/length) for more rollout diversity. Default: whole-history with
   reshuffled ticker order; revisit if rollouts are too correlated.
4. **Action cadence.** Continuous weight every bar vs rebalance-on-signal-change to cut
   turnover/noise. Default continuous; expose a discrete {0,.25,.5,.75,1} mode.
5. **Build order.** Stand up `collect→build→env→supervised baseline→analyze→validate`
   end-to-end first (fast, exercises the whole pipeline + first edge report), then drop
   in PPO — or go straight to PPO. Leaning baseline-first for de-risking; confirm.

## 13. Out of scope
Real brokerage / order routing / paper / live trading; shorts, margin, leverage,
options; per-ticker custom models or tuning; multi-ticker portfolio/allocation (later);
new indicators beyond the Prophets set; intraday/HFT logic (1D is the clock);
web/mobile UI (desktop only); vectorbt bulk scans (later).
