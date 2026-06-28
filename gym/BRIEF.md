# Signal Gym — project brief

Source material for the PRD. Describes the system we want to build on top of the
Prophets signal study (`../signal_study/`). Long-only spot throughout (Nick's
constraint).

## One-line vision
A single-ticker **trading gym**: load a ticker's price + Prophets signals, step the
bars forward, and an **agent manages a long-only spot portfolio** off the available
signals — buying/selling as it goes — while we **watch the chart and the agent's
decisions play out live**. ML models learn from signals + price action to find edge
cases, and every result is **statistically validated** so we don't fool ourselves.

## Why this shape (grounded in the study's findings)
- **Buy-and-hold is a brutal benchmark** (study universe ran +19%/yr). Rewarding raw
  P&L would teach the agent to just hold. → objective is **excess over buy-and-hold,
  risk-adjusted**.
- **The edge is rare and timing-based** (best methods ~3–4% time in market). → this is
  a **timing/sizing problem**: *when* and *how much*, not *always in*.
- **Edge lives in context** (confluence, higher-TF agreement, regime filters), not any
  single signal. → features must encode confluence/regime, not just raw fires.
- We mined ~1,700 combos. → **statistical validation (MCPT, runs test, walk-forward)
  is first-class, not an afterthought.**

## Locked decisions (2026-06-27)
| Decision | Choice |
|---|---|
| Agent paradigm | **Hybrid, staged** — v1 supervised ML edge-model + sizing policy; Phase 2 RL (PPO) in the same env |
| Backtest/execution engine | **Custom bar-stepped engine** (doubles as the gym + live-viz stepper) **+ vectorbt** for bulk edge-case search & Monte Carlo |
| Live visualization | **finplot** (desktop pyqtgraph), bar-stepped with trades + equity overlaid |
| Agent objective (reward) | **Excess over buy-and-hold, risk-adjusted** (differential-Sharpe of excess returns, minus turnover cost) |

## Architecture (layers)
1. **Data / feature layer** — turn `signal_study/results/*.json` (sparse signal
   fire-times + price series) into per-bar aligned **feature matrices**:
   - price features (returns, volatility, range position, drawdown),
   - per-signal features (fired-this-bar, bars-since-fired, fired-within-K),
   - confluence/regime features (count of bull/bear signals active, higher-TF
     agreement flags, Continuation-active, cloud state),
   - supervised target = baseline-relative **forward edge** at horizon H.
   - Output parquet per ticker + walk-forward train/test splits.
2. **Gym environment** — Gymnasium `reset()/step()`, one ticker per env:
   - obs = lookback window of features at current bar,
   - action = long-only target weight in [0,1] (continuous; discrete mode optional),
   - fills at **next bar open** (no lookahead), transaction costs applied,
   - reward = per-step risk-adjusted **excess over buy-and-hold**, turnover-penalized,
   - same stepper feeds the live viz.
3. **Agent** — (Phase 1) supervised edge-model (gradient-boosted trees: predict
   forward edge → sizing policy; interpretable, gives signal importances). (Phase 2)
   RL agent (PPO via stable-baselines3) on the identical env. Both benchmarked vs B&H.
4. **Execution/backtest** — custom stepped engine (reuses `signal_study/backtest.py`
   patterns: position, equity, next-bar fills, commission/slippage); **vectorbt** for
   fast bulk parameter/edge-case scans and generating runs for MCPT.
5. **Validation** — walk-forward out-of-sample; **Monte Carlo Permutation Test**
   (mcpt) for p-value that edge isn't chance; **runs test** for trade-return
   dependence; buy-and-hold benchmark everywhere; multiple-testing awareness.
6. **Live visualization (finplot)** — bar-stepped chart: candles, active-signal
   markers, agent position shading, trade arrows, equity-vs-B&H sub-pane, live
   PnL/reward readout. Replays a trained agent's decisions at controllable speed.
7. **Strategy baselines** — classic strategies + our logged rules (A3, A5b, Prophet
   Confluence Buy) as benchmarks the agent must beat (inspiration: je-suis-tm/quant-trading).

## Reference repos (inspiration, by layer)
- **TradingGym** (Yvictor) → gym env interface pattern.
- **backtrader** (+docs) → execution realism / analyzer ideas (engine is custom, not adopted).
- **vectorbt** → fast vectorized bulk backtests for edge-case search.
- **mcpt** (neurotrader888) → Monte Carlo permutation testing.
- **TradeDependenceRunsTest** (neurotrader888) → runs test for trade dependence.
- **je-suis-tm/quant-trading** → classic strategy baselines.

## Locked defaults
- **One GENERAL model trained across ALL tickers**, learning a *transferable* edge-case
  decision; applied per-ticker at runtime (single-ticker env). This is the goal, not
  per-ticker models — pooling was the study's key to honest edge.
  - **No raw ticker identity as a feature** (that invites memorizing per-ticker quirks
    and fights generalization). Only *generalizable* context allowed: sector/asset-class,
    volatility regime, etc. Held-out tickers used to test generalization.
- **Primary timeframe = 1D** (where the study's edge lives); multi-TF signals as features.
- Configurable lookback window W and horizon H (defaults ~ W=30 bars, H=6).
- Action: continuous target weight [0,1]; discrete {0,25,50,75,100%} as an option.

## Phasing
- **v1 (MVP):** feature layer → gym env → supervised edge-model agent → custom stepped
  backtest vs B&H → finplot live replay on one ticker → MCPT + runs-test validation.
- **v2:** RL (PPO) agent in the same env; vectorbt bulk edge-case search.
- **v3:** multi-ticker portfolio; paper/live considerations.

## Non-goals (v1)
- No live brokerage/real-money execution.
- No shorting/leverage/options (long-only spot).
- No multi-ticker portfolio optimization (single ticker per agent run).
