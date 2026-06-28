# PRD: Signal Gym

**Status:** Draft
**Owner:** Nick
**Last updated:** 2026-06-27

## 1. Summary
Signal Gym is a single-ticker trading environment and ML framework built on top of the
Prophets signal study (`../signal_study/`). One **general** model — trained across all 34
tickers in the study universe — learns a *transferable* edge-case decision from the
Prophets indicator signals plus price action, and uses it to manage a **long-only spot**
position on any individual ticker's bar-by-bar stream. The system steps through a ticker's
history as a gym, lets a supervised edge-model (v1) — and later an RL agent (v2) — decide
how much to hold, and **replays those decisions live on a finplot chart** with trades and
an equity-vs-buy-and-hold curve. Every result is **statistically validated** (Monte Carlo
permutation test, runs test, walk-forward). Built now because the study proved the Prophets
signals carry real but small, rare, context-dependent timing edge — the next step is to
learn to exploit it systematically and watch an agent do it, for Nick (solo trader/dev).

## 2. Problem
The Prophets signals carry a real edge, but the study showed it is **small, rare, and
context-dependent**: solo signals are sub-1% once trend is removed; the value lives in
interactions (confluence, higher-timeframe agreement, regime filters like "buy a divergence
only inside a continuation regime"); and it is concentrated in ~3–4% of bars. No hand-written
rule captures these interactions across 34 tickers and ~30 signals, and naive backtests
**actively mislead** — on a universe that trended +19%/yr, buy-and-hold beats almost every
rule on raw return, so raw P&L is the wrong yardstick and easy to fool yourself with. Nick
needs to (a) learn a general edge-decision from the data rather than hand-tune per ticker,
(b) prove it is real signal and not data-mined noise, and (c) see the agent trade bar-by-bar
so the behavior is legible and trustworthy.

## 3. Users
Primary (and only) user: **Nick** — a solo trader/developer who trades long-only spot,
authored the Prophets signal study, and is building this to discover, validate, and trust
signal-driven timing/sizing edges. Technical; runs everything locally; values interpretability
and statistical rigor over black-box performance claims.

## 4. Goals
1. A single **general model**, trained across all tickers (no per-ticker models), that
   achieves **positive risk-adjusted excess return over buy-and-hold** out-of-sample.
2. **Generalization**: positive excess on **held-out tickers not seen in training**.
3. **Statistical confidence** the edge is real — passes Monte Carlo permutation test
   (p < 0.05) and walk-forward out-of-sample, with trade-dependence (runs test) reported.
4. A **live finplot replay** where Nick watches the agent buy/sell with active signals,
   position, and equity-vs-B&H visible and steppable.
5. A **reusable gym environment** that both the v1 supervised policy and the v2 RL agent
   plug into unchanged.

## 5. Non-goals
1. **No real-money or brokerage execution** — simulation/replay only.
2. **No shorting, leverage, or options** — long-only spot exclusively.
3. **No per-ticker bespoke models** — the whole point is one general, transferable model.
4. **No multi-ticker portfolio optimization** in v1 — one ticker per agent run.
5. **No new signal generation** — the system consumes the existing Prophets signals; it does
   not invent indicators.
6. **No intraday/HFT focus** — 1D is the primary timeframe; faster TFs are features, not the
   trading clock.

## 6. User stories
1. As Nick, I want to convert the collected signal/price data into per-bar feature matrices, so that an ML model can train on confluence and regime context, not just raw fires. [MVP]
2. As Nick, I want to train one general edge-model across all tickers, so that I get a transferable decision rather than 34 overfit per-ticker models. [MVP]
3. As Nick, I want the model evaluated on held-out tickers, so that I know the edge generalizes beyond what it trained on. [MVP]
4. As Nick, I want to step a single ticker through a gym environment where the agent sets a long-only position each bar, so that I can simulate trading the signals. [MVP]
5. As Nick, I want the agent rewarded for risk-adjusted excess over buy-and-hold, so that it learns timing/sizing rather than just holding. [MVP]
6. As Nick, I want a stepped finplot chart showing candles, active signals, the agent's position, its trades, and an equity-vs-B&H curve, so that I can watch and understand its decisions live. [MVP]
7. As Nick, I want a Monte Carlo permutation test and runs test on the agent's results, so that I can trust the edge is not data-mined chance. [MVP]
8. As Nick, I want classic baselines and my logged rules (A3, A5b, Prophet Confluence Buy) backtested alongside, so that the agent has a benchmark to beat. [MVP]
9. As Nick, I want to drop a PPO reinforcement-learning agent into the same environment, so that I can compare a learned policy to the supervised model. [v2+]
10. As Nick, I want vectorbt-driven bulk scans of feature/parameter combinations, so that I can search edge cases fast and feed many runs into the permutation test. [v2+]
11. As Nick, I want to run an agent across multiple tickers as a portfolio, so that I can allocate capital across the universe. [v2+]

## 7. Functional requirements
1. Ingest `signal_study/results/*.json` (per-ticker price series + signal fire-times) for all tickers in the universe. [MVP]
2. Produce per-bar, time-aligned **feature matrices** per ticker: price features (returns, volatility, range position, drawdown), per-signal features (fired-this-bar, bars-since-fired, fired-within-K), and confluence/regime features (active bull/bear signal counts, higher-TF agreement flags, Continuation-active, cloud state). [MVP]
3. Encode only **generalizable context** (sector/asset-class, volatility regime); **must not** include raw ticker identity as a feature. [MVP]
4. Compute the supervised **target** as the baseline-relative forward edge at a configurable horizon H, with **no lookahead leakage**. [MVP]
5. Provide **walk-forward** train/test splits and a **held-out-ticker** split for generalization testing. [MVP]
6. Expose a **Gymnasium-compatible environment**: `reset()`/`step()` over one ticker; observation = lookback window of features; action = long-only target weight in [0,1]; fills at **next bar open**; transaction/turnover costs applied. [MVP]
7. Compute a per-step **reward = risk-adjusted excess over buy-and-hold** (differential-Sharpe of excess returns), penalized for turnover. [MVP]
8. Support a **continuous** target-weight action space, with a **discrete** {0,25,50,75,100%} mode as a configurable option. [MVP]
9. Train a **supervised gradient-boosted edge-model** on the pooled features and expose **feature importances** (which signals/contexts drive the decision). [MVP]
10. Convert the model's edge prediction into a **sizing policy** that produces the env action. [MVP]
11. Provide a **custom bar-stepped backtest engine** (position, equity, next-bar fills, commission/slippage) that runs a policy over a ticker and reports trades, equity curve, exposure, and excess-over-B&H metrics. [MVP]
12. Run a **buy-and-hold benchmark** for every ticker and report strategy metrics relative to it. [MVP]
13. Run a **Monte Carlo permutation test** producing a p-value that the agent's excess is not chance. [MVP]
14. Run a **runs test** for serial dependence in the agent's trade returns. [MVP]
15. Provide a **finplot live replay**: bar-stepped chart with candles, active-signal markers, agent position shading, trade entry/exit arrows, an equity-vs-B&H sub-pane, and a live PnL/reward readout, at controllable step speed. [MVP]
16. Backtest **classic baselines and the logged study rules** (A3, A5b, Prophet Confluence Buy) for comparison. [MVP]
17. Make runs **reproducible** (seeded, deterministic given fixed data + config). [MVP]
18. Provide a **PPO reinforcement-learning agent** (stable-baselines3) that trains and runs on the identical environment. [v2+]
19. Provide **vectorbt-based bulk scans** of feature/parameter/edge-case combinations, feeding the permutation-test harness. [v2+]
20. Support **multi-ticker portfolio** runs allocating capital across the universe. [v2+]

## 8. Non-functional requirements
- **No-lookahead correctness:** the single most important property — all features and fills must be strictly causal; a leakage bug invalidates every result. Enforced by design and tested.
- **Reproducibility:** seeded runs; identical config + data ⇒ identical results and replay.
- **Statistical rigor:** validation (MCPT, runs test, walk-forward, held-out tickers) is a first-class, non-optional part of any reported result — never an afterthought.
- **Performance:** finplot replay steps smoothly at interactive speed (target ≥ 10 bars/sec, adjustable); bulk scans (v2) leverage vectorbt for vectorized speed.
- **Offline operation:** runs entirely from the collected data; no live network dependency to train/backtest/replay. Data is refreshable via the existing TradingView MCP but not required at runtime.
- **Interpretability:** the v1 model must expose feature importances; decisions should be inspectable, not opaque.

## 9. Success metrics
- **Out-of-sample excess vs B&H:** agent's risk-adjusted excess return over buy-and-hold > 0 on the walk-forward test period (quantify: mean excess %/yr in-market and Sharpe-of-excess > 0).
- **Generalization gap:** excess on **held-out tickers** > 0, and within a defined tolerance of in-sample excess (e.g. held-out excess ≥ 50% of in-sample).
- **Statistical significance:** Monte Carlo permutation test **p < 0.05** for the agent's excess metric.
- **Trade dependence:** runs-test result reported (qualitative pass: no evidence the edge is an artifact of return autocorrelation).
- **Benchmark beat-rate:** agent beats buy-and-hold (risk-adjusted) on ≥ 60% of tickers out-of-sample.
- **Live replay usability (qualitative):** Nick can run a ticker replay end-to-end and read the agent's position, trades, and equity-vs-B&H at a glance.

## 10. Assumptions
- The Prophets signals carry a **transferable** edge (one that holds across tickers), not merely per-ticker idiosyncrasies — if false, a single general model cannot work.
- **34 tickers × ~475 daily bars** (plus 4H/1H feature context) is enough data to train a general model without catastrophic overfitting.
- The historical edge **persists** well enough that an out-of-sample/held-out test is meaningful (no total regime break inside the data window).
- The study's **baseline-relative edge** definition is the right target for supervised learning.
- **finplot** is sufficient for a smooth, legible stepped replay (no need for a web UI in v1).
- The existing `signal_study/backtest.py` patterns are a sound seed for the stepped engine.

## 11. Risks & open questions
- **Killer risk:** the edge **does not generalize / fails the permutation test** — i.e. the ~1,700 mined combos were largely data-mining artifacts and a general model finds nothing real on held-out tickers. If MCPT says "no signal," the core premise is invalidated; the project must be honest about this outcome. This is the first thing to test.
- **Sample-size risk:** daily data is small (~475 bars/ticker); a flexible model may overfit even when pooled. Mitigate with strong regularization, walk-forward, held-out tickers.
- **Reward-shaping risk:** a poorly shaped excess-over-B&H reward could collapse to a degenerate policy (always flat, or always 100%); needs careful turnover penalty and validation.
- **Leakage risk:** subtle lookahead in feature construction or fills would manufacture fake edge; requires explicit causal tests.
- **Regime risk:** the data window is a strong bull; an edge learned here may not hold in flat/bear regimes (though the study suggests timing signals would shine *more* there).
- **Open questions (for the spec):** exact feature set and encodings; precise reward formula (differential-Sharpe parameters, turnover cost); how to align/encode multi-timeframe signals onto the 1D clock; train/test/held-out split sizes given small data; how generalization context (sector, vol regime) is bucketed; how the sizing policy maps edge prediction → weight.

## 12. Out of scope
- Real brokerage connectivity, order routing, paper or live trading.
- Short positions, margin, leverage, derivatives, options.
- Per-ticker custom models or hand-tuned per-ticker parameters.
- Multi-ticker portfolio construction and capital allocation (v1).
- Creation of new indicators/signals beyond the existing Prophets set.
- Intraday/high-frequency trading logic.
- A web or mobile front-end (finplot desktop only in v1).
- Reinforcement learning and vectorbt bulk search (deferred to v2).
