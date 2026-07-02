# AUDIT.md — Repository audit for the "Vision" rebuild

*Phase 0 deliverable. Audited 2026-07-02, working tree at commit `005f75e` + large uncommitted
v2 refactor.*

## 1. What this repository actually is

This is **Signal Gym v2** — a local research framework that trains SB3 PPO agents to allocate a
long-only, multi-asset spot portfolio from OHLCV-derived features, and judges them honestly
against an equal-weight benchmark with a significance gate. It is **mid-refactor and
uncommitted**: the last commit is the population-evolution + 3-pane replay work; since then the
`finrl_lab/` scripts were folded into `gym/`, the single-ticker rev-3/4 docs and tests were
partially deleted, and the v2 reward-investigation core was added (untracked:
`portfolio.py`, `rewards.py`, `signif.py`, `regime.py`, `investigate.py`, `RESEARCH_LOG.md` +
their tests).

The prompt's assumption that this repo contains **Deep Trading Advisor** code (Keras
MLP/CNN/LSTM, Zipline, Dash/Plotly) is **wrong for this repo** — a full-text sweep finds no
Keras, TensorFlow, Zipline, Dash, or LSTM code anywhere. There is nothing of that lineage to
mine here; if those models exist they live in a different repository.

## 2. Inventory — three strata

### A. v2 core (current, tested, keep)

| Module | Role | Test status |
|---|---|---|
| `gym/portfolio.py` | Multi-asset env on FinRL `StockPortfolioEnv`: softmax weights, turnover cost, pluggable reward, causal covariance | pass |
| `gym/rewards.py` | Per-step reward registry (return, logret, diff_sharpe, active, active_dsr) | pass |
| `gym/stats.py` | Sharpe/Sortino/Calmar/active_sharpe vs external equal-weight benchmark | pass |
| `gym/signif.py` | OOS significance gate (MC permutation + runs test) | pass |
| `gym/regime.py` | Causal bull/bear/choppy labelling + per-regime eval | pass |
| `gym/investigate.py` | Head-to-head reward investigation: train per reward, val-select, test once, gate, log | pass |
| `gym/pipeline.py`, `gym/finrl_patch.py`, `gym/config.py` | FinRL data prep + DRL wrapper + config | pass |
| `gym/run.py` (investigate subcommand), `gym/control_panel.py` | CLI + PyQt panel | pass / manual |

### B. Population evolution + replay (kept deliberately in the refactor — the seed of Vision module 4)

- `gym/evo_portfolio.py` — GPU-batched population evolution over a multi-stock portfolio.
- `gym/evo_replay_panel.py` — **3-pane, bar-by-bar stepped replay** of a recorded evolution run
  (`evo_portfolio_record.npz`), rendered with pyqtgraph. This already implements the core of the
  requested "bar-by-bar RL playback visualizer" including per-generation recording-to-disk so
  replay never requires retraining.

### C. Legacy rev-3/4 single-ticker layer (broken by the refactor — retire)

`env.py`, `env_finrl.py`, `features.py` (1h bars), `collect.py`, `normalize.py`,
`backtest.py`, `baselines.py`, `valid.py`, `evo.py`, `evo_monitor.py`, `evo_replay.py`,
`population.py`, plus `run.py`'s collect/train/replay/backtest subcommands.

**Test evidence (full suite, project `.venv`, py3.11):** `127 tests → 72 passed, 4 failed,
51 errors`. Every error/failure is in this legacy stratum: `ImportError`/`TypeError` against
modules whose signatures the v2 refactor changed, and `test_foundation` asserting rev-3 `Config`
fields (`target_horizon`, `SIGNAL_STUDY_RESULTS`) that no longer exist. All v2-core tests pass.

## 3. Correctness assessment (look-ahead and metric hygiene)

The v2 core is in good shape — the failure modes the Vision prompt warns about were already
found and fixed here, with tests:

- **No look-ahead, verified**: covariance windows use bars `[t-lookback, t-1]` only; a decision
  at bar *t* earns only the *t→t+1* move; regime labels use trailing windows; dedicated causality
  tests pass (`test_env_no_lookahead` for v2 paths, perturbation tests).
- **No self-flattering metrics**: benchmark is external (equal-weight, same bars); model
  selection touches validation only; test slice scored once; significance gate on top. The
  research log records an honest negative result — the machinery does not inflate.
- **Known historical bug, fixed**: `√252`-annualization on 1-hour bars (rev-4) — v2 moved to
  daily bars; `stats.py` documents the fix.
- **Turnover cost charged** (`cost_pct · Σ|Δw|`), which stock FinRL omits — churn is not free.

Warts worth fixing during the rebuild (not correctness bugs):

1. **Flat imports** (`from pipeline import …`) require `gym/` itself on `sys.path`
   (see `tests/conftest.py`); should become proper package-relative imports.
2. `add_cov_features` is an O(bars × lookback) pandas pivot loop — fine for 8 tickers/daily,
   too slow for a bigger universe; vectorize or cache to Parquet.
3. FinRL **0.3.7** is kept alive by a monkeypatch shim (`finrl_patch.py`) because its dependency
   chain no longer pip-resolves — a maintenance liability and the strongest argument for the
   FinRL-X migration.
4. No slippage/latency model — only proportional turnover cost.
5. Data caching is ad-hoc; no SQLite/Parquet store.

## 4. FinRL-X — verified real, and a natural fit

Verified 2026-07-02: **FinRL-X exists** (arXiv 2603.21330, PAKDD 2026 DMO-FinTech workshop;
official implementation in `AI4Finance-Foundation/FinRL-Trading`; v1.0.0 released March 2026;
`pip install finrl-trading`; Python 3.11+; Apache-2.0). Key facts that matter for us:

- **Weight-centric contract**: every stage (selection → allocation → timing → risk overlay)
  emits a portfolio weight vector `w_t`. Our v2 env is *already* weight-centric (softmax onto
  the simplex) — adopting FinRL-X is an interface alignment, not a rewrite.
- Layers we can adopt: multi-source **data fetcher with SQLite caching** (replaces our yfinance
  ad-hoc path + the 0.3.7 monkeypatch), **`bt`-powered backtester** with transaction costs and
  multi-benchmark comparison (Vision module 5), optional Alpaca execution (out of scope — see §6).
- It does **not** mandate SB3; DRL allocators plug in as weight-generating strategies. We keep
  SB3 PPO/SAC as the training backend and expose trained policies to FinRL-X as strategies.
- Caveat: v1.0.0 is ~4 months old. Treat it as a **library behind adapters** (data + backtest),
  not as the process spine, so a breaking release cannot strand the lab.

## 5. UI decision (required by the prompt): PyQt5 + pyqtgraph — not Textual, not Electron/React

- The bar-by-bar playback requirement is the binding constraint. **pyqtgraph is a
  GPU-accelerated scene-graph renderer** — it is the "canvas-based, not matplotlib-per-frame"
  option, and the repo already proves it works here (3-pane replay panel, live evo monitor).
- **Textual** (TUI) cannot render smooth candlestick playback; it would need a side web view,
  splitting the app in two.
- **FastAPI + React/Canvas** is the most graphically capable option but rebuilds 100% of the
  existing UI for zero research gain, adds a server/browser boundary on a single-user local
  tool, and drops the working panel.
- Verdict: keep the desktop Qt shell, restyle it as the Vision terminal (dark, high-contrast,
  monospace, green/amber accents — Qt stylesheets make this cheap), and extend the existing
  replay panel with the generation scrubber. Revisit web only if Vision later needs remote
  access.

## 6. Scope caution carried over from the project's own constraints

The repo's locked design constraints: **long-only spot, research only, no broker execution, no
real capital**. Vision's Portfolio Monitor should therefore be **read-only** (prices, P&L,
exposure, correlation — no order routing), and FinRL-X's execution layer stays unused. Live
trading would be a separate, explicit decision — not something a rebuild silently enables.

## 7. Salvage verdict

| Disposition | What |
|---|---|
| **Keep as-is** | v2 core (§2A), its tests, `RESEARCH_LOG.md`, run manifests |
| **Keep + extend** | `evo_portfolio.py`, `evo_replay_panel.py` (→ Vision playback + scrubber), `control_panel.py` (→ Vision shell) |
| **Replace via FinRL-X adapters** | yfinance download path + `finrl_patch.py` (→ data fetcher w/ SQLite cache); ad-hoc eval plots (→ `bt` backtest module) |
| **Discard** | Legacy stratum §2C and its 55 broken tests; stale `run.py` subcommands that import it |
| **Does not exist here** | Deep Trading Advisor models — nothing to port from this repo |

## 8. Recommended path: evolve in place, not scratch rebuild

A from-scratch rebuild would discard a passing, causality-tested research core to rebuild the
same thing. The defensible move:

0. **M0 — Commit the WIP first.** The entire v2 core is currently uncommitted; any
   restructuring before a commit risks unrecoverable loss. Then delete the legacy stratum in a
   follow-up commit so the suite runs green.
1. **M1 — Data + screener**: FinRL-X data adapter (SQLite/Parquet cache, provider-swappable),
   indicator screener; drop `finrl_patch.py`. *(~2–3 sessions)*
2. **M2 — RL lab**: mostly exists; add SAC alongside PPO, structured per-generation run logs
   (JSON/NPZ manifest schema already started). *(~1–2 sessions)*
3. **M3 — Playback visualizer**: extend the 3-pane replay with a generation scrubber +
   holdings/action/reward/equity overlays for PPO runs (already done for evolution runs).
   *(~2 sessions)*
4. **M4 — Backtester + strategy promotion**: FinRL-X `bt` engine behind an adapter; add
   slippage/fee/latency model + unit tests for fills; promotion flow from investigation
   champion → formal backtest. *(~2–3 sessions)*
5. **M5 — Vision shell + theme**: unify panels under one dark Qt terminal shell; read-only
   portfolio monitor with correlation heatmap. *(~2 sessions)*

**Decision needed before Phase 1**: (a) approve evolve-in-place over scratch rebuild, and
(b) approve committing the current working tree as M0.
