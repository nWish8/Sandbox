# Signal Gym v2 — Research Log

A single, dated record of what was tested in the **v2 (FinRL multi-asset)** rebuild and what
was found. Newest entry first. This log starts fresh: the rev-1…rev-4 findings (single-ticker
Prophets timing, the `timing_sortino` experiments) are treated as **historical context only**,
not as current truth — the goal changed (see `prd.md`).

**Ground rules for entries**
- State the question, the setup (config + seed + split), the result, and the honest verdict.
- A negative result is a result. Record it as plainly as a positive one.
- No metric counts as a "pass" if it can be satisfied by a degenerate or leaky shortcut.

---

## 2026-06-29 — v2 rebuild begins

**Decisions locked** (from `prd.md`, via Nick): pivot the spine to FinRL's multi-asset
`StockPortfolioEnv` (action = softmax weights over N assets); treat the reward objective as an
**open investigation** (compare ≥4 corrected candidates on held-out data); the TradingView
"intentions" indicator is v2.

**Carried-forward context worth keeping** (the rest of the old `.md` record was discarded):
- The previous default objective, `timing_sortino`, scored the agent vs an *average-exposure
  twin* (`active_t = ret_t − mean_weight·bh_ret_t`). Intent — reward genuine timing, not
  de-risking — was sound. It was retired because the implementation: (a) built the twin from
  `mean_weight` over the **whole** evaluation window (non-causal); (b) annualized with `√252`
  on a **1-hour** bar clock (wrong scale); (c) was self-relative (no external benchmark);
  (d) had a denominator fallback that flattered lucky agents. Empirically it overfit:
  in-sample timing-fitness climbed while held-out timing-fitness went negative.
- Measured earlier: for FinRL single-env DRL the **GPU is slower** than CPU (single CPU-bound
  env loop + tiny net). GPU only pays for batched population work.

**T1 — cleanup done.** Removed the LightGBM baseline footprint (`edge_model.lgbm`,
`edge_report.json`, `lightgbm`/`shap` from requirements) and the 6 stale rev-3 test files that
imported deleted modules (`agent_sup`, `agent_rl`, `analysis`, `train`). Deleted the legacy
research docs (`BRIEF`, `WALKTHROUGH`, `evo_spec`, `spec`, `tasks`). This file is the new record.

*(Subsequent tasks — env foundation, reward registry, investigation harness, regime data,
significance gate — append their findings below as they land.)*

## 2026-06-29 — reward investigation

Universe: 8 tickers · train 2014-01-01..2022-01-01 · trade 2022-01-01..2024-01-01 · PPO 3000 steps · seed 42.

Ranked by **validation** `active_sharpe` (selection never touches test):

```
      reward | val_active_sh test_active_sh | val_   sharpe test_   sharpe | val_  sortino test_  sortino | val_   calmar test_   calmar | val_total_ret test_total_ret
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
      return |        -0.899         -0.219 |        +0.280         +1.503 |        +0.403         +2.279 |        +0.229         +2.230 |        +0.032         +0.149
 diff_sharpe |        -1.096         -2.737 |        +0.277         +1.441 |        +0.400         +2.184 |        +0.226         +2.108 |        +0.032         +0.142
      active |        -1.378         -1.254 |        +0.270         +1.465 |        +0.390         +2.219 |        +0.217         +2.137 |        +0.031         +0.145
      logret |        -1.427         -1.089 |        +0.275         +1.468 |        +0.396         +2.225 |        +0.223         +2.116 |        +0.031         +0.146
  active_dsr |        -1.934         +0.403 |        +0.258         +1.518 |        +0.371         +2.304 |        +0.200         +2.294 |        +0.028         +0.151
```

**Verdict:** val-selected reward 'return' has test active_sharpe -0.219 -> no out-of-sample edge over equal-weight. Honest negative.

**Significance gate** (champion test active returns): gate: Sharpe=-0.219  MCPT p=0.5730  runs p=0.3770 -> not significant

**Champion test performance by market regime** (causal bull/bear/choppy labelling):

```
  regime |        n_bars | active_sharpe |        sharpe |  total_return
------------------------------------------------------------------------
    bull |           127 |        -0.073 |         2.349 |         0.129
    bear |            61 |        -0.450 |         1.418 |         0.043
  choppy |            31 |        -0.192 |        -1.961 |        -0.024
```

Run manifest (reproducibility): `investigation_20260629_221358.json`

## 2026-07-03 — reward investigation

Universe: 8 tickers · train 2014-01-01..2022-01-01 · trade 2022-01-01..2024-01-01 · PPO 20000 steps · seed 42.

Ranked by **validation** `active_sharpe` (selection never touches test):

```
      reward | val_active_sh test_active_sh | val_   sharpe test_   sharpe | val_  sortino test_  sortino | val_   calmar test_   calmar | val_total_ret test_total_ret
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
      active |        +0.377         -2.126 |        +0.304         +1.367 |        +0.439         +2.068 |        +0.263         +1.984 |        +0.037         +0.135
  active_dsr |        -0.563         -0.746 |        +0.268         +1.451 |        +0.388         +2.198 |        +0.210         +2.134 |        +0.030         +0.145
      return |        -1.386         -1.937 |        +0.256         +1.427 |        +0.368         +2.167 |        +0.199         +2.134 |        +0.028         +0.140
      logret |        -1.934         +0.169 |        +0.202         +1.517 |        +0.289         +2.293 |        +0.126         +2.290 |        +0.018         +0.151
 diff_sharpe |        -2.524         -1.666 |        +0.198         +1.426 |        +0.285         +2.153 |        +0.119         +2.109 |        +0.017         +0.140
```

**Verdict:** val-selected reward 'active' has test active_sharpe -2.126 -> no out-of-sample edge over equal-weight. Honest negative.

**Significance gate** (champion test active returns): gate: Sharpe=-2.126  MCPT p=0.9760  runs p=0.2614 -> not significant

**Champion test performance by market regime** (causal bull/bear/choppy labelling):

```
  regime |        n_bars | active_sharpe |        sharpe |  total_return
------------------------------------------------------------------------
    bull |           127 |        -1.406 |         2.264 |         0.125
    bear |            61 |        -3.391 |         1.203 |         0.036
  choppy |            31 |        -1.927 |        -2.091 |        -0.026
```

Run manifest (reproducibility): `investigation_20260703_153609.json`

## 2026-07-04 — reward investigation

Universe: 8 tickers · train 2014-01-01..2022-01-01 · trade 2022-01-01..2024-01-01 · PPO 20000 steps · seed 42.

Ranked by **validation** `active_sharpe` (selection never touches test):

```
      reward | val_active_sh test_active_sh | val_   sharpe test_   sharpe | val_  sortino test_  sortino | val_   calmar test_   calmar | val_total_ret test_total_ret
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  active_dsr |        -0.178         -0.219 |        +0.260         +1.460 |        +0.369         +2.215 |        +0.193         +2.103 |        +0.029         +0.144
      return |        -0.439         -0.571 |        +0.254         +1.355 |        +0.366         +2.060 |        +0.205         +2.061 |        +0.027         +0.139
 diff_sharpe |        -0.510         -0.669 |        +0.226         +1.312 |        +0.324         +1.986 |        +0.163         +1.951 |        +0.022         +0.134
      logret |        -1.074         -3.101 |        +0.174         +0.951 |        +0.250         +1.416 |        +0.093         +1.150 |        +0.013         +0.090
      active |        -1.237         -0.738 |        +0.143         +1.401 |        +0.205         +2.118 |        +0.051         +1.951 |        +0.008         +0.137
```

**Verdict:** val-selected reward 'active_dsr' has test active_sharpe -0.219 -> no out-of-sample edge over equal-weight. Honest negative.

**Significance gate** (champion test active returns): gate: Sharpe=-0.219  MCPT p=0.5824  runs p=0.9659 -> not significant

**Champion test performance by market regime** (causal bull/bear/choppy labelling):

```
  regime |        n_bars | active_sharpe |        sharpe |  total_return
------------------------------------------------------------------------
    bull |           127 |        -1.438 |         2.047 |         0.109
    bear |            61 |         0.547 |         1.539 |         0.046
  choppy |            31 |         2.679 |        -1.039 |        -0.014
```

Run manifest (reproducibility): `investigation_20260704_002118.json`

## 2026-07-04 — walk-forward (ppo/active_dsr, 4 folds × 20000 steps)

Universe: 8 tickers · 2014-01-01..2024-01-01 · anchored expanding window · seed 42.

```
fold test window              active_sh   sharpe    total    maxDD
------------------------------------------------------------------
   0 2019-02-13..2020-05-01      -1.347   +0.509   +0.137   -0.303
   1 2020-05-04..2021-07-21      -1.308   +2.187   +0.530   -0.092
   2 2021-07-22..2022-10-07      -0.498   +0.024   -0.014   -0.161
   3 2022-10-10..2023-12-28      -0.676   +1.964   +0.363   -0.079

stitched OOS (1224 bars): active_sharpe=-0.804  sharpe=+0.975  total=+1.338  maxDD=-0.303
stitched gate: Sharpe=-0.804  MCPT p=0.9590  block p=0.9606  runs p=0.5322 -> not significant
fold dispersion: 0/4 folds positive, spread [-1.347, -0.498]
```
