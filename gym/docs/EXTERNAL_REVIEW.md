# External repo review — what Vision adopted, deferred, and rejected

*Reviewed 2026-07-04 against the project's locked constraints: long-only spot, daily
bars, price-only features, research-only (no execution). Every "adopted" row landed in
code with tests the same day.*

## Adopted

| Source | Insight | Where it landed |
|---|---|---|
| **stefan-jansen/machine-learning-for-trading** (López de Prado's purged CV, via the ML4T reference) | Walk-forward splits with overlapping-label leakage need a **purged embargo**: a row at date *t* carries an h-bar forward target, so training rows within *h* bars of the cut share their target window with the test set and flatter the metric. This was a live bug in our `projections.py` coverage check. | `projections.walkforward_coverage` now purges the last `horizon` training dates before the cut and reports `train_end`/`test_start`; test asserts the gap ≥ h. |
| **neurotrader888/mcpt** | MCPT by **permuting the market, not the returns**: decompose bars, shuffle with structure preserved (marginal distributions, cross-sectional correlation) and serial order destroyed, re-run the strategy per permutation. Tests whether the edge needs real temporal structure. | `rule_policies.permute_market` (shared row permutation of panel log-returns, first bar anchored) + `market_permutation_pvalue`, run against rule baselines. Full data-permutation for RL agents is rejected as infeasible (retraining per permutation); they keep the return-level gate. |
| **neurotrader888/mcpt** (return-level analogue) | Per-bar sign-flip MCPT builds an unrealistically easy null when returns are serially dependent — exactly the case our runs test flags. **Block sign-flip** (one Rademacher sign per block) keeps within-block dependence in the null. | `signif.mcpt_sharpe(block_len=...)` + `default_block_len` (~n^⅓, min 5); `gate()` now runs both and `edge_significant` requires **both** p < α. Test proves the block null is stricter on streaky series. |
| **je-suis-tm/quant-trading** | RL agents need dumb-rule company: classic strategies as baselines under **identical accounting**, or "beat equal-weight" claims are unanchored. | `rule_policies.momentum_topk` (cross-sectional rotation) + `inverse_vol` (risk-parity-lite), causal and simplex-valid by test; `promote()` now appends an agent vs equal-weight vs rules table under the same cost model. |
| **mementum/backtrader** | Execution realism: decisions after the close fill at the **next open**, so overnight gaps are earned by the *old* position. Close-to-close accounting silently awards gaps to the new weights. | `strategy_eval.replay_weights_ohlc` (gap earned by old weights, intraday by new, compounded); `promote --fills open`. Gapless-market equivalence to close-fill is tested to 1e-10. |
| **kernc/backtesting.py** | Trade-quality stats beyond Sharpe: profit factor, concentration, extremes. (Our `stats.py` was already modelled on this library.) | `evaluate()` report adds `profit_factor`, `hhi_mean` (concentration), `best_bar`/`worst_bar`; rendered in `format_report`. |

## Round 2 adoptions (2026-07-04, second pass on the deferred set)

| Source | Insight | Where it landed |
|---|---|---|
| **microsoft/qlib** | The *Rolling Retraining* workflow: one train/test split is a single draw; retrain on an anchored expanding window per segment, roll the model over its own segment only, stitch the segments into one continuous OOS record, and look at fold dispersion. | `walkforward.py` (`make_folds` + `walkforward_run` + fold-dispersion report), CLI `gym.run walkforward`, gated by the same block-robust gate, logged to RESEARCH_LOG. The dependency itself stays out (heavy infra, own data format). |
| **microsoft/qlib / ML4T** (evaluation practice) | Report the **rank IC** (Spearman of forecast vs realised) — a band-coverage number can look fine while the forecast has zero ordering power. | `projections.walkforward_coverage` now reports OOS `rank_ic`, rendered with a blunt strength label ("no real ranking power" below 0.03). |
| **goldmansachs/gs-quant** (measure style) | Drawdown *persistence* matters separately from depth: longest underwater spell and share of time underwater. | `strategy_eval.evaluate` → `max_dd_duration`, `pct_time_underwater`, in `format_report`. |
| **cantaro86/Financial-Models-Numerical-Methods** | Returns are not Gaussian; report the tail, not just σ: VaR/CVaR (95), skew, excess kurtosis. | `strategy_eval.evaluate` → `var_95`, `cvar_95`, `skew`, `excess_kurtosis`, hand-computed tests. |
| **s0ap/gs-quantitative-strategies-research-notes** | The canonical momentum construction is **12-1**: skip the most recent month to sidestep short-term reversal. | `momentum_topk(skip=...)` + `momentum_12_1` in `RULES` (auto-appears in promotion baselines; rules whose warm-up exceeds the window are skipped with a note). Test proves the skip window dodges a crafted late crash. |

## Still deferred

- **Yvictor/TradingGym** — the original inspiration for the env pattern; already absorbed
  in rev-1..4. Nothing left to take that `PortfolioEnv` doesn't do.
- **wilsonfreitas/awesome-quant** — directory, not a library. Flagged for later:
  `vectorbt` (bulk parameter sweeps) when the strategy lab needs mass search;
  `exchange_calendars` (already installed via FinRL) if trading-day-aware cache coverage
  ever matters.
- **goldmansachs/gs-quant** (the library itself) — the interesting parts (risk, pricing)
  need a GS backend; the measures we wanted are now implemented locally.
- **stefan-jansen/ml4t** (rest of it) — meta-labeling and sample-uniqueness weighting are
  candidates for the projections module *after* it demonstrates OOS ranking power worth
  refining (the new rank IC line is the tripwire: no IC, no meta-model).
- **qlib Alpha158 factor zoo** — a curated causal-factor library; worth mining for env/
  screener features when the regime-conditional investigation motivates a feature push.

## Rejected (honestly out of scope)

- **jxm35/LimitOrderBook-MatchingEngine** — microstructure/LOB simulation needs intraday
  order-book data we don't have and won't ingest (price-only, daily, research-only). Our
  slippage model stays at the bps-on-turnover level appropriate to daily bars.
- **cantaro86/Financial-Models-Numerical-Methods** — excellent numerical notebooks
  (jump-diffusion, Kalman, option pricing) aimed at derivatives; the project is spot-only
  by constraint. Nothing transplants without violating scope.
- **s0ap/gs-quantitative-strategies-research-notes** — strategy *ideas* (momentum/carry
  families); the momentum family is now represented by the baseline rules. The notes are
  reading material, not code to adopt.
- **neurotrader888/TradeDependenceRunsTest** — already implemented: `signif.runs_test`
  has carried the Wald–Wolfowitz test since v2 (this repo was its original reference).

## Net effect on the honesty machinery

The gate got strictly harder to fool (block MCPT), the projections coverage metric lost a
leak that flattered it, rule baselines put every agent's active_sharpe in context, and
open fills stop close-to-close accounting from crediting overnight gaps the strategy
couldn't have captured. All four changes push in the same direction as the project's
founding rule: never let a metric flatter itself.
