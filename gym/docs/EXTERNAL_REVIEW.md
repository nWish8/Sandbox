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

## Deferred (worth it, not now)

- **microsoft/qlib** — the *rolling retrain* workflow (train → roll window forward → retrain
  → stitched OOS record) is the right upgrade path for promotion ("walk-forward
  promotion" on the roadmap). Adopting qlib itself is not: heavy infrastructure, its own
  data format, overlaps everything we already have. Take the workflow shape, not the
  dependency.
- **Yvictor/TradingGym** — the original inspiration for the env pattern; already absorbed
  in rev-1..4. Nothing left to take that `PortfolioEnv` doesn't do.
- **wilsonfreitas/awesome-quant** — directory, not a library. Flagged for later:
  `vectorbt` (bulk parameter sweeps) when the strategy lab needs mass search.
- **goldmansachs/gs-quant** — the timeseries measure library is nice but mostly duplicates
  `stats.py`; the interesting parts (risk, pricing) need a GS backend. Revisit only if a
  measure we lack comes up.
- **stefan-jansen/ml4t** (rest of it) — meta-labeling and sample-weighting are candidates
  for the projections module *after* it demonstrates any OOS value worth refining.

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
