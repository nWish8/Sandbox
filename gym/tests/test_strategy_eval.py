"""strategy_eval.py — the formal backtester must agree with PortfolioEnv to float
precision, cost/latency math must be exact on hand-computed cases, and the optional
bt engine cross-check must agree with our accounting. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio import PortfolioEnv, add_cov_features
from strategy_eval import (CostModel, asset_returns_from_history, bt_crosscheck,
                           evaluate, format_report, replay_weights)

TICS = ["AAA", "BBB", "CCC"]
TECH = ["macd", "rsi_30"]


def _cov_df(n_bars=30, seed=3, lookback=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_bars, freq="D")
    rows = []
    for tic in TICS:
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, n_bars))
        for d, p in zip(dates, price):
            rows.append({"date": d, "tic": tic, "close": p,
                         "macd": rng.normal(), "rsi_30": rng.uniform(20, 80)})
    return add_cov_features(pd.DataFrame(rows), lookback=lookback)


def test_replay_matches_portfolio_env_exactly():
    """Same weights, same costs ⇒ the vectorized backtester reproduces the env's equity.
    The env and the backtester cross-validate each other's accounting."""
    cost_pct = 0.0015
    df = _cov_df()
    env = PortfolioEnv(df, stock_dim=len(TICS), tech_indicator_list=TECH,
                       cost_pct=cost_pct, lookback=3)
    env.reset()
    rng = np.random.default_rng(7)
    done = False
    while not done:
        _, _, done, _, _ = env.step(rng.normal(0, 2, len(TICS)))   # wandering weights
    hist = env.portfolio_history()

    weights = hist[[f"w_{t}" for t in TICS]].to_numpy()
    asset_rets = asset_returns_from_history(df, sorted(TICS))
    res = replay_weights(weights, asset_rets, CostModel(cost_pct=cost_pct, slippage_bps=0.0))

    env_equity = (hist["value"] / hist["value"].iloc[0]).to_numpy()
    np.testing.assert_allclose(res["equity"].to_numpy(), env_equity, rtol=1e-10)
    np.testing.assert_allclose(res["turnover"].to_numpy()[1:],
                               hist["turnover"].to_numpy()[1:], rtol=1e-10)


def test_delay_uses_stale_weights_hand_computed():
    weights = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    rets = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.2]])
    free = CostModel(cost_pct=0.0, slippage_bps=0.0)

    now = replay_weights(weights, rets, free)
    np.testing.assert_allclose(now["ret"], [0.0, 0.1, 0.2], atol=1e-12)

    late = replay_weights(weights, rets, CostModel(cost_pct=0.0, slippage_bps=0.0,
                                                   delay_bars=1))
    # delayed: bar1 still holds [0.5,0.5] → 0.05; bar2 holds [1,0] → 0.0
    np.testing.assert_allclose(late["ret"], [0.0, 0.05, 0.0], atol=1e-12)


def test_slippage_strictly_hurts_when_trading():
    rng = np.random.default_rng(0)
    T, N = 40, 3
    weights = rng.dirichlet(np.ones(N), size=T)
    rets = np.vstack([np.zeros(N), rng.normal(0.001, 0.01, (T - 1, N))])
    eq = [replay_weights(weights, rets, CostModel(cost_pct=0.001, slippage_bps=s))
          ["equity"].iloc[-1] for s in (0.0, 5.0, 25.0)]
    assert eq[0] > eq[1] > eq[2]


def test_evaluate_report_fields():
    rng = np.random.default_rng(1)
    T, N = 60, 3
    weights = rng.dirichlet(np.ones(N), size=T)
    rets = np.vstack([np.zeros(N), rng.normal(0.0005, 0.01, (T - 1, N))])
    rep = evaluate(weights, rets, CostModel())
    assert {"stats", "win_rate", "active_win_rate", "avg_turnover",
            "total_friction_paid", "equity", "drawdown"} <= set(rep)
    assert 0.0 <= rep["win_rate"] <= 1.0
    assert len(rep["equity"]) == T and len(rep["drawdown"]) == T
    assert np.isfinite(rep["stats"]["sharpe"])
    text = format_report(rep)
    assert "ACTIVE sharpe" in text and "slippage" in text


def test_open_fill_equals_close_fill_without_gaps():
    """When every open equals the previous close, open-fill accounting must reproduce
    close-fill exactly — the two implementations cross-validate."""
    from strategy_eval import replay_weights_ohlc
    rng = np.random.default_rng(4)
    T, N = 50, 3
    weights = rng.dirichlet(np.ones(N), size=T)
    closes = np.vstack([np.full(N, 100.0),
                        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (T - 1, N)), axis=0))])
    opens = np.vstack([closes[[0]], closes[:-1]])             # gapless market
    rets = np.zeros_like(closes)
    rets[1:] = closes[1:] / closes[:-1] - 1.0
    cost = CostModel(cost_pct=0.001, slippage_bps=3.0)
    a = replay_weights(weights, rets, cost)
    b = replay_weights_ohlc(weights, opens, closes, cost)
    np.testing.assert_allclose(b["equity"], a["equity"], rtol=1e-10)


def test_open_fill_old_weights_earn_the_gap():
    """Hand-computed: a +10% overnight gap on the asset you're switching INTO is earned by
    the OLD weights under open fills (you weren't in yet), unlike close fills."""
    from strategy_eval import replay_weights_ohlc
    closes = np.array([[100.0, 100.0], [110.0, 100.0]])       # asset0 gaps +10%, flat after
    opens = np.array([[100.0, 100.0], [110.0, 100.0]])        # whole move is the gap
    weights = np.array([[0.5, 0.5], [1.0, 0.0]])              # switch fully into asset0
    free = CostModel(cost_pct=0.0, slippage_bps=0.0)
    res = replay_weights_ohlc(weights, opens, closes, free)
    assert res["ret"].iloc[1] == pytest.approx(0.05)          # old 50/50 earned the gap
    rets = np.array([[0.0, 0.0], [0.10, 0.0]])
    close_fill = replay_weights(weights, rets, free)
    assert close_fill["ret"].iloc[1] == pytest.approx(0.10)   # close fill credits it all


def test_risk_and_tail_extras_hand_computed():
    """Underwater spell length, VaR/CVaR on a crafted path with known drawdown."""
    rets = np.array([[0.0], [0.10], [-0.05], [0.01], [0.20]])       # single asset
    weights = np.ones((5, 1))
    rep = evaluate(weights, rets, CostModel(cost_pct=0.0, slippage_bps=0.0))
    # equity [1, 1.1, 1.045, 1.0555, 1.2666]: bars 2 and 3 sit under the bar-1 peak
    assert rep["max_dd_duration"] == 2
    assert rep["pct_time_underwater"] == pytest.approx(2 / 5)
    assert rep["var_95"] <= -0.04                       # 5th pct of the 4 live bars
    assert rep["cvar_95"] == pytest.approx(-0.05)       # mean of the tail = the one loss
    assert np.isfinite(rep["skew"])


def test_benchmark_identical_portfolio_scores_zero_active():
    """Constant equal weights ARE the benchmark; float noise must not become a fake
    (anti-)edge in the report."""
    rng = np.random.default_rng(6)
    T, N = 80, 3
    rets = np.vstack([np.zeros(N), rng.normal(0.0005, 0.01, (T - 1, N))])
    eq = np.tile(np.full(N, 1.0 / N), (T, 1))
    rep = evaluate(eq, rets, CostModel(cost_pct=0.001, slippage_bps=5.0))
    assert rep["stats"]["active_sharpe"] == 0.0


def test_report_extras_present_and_sane():
    rng = np.random.default_rng(5)
    T, N = 60, 3
    weights = rng.dirichlet(np.ones(N), size=T)
    rets = np.vstack([np.zeros(N), rng.normal(0.0005, 0.01, (T - 1, N))])
    rep = evaluate(weights, rets, CostModel())
    assert rep["fills"] == "close"
    assert rep["profit_factor"] > 0
    assert 1.0 / N <= rep["hhi_mean"] <= 1.0                  # between spread and all-in
    assert rep["worst_bar"] <= rep["best_bar"]


def test_bt_crosscheck_agrees_with_replay():
    """Zero-cost: FinRL-X's bt engine and our accounting must produce the same equity."""
    rng = np.random.default_rng(2)
    T, N = 30, 2
    tics = ["AAA", "BBB"]
    weights = rng.dirichlet(np.ones(N), size=T)
    rets = np.vstack([np.zeros(N), rng.normal(0.001, 0.015, (T - 1, N))])

    ours = replay_weights(weights, rets, CostModel(cost_pct=0.0, slippage_bps=0.0))
    total_ours = float(ours["equity"].iloc[-1] - 1.0)
    check = bt_crosscheck(weights, rets, tics)
    if check is None:
        pytest.skip("bt not installed")
    assert check["bt_total_return"] == pytest.approx(total_ours, abs=2e-3)
