"""T4 — investigation harness: the chronological holdout split and the reporting/verdict
logic. Pure-data tests (no network, no training)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from investigate import format_table, split_val_test, verdict
from portfolio import add_cov_features


def _cov_df(n_bars=40, lookback=5):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2021-01-01", periods=n_bars, freq="D")
    rows = []
    for tic in ("AAA", "BBB", "CCC"):
        price = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n_bars))
        for d, p in zip(dates, price):
            rows.append({"date": d, "tic": tic, "close": p, "macd": rng.normal()})
    return add_cov_features(pd.DataFrame(rows), lookback=lookback)


def test_split_val_test_is_chronological_and_disjoint():
    df = _cov_df()
    val, test = split_val_test(df, val_frac=0.5)
    # re-indexed from 0
    assert val.index.min() == 0 and test.index.min() == 0
    # disjoint, contiguous, covering the whole OOS period in time order
    assert val["date"].max() < test["date"].min()
    assert val["date"].nunique() + test["date"].nunique() == df.index.nunique()


def test_split_handles_tiny_period_without_empty_side():
    df = _cov_df(n_bars=12, lookback=5)        # only 7 OOS bars
    val, test = split_val_test(df, val_frac=0.5)
    assert val.index.nunique() >= 1 and test.index.nunique() >= 1


def test_verdict_reads_positive_and_negative_test_edge():
    pos = [{"reward": "active_dsr", "val": {"active_sharpe": 1.2},
            "test": {"active_sharpe": 0.8}}]
    neg = [{"reward": "logret", "val": {"active_sharpe": 0.9},
            "test": {"active_sharpe": -0.4}}]
    assert "POSITIVE" in verdict(pos)
    assert "negative" in verdict(neg).lower()


def test_format_table_renders_all_rewards():
    rows = [
        {"reward": "active_dsr", "val": {"active_sharpe": 1.0, "sharpe": 0.5},
         "test": {"active_sharpe": 0.2, "sharpe": 0.3}},
        {"reward": "logret", "val": {"active_sharpe": -0.1, "sharpe": 0.4},
         "test": {"active_sharpe": -0.2, "sharpe": 0.2}},
    ]
    table = format_table(rows, metrics=("active_sharpe", "sharpe"))
    assert "active_dsr" in table and "logret" in table
    assert "val_active_s" in table and "test_active_s" in table
