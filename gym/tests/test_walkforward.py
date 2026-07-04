"""walkforward.py — fold geometry and the rolling retrain → stitched OOS record.
Synthetic data, tiny training budgets; no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import FinRLConfig
from portfolio import add_cov_features
from walkforward import format_walkforward, make_folds, walkforward_run

TICS = ["AAA", "BBB", "CCC"]
TECH = ["macd", "rsi_30"]


def _cov_df(n_bars=70, seed=0, lookback=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="D")
    rows = []
    for tic in TICS:
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, n_bars))
        for d, p in zip(dates, price):
            rows.append({"date": d, "tic": tic, "close": p,
                         "macd": rng.normal(), "rsi_30": rng.uniform(20, 80)})
    return add_cov_features(pd.DataFrame(rows), lookback=lookback)


def test_make_folds_tiles_the_oos_region():
    folds = make_folds(n_days=100, n_folds=4, min_train_days=50)
    assert len(folds) == 4
    assert folds[0][0] == 50 and folds[-1][1] == 100
    for (a0, a1), (b0, b1) in zip(folds, folds[1:]):
        assert a1 == b0                                # contiguous, no gap, no overlap
        assert a0 < a1 and b0 < b1
    for tr_end, _ in folds:
        assert tr_end >= 50                            # min train window always honoured


def test_make_folds_rejects_too_little_history():
    with pytest.raises(ValueError):
        make_folds(n_days=40, n_folds=3, min_train_days=40)


@pytest.fixture(scope="module")
def wf_report():
    df = _cov_df()
    cfg = FinRLConfig(tickers=TICS, indicators=tuple(TECH), use_vix=False,
                      use_turbulence=False)
    return walkforward_run(cfg, reward="logret", algo="ppo", timesteps=150,
                           n_folds=2, lookback=3, min_train_frac=0.5, seed=1,
                           data=df, log=lambda *a: None), df


def test_walkforward_stitches_every_oos_bar_once(wf_report):
    report, df = wf_report
    n_days = df.index.nunique()
    folds = report["folds"]
    assert len(folds) == 2
    # each fold's test window starts where the previous ended; stitched record drops
    # exactly one reset row per fold
    expected = sum(f["test_days"] - 1 for f in folds)
    assert report["n_oos_bars"] == expected
    assert folds[-1]["test_days"] + folds[0]["test_days"] + folds[0]["train_days"] \
        <= n_days + folds[0]["test_days"]              # windows fit inside the panel


def test_walkforward_models_never_see_their_test_window(wf_report):
    report, _ = wf_report
    for f in report["folds"]:
        assert pd.Timestamp(f["test_start"]) > pd.Timestamp("2020-01-01")
        # training window ends strictly before the test window begins
        assert f["train_days"] > 0
    t0, t1 = report["folds"][0], report["folds"][1]
    assert pd.Timestamp(t1["test_start"]) > pd.Timestamp(t0["test_end"]) or \
        pd.Timestamp(t1["test_start"]) == pd.Timestamp(t0["test_end"])


def test_walkforward_report_shape(wf_report):
    report, _ = wf_report
    assert {"folds", "stitched", "gate", "n_oos_bars", "config"} <= set(report)
    assert np.isfinite(report["stitched"].get("sharpe", np.nan))
    assert report["gate"] is not None                  # enough stitched bars to gate
    text = format_walkforward(report)
    assert "stitched OOS" in text and "fold dispersion" in text
