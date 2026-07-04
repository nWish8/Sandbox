"""projections.py — target alignment (no lookahead into fitting), band ordering,
walk-forward coverage sanity, projection integration. Fake provider, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections import (FEATURES, build_dataset, fit_quantiles, pooled_dataset,
                         predict_bands, project, walkforward_coverage)


def _px(n=400, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))), index=idx)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.integers(1, 9, n) * 1e6})


def test_target_is_exact_forward_return():
    h = 10
    px = _px()
    d = build_dataset(px, horizon=h)
    c = px["close"]
    expected = c.shift(-h) / c - 1.0
    pd.testing.assert_series_equal(d["target"], expected, check_names=False)
    assert d["target"].iloc[-h:].isna().all()          # unknown future stays NaN


def test_features_unchanged_when_future_truncated():
    """Row t's features must be identical whether or not later bars exist."""
    px = _px()
    full = build_dataset(px, horizon=5)
    short = build_dataset(px.iloc[:300], horizon=5)
    pd.testing.assert_frame_equal(full[FEATURES].iloc[:300], short[FEATURES])


def test_bands_never_cross():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURES)))
    y = X[:, 0] * 0.01 + rng.normal(0, 0.02, 400)
    models = fit_quantiles(X, y, seed=0)
    bands = predict_bands(models, X[:50])
    assert (np.diff(bands, axis=1) >= 0).all()


def test_walkforward_coverage_sane(fake_md):
    data = pooled_dataset(fake_md, ["AAA", "BBB", "CCC"], "2019-01-01", "2022-12-31",
                          horizon=10)
    cov = walkforward_coverage(data, horizon=10, seed=0)
    assert cov["n_test"] > 100
    # nominal band is 80%; on stationary synthetic data anything wildly outside
    # [0.55, 0.98] means the quantile machinery is broken, not just noisy
    assert 0.55 <= cov["coverage"] <= 0.98


def test_walkforward_embargo_purges_overlapping_targets(fake_md):
    """A train row's target window [t, t+h] must never overlap the test period: the last
    training date has to sit at least `horizon` trading days before the first test date."""
    h = 10
    data = pooled_dataset(fake_md, ["AAA", "BBB"], "2019-01-01", "2022-12-31", horizon=h)
    cov = walkforward_coverage(data, horizon=h, seed=0)
    dates = data.dropna(subset=["target"]).index.unique().sort_values()
    gap = (dates.get_loc(pd.Timestamp(cov["test_start"]))
           - dates.get_loc(pd.Timestamp(cov["train_end"])))
    assert gap >= h


def test_walkforward_refuses_tiny_samples():
    idx = pd.date_range("2022-01-03", periods=40, freq="B")
    tiny = pd.DataFrame({f: np.random.default_rng(0).normal(size=40) for f in FEATURES},
                        index=idx)
    tiny["target"] = 0.0
    out = walkforward_coverage(tiny, horizon=5)
    assert np.isnan(out["coverage"])                   # refuses rather than pretending


def test_project_integration(fake_md):
    table, honesty = project(["AAA", "BBB"], horizon=10, end="2022-12-31",
                             window_days=1000, md=fake_md, seed=0)
    assert sorted(table["tic"]) == ["AAA", "BBB"]
    assert {"close", "q10_pct", "q50_pct", "q90_pct"} <= set(table.columns)
    assert (table["q10_pct"] <= table["q50_pct"]).all()
    assert (table["q50_pct"] <= table["q90_pct"]).all()
    assert "coverage" in honesty
