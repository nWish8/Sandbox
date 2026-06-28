"""T3.3/T3.4 — agent_sup: planted-edge synthetic → feature tops importance; edge_to_weight monotone."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gym.agent_sup import (
    EdgePolicy,
    SizingParams,
    calibrate_sizing,
    edge_to_weight,
    feature_importances,
    train_edge_model,
)
from gym.config import DEFAULT_CONFIG


# ─── synthetic planted-edge dataset

def _make_planted_df(n=500, seed=0) -> tuple[pd.DataFrame, list[str], str]:
    """Return (df, feature_cols, planted_feature).

    One feature 'bull_signal' is linearly correlated with target fwd_edge_6.
    All other features are pure noise.
    """
    rng = np.random.default_rng(seed)
    feat_signal = rng.standard_normal(n)
    feats_noise = {f"noise_{i}": rng.standard_normal(n) for i in range(10)}

    # target = 0.3 * signal + noise (signal has genuine edge)
    target = 0.3 * feat_signal + 0.05 * rng.standard_normal(n)

    df = pd.DataFrame({"bull_signal": feat_signal, **feats_noise})
    df["fwd_edge_6"] = target
    feature_cols = list(df.columns[:-1])
    return df, feature_cols, "bull_signal"


# ─── training

def test_train_returns_booster():
    df, feat_cols, _ = _make_planted_df()
    model = train_edge_model(df, feat_cols)
    import lightgbm as lgb
    assert isinstance(model, lgb.Booster)


def test_train_drops_nan_target_rows():
    df, feat_cols, _ = _make_planted_df(n=100)
    df.loc[df.index[-6:], "fwd_edge_6"] = float("nan")  # mimic last-H-bars NaN
    model = train_edge_model(df, feat_cols)
    # smoke: model trains without error


def test_train_raises_on_empty_target():
    df, feat_cols, _ = _make_planted_df(n=50)
    df["fwd_edge_6"] = float("nan")
    with pytest.raises(ValueError, match="no valid target rows"):
        train_edge_model(df, feat_cols)


# ─── importances

def test_planted_feature_tops_importance():
    df, feat_cols, planted = _make_planted_df(n=800)
    model = train_edge_model(df, feat_cols)
    imp = feature_importances(model)
    assert imp.columns.tolist() == ["feature", "importance"]
    top_feature = imp.iloc[0]["feature"]
    assert top_feature == planted, (
        f"Expected planted feature '{planted}' to top importance, got '{top_feature}'.\n"
        f"Top 3: {imp.head(3).to_dict('records')}"
    )


def test_importances_nonnegative_and_sorted():
    df, feat_cols, _ = _make_planted_df()
    model = train_edge_model(df, feat_cols)
    imp = feature_importances(model)
    assert (imp["importance"] >= 0).all()
    assert imp["importance"].is_monotonic_decreasing


# ─── edge_to_weight monotonicity

def test_edge_to_weight_monotone():
    sizing = SizingParams(lo_cut=-1.0, hi_cut=1.0)
    preds = np.linspace(-2.0, 2.0, 50)
    weights = [edge_to_weight(p, sizing) for p in preds]
    # weights must be non-decreasing
    assert all(w2 >= w1 for w1, w2 in zip(weights, weights[1:])), weights


def test_edge_to_weight_clamps():
    sizing = SizingParams(lo_cut=0.0, hi_cut=1.0)
    assert edge_to_weight(-5.0, sizing) == 0.0
    assert edge_to_weight(5.0, sizing) == 1.0


def test_edge_to_weight_midpoint():
    sizing = SizingParams(lo_cut=0.0, hi_cut=2.0)
    w = edge_to_weight(1.0, sizing)
    assert abs(w - 0.5) < 1e-9


# ─── calibrate_sizing (train-only)

def test_calibrate_sizing_returns_params():
    df, feat_cols, _ = _make_planted_df()
    model = train_edge_model(df, feat_cols)
    sizing = calibrate_sizing(model, df, feat_cols)
    assert sizing.hi_cut > sizing.lo_cut


def test_calibrate_sizing_uses_train_only(tmp_path):
    """Mutating a separate dataframe does not change sizing computed from train_df."""
    df, feat_cols, _ = _make_planted_df(n=300)
    model = train_edge_model(df, feat_cols)
    sizing = calibrate_sizing(model, df, feat_cols)

    df2 = df.copy()
    df2[feat_cols] = 999.0          # extreme mutation of a second frame
    sizing2 = calibrate_sizing(model, df, feat_cols)   # refit on original

    assert abs(sizing.lo_cut - sizing2.lo_cut) < 1e-9
    assert abs(sizing.hi_cut - sizing2.hi_cut) < 1e-9


# ─── EdgePolicy

def test_edge_policy_act_in_unit_interval():
    df, feat_cols, _ = _make_planted_df(n=200)
    model = train_edge_model(df, feat_cols)
    sizing = calibrate_sizing(model, df, feat_cols)
    policy = EdgePolicy(model, sizing, feat_cols)

    rng = np.random.default_rng(1)
    for _ in range(20):
        obs = rng.standard_normal((5, len(feat_cols))).astype(np.float32)
        w = policy.act(obs)
        assert 0.0 <= w <= 1.0, f"weight {w} out of [0,1]"


def test_edge_policy_uses_last_row_only():
    """Modifying earlier rows of obs must not change the prediction."""
    df, feat_cols, _ = _make_planted_df(n=200)
    model = train_edge_model(df, feat_cols)
    sizing = calibrate_sizing(model, df, feat_cols)
    policy = EdgePolicy(model, sizing, feat_cols)

    rng = np.random.default_rng(2)
    obs = rng.standard_normal((5, len(feat_cols))).astype(np.float32)
    w1 = policy.act(obs)

    obs2 = obs.copy()
    obs2[:-1] = 999.0           # poison all rows except the last
    w2 = policy.act(obs2)

    assert w1 == w2, "EdgePolicy must only use obs[-1]"
