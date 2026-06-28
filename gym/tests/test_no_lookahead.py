"""T2.4 — the critical causality guarantees. A missed leak invalidates every result.

(a) feature@t depends only on bars <= t;
(b) normalizers (vol terciles) are fit on TRAINING tickers only — mutating a
    validation/test ticker cannot move them;
(c) target uses only future data and is NaN on the final H bars.
"""
from __future__ import annotations

import json

import numpy as np

from gym import features as F
from gym.config import load_tickers, slug


def test_feature_causality_perturbation(tmp_cfg):
    """Perturbing bar k leaves every FEATURE at bars < k byte-identical."""
    cfg = tmp_cfg
    vocab = F.build_vocab(load_tickers(cfg.tickers_file), cfg)
    base = F.build_ticker_frame("NYSE:LMT", vocab, cfg)
    feat_cols = F.feature_columns(base, cfg)

    # mutate the source close at bar k to a wild value, rebuild
    k = 300
    path = cfg.signal_study_results / "effectiveness_NYSE_LMT.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for tf in data["timeframes"]:
        if tf["label"] == "1D":
            tf["series"]["c"][k] *= 5.0
    path.write_text(json.dumps(data), encoding="utf-8")

    perturbed = F.build_ticker_frame("NYSE:LMT", vocab, cfg)
    before = base[feat_cols].iloc[:k].to_numpy()
    after = perturbed[feat_cols].iloc[:k].to_numpy()
    assert np.array_equal(np.nan_to_num(before, nan=-999), np.nan_to_num(after, nan=-999))
    # and the perturbation DID change something at/after k (sanity: the test can detect leaks)
    assert not np.array_equal(
        np.nan_to_num(base[feat_cols].iloc[k:].to_numpy(), nan=-999),
        np.nan_to_num(perturbed[feat_cols].iloc[k:].to_numpy(), nan=-999),
    )


def test_normalizers_fit_on_training_only(tmp_cfg):
    """Editing a TEST ticker's prices must not change vol terciles or any training
    row's vol_regime."""
    cfg = tmp_cfg
    F.build_features(cfg)
    sp = json.loads(cfg.splits_file.read_text())
    edges_before = sp["normalizers"]["vol_tercile_edges"]
    train_slug = slug(sp["training_tickers"][0])
    train_regime_before = F.load_ticker(train_slug, cfg)["vol_regime"].to_numpy()

    # blow up a TEST ticker's volatility at the source and rebuild
    test_sym = sp["test_tickers"][0]
    path = cfg.signal_study_results / f"effectiveness_{slug(test_sym)}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for tf in data["timeframes"]:
        if tf["label"] == "1D":
            c = tf["series"]["c"]
            tf["series"]["c"] = [x * (1 + 0.5 * (i % 2)) for i, x in enumerate(c)]
    path.write_text(json.dumps(data), encoding="utf-8")

    F.build_features(cfg)
    sp2 = json.loads(cfg.splits_file.read_text())
    assert sp2["normalizers"]["vol_tercile_edges"] == edges_before
    assert np.array_equal(F.load_ticker(train_slug, cfg)["vol_regime"].to_numpy(),
                          train_regime_before)


def test_terciles_equal_train_only_fit(tmp_cfg):
    """The stored edges equal a fit over TRAINING frames alone (not the full pool).

    Refit from freshly built (pre-fill) frames — the parquet's vol_20 has its warmup
    NaNs filled to 0 for PPO, which would contaminate a refit, whereas the build fits
    terciles on the raw pre-fill volatility.
    """
    cfg = tmp_cfg
    F.build_features(cfg)
    sp = json.loads(cfg.splits_file.read_text())
    vocab = F.build_vocab(load_tickers(cfg.tickers_file), cfg)
    train_frames = [F.build_ticker_frame(t, vocab, cfg) for t in sp["training_tickers"]]
    recomputed = F._fit_vol_terciles(train_frames)
    assert np.allclose(recomputed, sp["normalizers"]["vol_tercile_edges"])
    # and a fit over ALL tickers would differ (proving it's train-only, not the full pool)
    all_frames = [F.build_ticker_frame(t, vocab, cfg) for t in load_tickers(cfg.tickers_file)]
    assert not np.allclose(F._fit_vol_terciles(all_frames),
                           sp["normalizers"]["vol_tercile_edges"])


def test_target_uses_future_only(tmp_cfg):
    cfg = tmp_cfg
    vocab = F.build_vocab(load_tickers(cfg.tickers_file), cfg)
    df = F.build_ticker_frame("NYSE:LMT", vocab, cfg)
    H = cfg.target_horizon
    c = df["close"].to_numpy()
    fwd = df[f"fwd_ret_{H}"].to_numpy()
    i = 100
    assert np.isclose(fwd[i], c[i + H] / c[i] - 1.0)       # exactly the H-bar fwd return
    assert np.isnan(fwd[-1])                                # last H bars NaN
