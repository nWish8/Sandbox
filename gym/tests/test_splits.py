"""T2.3 — 3-way ticker split: disjoint, complete, stratified, deterministic."""
from __future__ import annotations

from gym import features as F
from gym.config import asset_class_of, load_tickers


def test_splits_disjoint_and_complete(config):
    tickers = load_tickers(config.tickers_file)
    sp = F.make_splits(tickers, config)
    tr, va, te = sp["training_tickers"], sp["validation_tickers"], sp["test_tickers"]
    allsets = tr + va + te
    assert sorted(allsets) == sorted(tickers)              # complete
    assert len(set(tr) & set(va)) == 0                     # disjoint
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0


def test_splits_deterministic(config):
    tickers = load_tickers(config.tickers_file)
    a = F.make_splits(tickers, config)
    b = F.make_splits(tickers, config)
    assert a["training_tickers"] == b["training_tickers"]
    assert a["validation_tickers"] == b["validation_tickers"]
    assert a["test_tickers"] == b["test_tickers"]


def test_training_spans_all_classes_and_holdout_spans_multiple(config):
    tickers = load_tickers(config.tickers_file)
    sp = F.make_splits(tickers, config)
    classes = {asset_class_of(t) for t in tickers}
    train_classes = {asset_class_of(t) for t in sp["training_tickers"]}
    assert train_classes == classes                        # every class seen in training
    assert len({asset_class_of(t) for t in sp["validation_tickers"]}) >= 2
    assert len({asset_class_of(t) for t in sp["test_tickers"]}) >= 2
    assert len(sp["validation_tickers"]) > 0 and len(sp["test_tickers"]) > 0


def test_build_roundtrips_and_target_nan_tail(tmp_cfg):
    cfg = tmp_cfg
    F.build_features(cfg)
    df = F.load_ticker("NYSE_LMT", cfg)
    H = cfg.target_horizon
    assert df[f"fwd_edge_{H}"].iloc[-H:].isna().all()
    assert df[f"fwd_edge_{H}"].iloc[:-H].notna().all()
    # dtypes survive round-trip; feature cols are NaN-free for PPO
    import json
    sp = json.loads(cfg.splits_file.read_text())
    assert not df[sp["feature_columns"]].isna().any().any()
