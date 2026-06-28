"""T2.9 — continuous-value features: causal alignment, slope, train-only z-score."""
from __future__ import annotations

import numpy as np
import pytest

from gym import features as F

DAY = 86_400


def _days(n, start=1_700_000_000):
    return [start + i * DAY for i in range(n)]


def _cont_signal(t, v, direction="n/a"):
    """Build a continuous-signal entry as collect.py would emit it."""
    return {"direction": direction, "kind": "continuous",
            "values": {"t": list(t), "v": list(v)}}


# ─── vocab split

def test_cont_vocab_picks_continuous_only(write_fixture):
    cfg, write = write_fixture
    n = 12
    t = _days(n)
    c = [100.0 + i for i in range(n)]
    sigs = {
        "Study|RSI Osc": _cont_signal(t, [50 + i for i in range(n)]),
        "Study|Buy Signal": {"direction": "long", "kind": "event",
                             "fires": [t[3], t[7]]},
    }
    write("TEST:AAA", t, c, signals=sigs)

    ev = F.build_vocab(["TEST:AAA"], cfg)
    co = F.build_cont_vocab(["TEST:AAA"], cfg)
    ev_names = {v["short"] for v in ev}
    co_names = {v["short"] for v in co}
    assert "RSI Osc" in co_names
    assert "RSI Osc" not in ev_names          # continuous excluded from event vocab
    assert "Buy Signal" in ev_names
    assert "Buy Signal" not in co_names


# ─── causal alignment + slope

def test_continuous_value_aligns_to_bar(write_fixture):
    cfg, write = write_fixture
    n = 10
    t = _days(n)
    c = [100.0] * n
    vals = [float(10 * i) for i in range(n)]    # 0,10,20,...
    write("TEST:BBB", t, c, signals={"S|Osc": _cont_signal(t, vals)})

    cont_vocab = F.build_cont_vocab(["TEST:BBB"], cfg)
    df = F.build_ticker_frame("TEST:BBB", F.build_vocab(["TEST:BBB"], cfg), cfg,
                              cont_vocab=cont_vocab)
    cp = cont_vocab[0]["col"]
    # single-frame build does NOT z-score (that happens in build_features) -> raw values
    np.testing.assert_allclose(df[f"{cp}_val"].to_numpy(), vals, rtol=1e-6)


def test_continuous_slope_3(write_fixture):
    cfg, write = write_fixture
    n = 10
    t = _days(n)
    c = [100.0] * n
    vals = [float(i * i) for i in range(n)]     # 0,1,4,9,16,...
    write("TEST:CCC", t, c, signals={"S|Osc": _cont_signal(t, vals)})

    cont_vocab = F.build_cont_vocab(["TEST:CCC"], cfg)
    df = F.build_ticker_frame("TEST:CCC", F.build_vocab(["TEST:CCC"], cfg), cfg,
                              cont_vocab=cont_vocab)
    cp = cont_vocab[0]["col"]
    slope = df[f"{cp}_slope_3"].to_numpy()
    assert np.isnan(slope[0]) or slope[0] == 0  # first 3 undefined (filled later)
    # slope[5] = val[5]-val[2] = 25-4 = 21
    assert abs(slope[5] - (vals[5] - vals[2])) < 1e-5


def test_continuous_warmup_is_causal(write_fixture):
    """A value present only from bar 4 onward leaves earlier bars NaN (no future leak)."""
    cfg, write = write_fixture
    n = 10
    t = _days(n)
    c = [100.0] * n
    # values only exist for bars 4..9
    vt = t[4:]
    vv = [float(i) for i in range(4, n)]
    write("TEST:DDD", t, c, signals={"S|Osc": _cont_signal(vt, vv)})

    cont_vocab = F.build_cont_vocab(["TEST:DDD"], cfg)
    df = F.build_ticker_frame("TEST:DDD", F.build_vocab(["TEST:DDD"], cfg), cfg,
                              cont_vocab=cont_vocab)
    cp = cont_vocab[0]["col"]
    val = df[f"{cp}_val"].to_numpy()
    assert np.isnan(val[:4]).all()             # nothing before first value
    assert not np.isnan(val[4:]).any()


# ─── train-only z-score

def test_cont_zscore_fit_is_train_only(write_fixture):
    """_fit_cont_zscore must use only the frames passed (train); a held-out frame
    with extreme values cannot shift the fitted mean/std."""
    cfg, write = write_fixture
    n = 12
    t = _days(n)
    c = [100.0] * n
    # two 'train' frames with modest values, one 'test' frame with extreme values
    write("TEST:TR1", t, c, signals={"S|Osc": _cont_signal(t, [float(i) for i in range(n)])})
    write("TEST:TR2", t, c, signals={"S|Osc": _cont_signal(t, [float(i + 1) for i in range(n)])})
    write("TEST:TST", t, c, signals={"S|Osc": _cont_signal(t, [9999.0] * n)})

    cont_vocab = F.build_cont_vocab(["TEST:TR1", "TEST:TR2", "TEST:TST"], cfg)
    vocab = F.build_vocab(["TEST:TR1", "TEST:TR2", "TEST:TST"], cfg)
    tr1 = F.build_ticker_frame("TEST:TR1", vocab, cfg, cont_vocab)
    tr2 = F.build_ticker_frame("TEST:TR2", vocab, cfg, cont_vocab)
    tst = F.build_ticker_frame("TEST:TST", vocab, cfg, cont_vocab)

    stats_train = F._fit_cont_zscore([tr1, tr2], cont_vocab)
    stats_all   = F._fit_cont_zscore([tr1, tr2, tst], cont_vocab)
    cp = cont_vocab[0]["col"]
    # the extreme test frame would massively inflate the mean if included
    assert stats_train[cp][0] < 50, stats_train
    assert stats_all[cp][0] > 100, "sanity: including extreme frame DOES shift the fit"


def test_build_features_stores_cont_zscore(write_fixture):
    """End-to-end: build_features writes cont_zscore normalizers and z-scored vals."""
    import json
    cfg, write = write_fixture
    n = 30
    t = _days(n)
    rng = np.random.default_rng(0)
    for i in range(8):
        c = list(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
        vals = list(rng.normal(50, 10, n))
        write(f"TEST:T{i}", t, c, signals={"S|Osc": _cont_signal(t, vals)})

    F.build_features(cfg)
    splits = json.loads(cfg.splits_file.read_text(encoding="utf-8"))
    assert "cont_zscore" in splits["normalizers"]
    assert len(splits["cont_vocab"]) >= 1

    # pooled train z-scored vals should be ~0 mean
    cp = splits["cont_vocab"][0]["col"]
    train = splits["training_tickers"]
    import pandas as pd
    pooled = pd.concat([F.load_ticker(F.ticker_slug(s), cfg)[f"{cp}_val"] for s in train])
    assert abs(pooled.mean()) < 0.5, f"z-scored train mean should be ~0, got {pooled.mean()}"
