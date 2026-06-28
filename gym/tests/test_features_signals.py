"""T2.2 — signal, condition, interaction features on controlled fixtures."""
from __future__ import annotations

import numpy as np

from gym import features as F

DAY = 86400


def _days(n, start=1_700_000_000):
    return [start + i * DAY for i in range(n)]


def _sig(direction, fire_idx, t):
    return {"direction": direction, "fires": [t[i] for i in fire_idx]}


def test_fired_bars_since_within(write_fixture):
    cfg, write = write_fixture
    n = 10
    t = _days(n)
    # a long signal fires on bars 2 and 6
    signals = {"Prophet|Buy Signal": _sig("long", [2, 6], t)}
    write("TEST:AAA", t, [100.0] * n, signals=signals)
    vocab = F.build_vocab(["TEST:AAA"], cfg)
    df = F.build_ticker_frame("TEST:AAA", vocab, cfg)

    fired = df["buy_signal_fired"].to_numpy()
    assert list(np.flatnonzero(fired == 1.0)) == [2, 6]
    bs = df["buy_signal_bars_since"].to_numpy()
    assert bs[0] == cfg.bars_since_cap and bs[1] == cfg.bars_since_cap   # before first fire
    assert bs[2] == 0 and bs[3] == 1 and bs[4] == 2
    assert bs[6] == 0 and bs[7] == 1
    within = df["buy_signal_within_3"].to_numpy()   # bars_since <= 2
    assert within[2] == 1 and within[4] == 1 and within[5] == 0
    assert df["buy_signal_ever"].to_numpy()[1] == 0 and df["buy_signal_ever"].to_numpy()[2] == 1


def test_active_counts_by_direction(write_fixture):
    cfg, write = write_fixture
    n = 8
    t = _days(n)
    signals = {
        "P|Buy Signal": _sig("long", [3], t),
        "P|Small Buy": _sig("long", [3], t),
        "P|Sell Signal": _sig("short", [3], t),
    }
    write("TEST:BBB", t, [100.0] * n, signals=signals)
    vocab = F.build_vocab(["TEST:BBB"], cfg)
    df = F.build_ticker_frame("TEST:BBB", vocab, cfg)
    # within_3 window = bars 3,4,5
    assert df["bull_active_count"].iloc[3] == 2
    assert df["bull_active_count"].iloc[5] == 2
    assert df["bull_active_count"].iloc[6] == 0
    assert df["bear_active_count"].iloc[3] == 1


def test_cloud_state_collapses_senkou(write_fixture):
    cfg, write = write_fixture
    n = 8
    t = _days(n)
    signals = {
        "Ichi|Senkou Span A (26 Period) Above Span B Cloud": _sig("n/a", [1, 2], t),  # bull
        "Ichi|Senkou Span A (26 Period) Below Span B Cloud": _sig("n/a", [5, 6], t),  # bear
    }
    write("TEST:CCC", t, [100.0] * n, signals=signals)
    vocab = F.build_vocab(["TEST:CCC"], cfg)
    df = F.build_ticker_frame("TEST:CCC", vocab, cfg)
    cs = df["cloud_state"].to_numpy()
    assert cs[0] == 0          # undecided before first cloud fire
    assert cs[1] == 1 and cs[2] == 1 and cs[3] == 1   # bull, forward-filled
    assert cs[5] == -1 and cs[7] == -1                # flips bear, forward-filled


def test_every_event_signal_becomes_columns(tmp_cfg):
    """All signals in the universe are represented (no data-layer pruning, spec D5).
    Every event-vocab entry yields its column set; every continuous-vocab entry yields
    a _val column; the two vocabularies are mutually exclusive (continuous wins)."""
    cfg = tmp_cfg
    tickers = F.load_tickers(cfg.tickers_file)
    vocab = F.build_vocab(tickers, cfg)
    cont_vocab = F.build_cont_vocab(tickers, cfg)
    df = F.build_ticker_frame("NYSE:LMT", vocab, cfg, cont_vocab=cont_vocab)
    cols = set(df.columns)

    assert len(vocab) > 0
    for v in vocab:
        assert f"{v['col']}_fired" in cols, v["short"]
    for v in cont_vocab:
        assert f"{v['col']}_val" in cols, v["short"]

    # a plot continuous on any ticker must not also leak event columns
    ev_names = {v["short"] for v in vocab}
    co_names = {v["short"] for v in cont_vocab}
    assert ev_names.isdisjoint(co_names)
