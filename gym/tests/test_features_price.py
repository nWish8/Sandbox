"""T2.1 — loader, meta/exec, and price features on a controlled fixture."""
from __future__ import annotations

import numpy as np

from gym import features as F

DAY = 86400


def _days(n, start=1_700_000_000):
    return [start + i * DAY for i in range(n)]


def test_fill_price_next_open_with_ohlc(write_fixture):
    cfg, write = write_fixture
    n = 12
    c = [100.0 + i for i in range(n)]
    o = [c[i] - 0.5 for i in range(n)]
    write("TEST:AAA", _days(n), c, ohlc={"o": o, "h": [x + 1 for x in c],
                                          "l": [x - 1 for x in c], "v": [1] * n})
    vocab = F.build_vocab(["TEST:AAA"], cfg)
    df = F.build_ticker_frame("TEST:AAA", vocab, cfg)
    # fill at next bar's open; last row NaN
    assert df["fill_price"].iloc[0] == o[1]
    assert df["fill_price"].iloc[5] == o[6]
    assert np.isnan(df["fill_price"].iloc[-1])


def test_fill_price_fallback_is_prior_close(write_fixture):
    """Close-only: open := prior close, so fill_price[i] = open[i+1] = close[i]
    (the close-to-close fill point); last row NaN."""
    cfg, write = write_fixture
    n = 8
    c = [10.0 * (i + 1) for i in range(n)]
    write("TEST:BBB", _days(n), c)             # no OHLC
    vocab = F.build_vocab(["TEST:BBB"], cfg)
    df = F.build_ticker_frame("TEST:BBB", vocab, cfg)
    assert df["fill_price"].iloc[0] == c[0]
    assert df["fill_price"].iloc[5] == c[5]
    assert np.isnan(df["fill_price"].iloc[-1])


def test_log_returns(write_fixture):
    cfg, write = write_fixture
    n = 10
    c = [100.0 * (1.1 ** i) for i in range(n)]   # constant +10% per bar
    write("TEST:CCC", _days(n), c)
    vocab = F.build_vocab(["TEST:CCC"], cfg)
    df = F.build_ticker_frame("TEST:CCC", vocab, cfg)
    assert np.isnan(df["ret_1"].iloc[0])
    assert np.isclose(df["ret_1"].iloc[1], np.log(1.1))
    assert np.isclose(df["ret_3"].iloc[3], np.log(1.1 ** 3))


def test_price_features_are_trailing_only(write_fixture):
    cfg, write = write_fixture
    n = 40
    c = list(100 + np.cumsum(np.r_[0.0, np.random.default_rng(0).normal(0, 1, n - 1)]))
    write("TEST:DDD", _days(n), c)
    vocab = F.build_vocab(["TEST:DDD"], cfg)
    df = F.build_ticker_frame("TEST:DDD", vocab, cfg)
    # vol_20 needs 20 obs -> NaN before that
    assert df["vol_20"].iloc[:19].isna().all()
    assert not np.isnan(df["vol_20"].iloc[25])
