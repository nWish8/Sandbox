"""marketdata.py — cache behaviour, FinRL contract, provider fallback. No network."""
from __future__ import annotations

import pandas as pd
import pytest

from marketdata import MarketData, OHLCV_COLS


def test_finrl_contract(fake_md):
    df = fake_md.get_finrl(["AAA", "BBB"], "2022-01-01", "2022-06-30")
    assert list(df.columns) == OHLCV_COLS + ["tic", "day"]
    assert df["date"].dtype == object and df["date"].str.match(r"\d{4}-\d{2}-\d{2}").all()
    assert set(df["tic"]) == {"AAA", "BBB"}
    # sorted by (date, tic); day = weekday of date
    assert df.equals(df.sort_values(["date", "tic"]).reset_index(drop=True))
    assert (df["day"] == pd.to_datetime(df["date"]).dt.dayofweek).all()


def test_cache_hit_no_refetch(fake_md):
    fake_md.get_finrl(["AAA"], "2022-01-01", "2022-12-31")
    n = len(fake_md.source.calls)
    again = fake_md.get_finrl(["AAA"], "2022-01-01", "2022-12-31")
    assert len(fake_md.source.calls) == n          # served from parquet, no new fetch
    assert len(again) > 200


def test_subrange_served_from_cache(fake_md):
    full = fake_md.get_ohlcv("AAA", "2022-01-01", "2022-12-31")
    n = len(fake_md.source.calls)
    sub = fake_md.get_ohlcv("AAA", "2022-03-01", "2022-04-30")
    assert len(fake_md.source.calls) == n
    assert sub.index.min() >= pd.Timestamp("2022-03-01")
    assert sub.index.max() <= pd.Timestamp("2022-04-30")
    pd.testing.assert_frame_equal(sub, full.loc["2022-03-01":"2022-04-30"])


def test_range_extension_fetches_union(fake_md):
    fake_md.get_ohlcv("AAA", "2022-03-01", "2022-06-30")
    fake_md.get_ohlcv("AAA", "2021-01-01", "2022-12-31")    # wider both sides
    assert fake_md.source.calls[-1] == ("AAA", "2021-01-01", "2022-12-31")
    # coverage grew: the widest range is now cached, narrow requests hit the cache
    n = len(fake_md.source.calls)
    fake_md.get_ohlcv("AAA", "2021-06-01", "2021-07-01")
    assert len(fake_md.source.calls) == n


def test_wide_view_shape(fake_md):
    px = fake_md.get_ohlcv("AAA", "2022-01-01", "2022-03-31")
    assert list(px.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(px.index, pd.DatetimeIndex)
    assert px.index.is_monotonic_increasing


def test_empty_provider_raises_without_cache(tmp_path):
    class EmptySource:
        name = "empty"

        def fetch(self, tic, start, end):
            return pd.DataFrame(columns=OHLCV_COLS)

    md = MarketData(source=EmptySource(), cache_dir=tmp_path / "ohlcv")
    with pytest.raises(ValueError, match="no data"):
        md.get_ohlcv("ZZZ", "2022-01-01", "2022-02-01")


def test_offline_provider_serves_stale_cache(fake_md, tmp_path):
    fake_md.get_ohlcv("AAA", "2022-01-01", "2022-06-30")

    class DeadSource:
        name = "dead"

        def fetch(self, tic, start, end):
            return pd.DataFrame(columns=OHLCV_COLS)

    offline = MarketData(source=DeadSource(), cache_dir=fake_md.cache_dir)
    px = offline.get_ohlcv("AAA", "2022-01-01", "2022-12-31")   # wider than cached → fetch
    assert len(px) > 100                                        # → empty → stale cache served
