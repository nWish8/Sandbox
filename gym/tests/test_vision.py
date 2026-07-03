"""vision.py pure logic — positions table math, state loading, correlation matrix.
(The Qt shell is verified by offscreen screenshots, not unit-tested.)"""
from __future__ import annotations

import json

import numpy as np
import pytest

from vision import build_positions_table, load_portfolio_state, returns_correlation


def test_load_state_missing_file(tmp_path):
    assert load_portfolio_state(tmp_path / "nope.json") == []


def test_load_state_roundtrip(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"positions": [{"tic": "AAPL", "shares": 2, "cost_basis": 100}]}),
                 encoding="utf-8")
    assert load_portfolio_state(f)[0]["tic"] == "AAPL"


def test_positions_table_math():
    positions = [{"tic": "AAA", "shares": 10, "cost_basis": 100.0},
                 {"tic": "BBB", "shares": 5, "cost_basis": 200.0}]
    last = {"AAA": 110.0, "BBB": 180.0}
    prev = {"AAA": 100.0, "BBB": 200.0}
    t = build_positions_table(positions, last, prev)

    a, b = t["rows"]
    assert a["value"] == pytest.approx(1100.0) and b["value"] == pytest.approx(900.0)
    assert a["pnl"] == pytest.approx(100.0) and b["pnl"] == pytest.approx(-100.0)
    assert a["pnl_pct"] == pytest.approx(0.10) and b["pnl_pct"] == pytest.approx(-0.10)
    assert a["day_pct"] == pytest.approx(0.10) and b["day_pct"] == pytest.approx(-0.10)
    assert a["weight"] + b["weight"] == pytest.approx(1.0)
    assert t["total_value"] == pytest.approx(2000.0)
    assert t["total_pnl"] == pytest.approx(0.0)


def test_positions_table_missing_price():
    t = build_positions_table([{"tic": "GONE", "shares": 1, "cost_basis": 10.0}], {}, {})
    assert np.isnan(t["rows"][0]["last"])


def test_returns_correlation(fake_md):
    corr = returns_correlation(fake_md, ["AAA", "BBB", "CCC"], end="2022-12-31", days=60)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
    assert ((corr.to_numpy() >= -1.0) & (corr.to_numpy() <= 1.0)).all()
