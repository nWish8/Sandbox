"""T1.1 / T1.2 — foundation: config populates, tickers load, every ticker has a class."""
from __future__ import annotations


def test_config_has_documented_defaults(config):
    # a sampling of the documented defaults (spec §4 / config.py)
    assert config.target_horizon == 6
    assert config.lookback == 30
    assert config.commission_bps == 10.0
    assert config.reward_decay == 0.02
    assert config.split_ratios == (0.70, 0.15, 0.15)
    assert config.seed == 42
    assert config.gbt.deterministic is True
    assert config.ppo.ent_coef > 0


def test_config_serialises(tmp_path, config):
    p = tmp_path / "meta.json"
    config.save(p)
    import json
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["target_horizon"] == 6
    assert isinstance(d["signal_study_results"], str)   # Path -> str


def test_tickers_load():
    from gym.config import load_tickers
    tickers = load_tickers()
    assert len(tickers) == 34
    assert "NYSE:LMT" in tickers
    assert all(":" in t for t in tickers)


def test_every_ticker_has_asset_class():
    from gym.config import load_tickers, asset_class_of
    for t in load_tickers():
        ac = asset_class_of(t)
        assert ac in (0, 1, 2, 3), f"{t} -> {ac}"


def test_slug_matches_source_files():
    from gym.config import slug, SIGNAL_STUDY_RESULTS, load_tickers
    for t in load_tickers():
        f = SIGNAL_STUDY_RESULTS / f"effectiveness_{slug(t)}.json"
        assert f.exists(), f"missing source file for {t}: {f.name}"
