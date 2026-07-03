"""run_replay_panel.playback_frame — pure slicing logic on a synthetic RunRecord.
(The Qt widget itself is verified by offscreen render, not unit-tested here.)"""
from __future__ import annotations

import json

import numpy as np
import pytest

from runlog import RunRecord


@pytest.fixture
def rec(tmp_path):
    """Hand-crafted consistent recording: 3 checkpoints × 5 bars × 2 assets."""
    C, T, N = 3, 5, 2
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, (C, T))
    returns[:, 0] = 0.0
    equity = np.cumprod(1.0 + returns, axis=1)
    weights = rng.dirichlet(np.ones(N), size=(C, T))
    bench = rng.normal(0.001, 0.01, T)
    bench[0] = 0.0
    d = tmp_path / "run1"
    d.mkdir()
    np.savez_compressed(d / "record.npz",
                        steps=np.array([0, 100, 200]),
                        dates=np.array([f"2024-01-0{i+1}" for i in range(T)], dtype="U10"),
                        bench_ret=bench, equity=equity, returns=returns,
                        turnover=np.abs(rng.normal(0, 0.1, (C, T))), weights=weights)
    (d / "manifest.json").write_text(json.dumps(
        {"run_id": "run1", "algo": "ppo", "reward": "logret", "tics": ["AAA", "BBB"],
         "seed": 1, "timesteps": 200, "every": 100, "lookback": 3, "n_checkpoints": C}),
        encoding="utf-8")
    return RunRecord(d)


def test_frame_reveals_up_to_cursor(rec):
    from run_replay_panel import playback_frame
    f = playback_frame(rec, ci=1, t=2)
    assert f["equity"].shape == (3,)                 # bars 0..2 inclusive
    assert f["bench"].shape == (3,)
    assert f["weights"].shape == (3, 2)
    assert f["step"] == 100 and f["date"] == "2024-01-03"
    np.testing.assert_allclose(f["equity"], rec.equity[1, :3])


def test_frame_clamps_out_of_range(rec):
    from run_replay_panel import playback_frame
    f = playback_frame(rec, ci=99, t=-5)
    assert f["ci"] == rec.n_checkpoints - 1 and f["t"] == 0
    assert f["equity"].shape == (1,)


def test_skill_curve_is_final_equity_per_checkpoint(rec):
    from run_replay_panel import playback_frame
    f = playback_frame(rec, 0, 0)
    np.testing.assert_allclose(f["skill"], rec.equity[:, -1])
    assert f["final_equity"] == pytest.approx(rec.equity[0, -1])
