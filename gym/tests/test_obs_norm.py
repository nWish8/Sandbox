"""T5.4 — observation normalization (PPO path): train-only fit, env standardizes."""
from __future__ import annotations

import numpy as np

from gym import features as F
from gym.agent_rl import fit_obs_stats
from gym.env import SignalGymEnv

DAY = 86_400


def _frame(write_fixture, ticker="TEST:ON1", n=60, seed=0):
    cfg, write = write_fixture
    rng = np.random.default_rng(seed)
    c = list(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
    write(ticker, [1_700_000_000 + i * DAY for i in range(n)], c)
    vocab = F.build_vocab([ticker], cfg)
    df = F.build_ticker_frame(ticker, vocab, cfg)
    feat = F.feature_columns(df, cfg)
    df[feat] = df[feat].fillna(0.0)
    return cfg, df


def test_fit_obs_stats_shapes(write_fixture):
    cfg, df = _frame(write_fixture)
    mean, std = fit_obs_stats([df], cfg)
    n_feat = len(F.feature_columns(df, cfg))
    assert mean.shape == (n_feat,)
    assert std.shape == (n_feat,)
    assert (std > 0).all()                      # zero std floored to 1


def test_obs_stats_standardizes_env_observations(write_fixture):
    cfg, df = _frame(write_fixture, n=80)
    mean, std = fit_obs_stats([df], cfg)

    raw_env = SignalGymEnv(df, cfg, lookback=5)
    norm_env = SignalGymEnv(df, cfg, lookback=5, obs_stats=(mean, std))

    raw_obs, _ = raw_env.reset()
    norm_obs, _ = norm_env.reset()
    # normalized obs must differ from raw and be on a tighter scale
    assert not np.allclose(raw_obs, norm_obs)
    # pooled normalized features over the whole frame are ~zero-mean / unit-scale
    full = (df[norm_env.feat_cols].to_numpy(np.float32) - mean) / std
    assert abs(full.mean()) < 0.2
    assert full.std() < 5.0


def test_obs_stats_does_not_change_equity(write_fixture):
    """Normalization changes observations only — fills/rewards use raw prices, so the
    equity path for a fixed action sequence is identical with or without obs_stats."""
    cfg, df = _frame(write_fixture, n=80)
    mean, std = fit_obs_stats([df], cfg)

    class Fixed:
        def act(self, obs):  # ignores obs entirely
            return 1.0

    from gym.backtest import run_backtest
    raw = run_backtest(SignalGymEnv(df, cfg, lookback=5), Fixed())
    nrm = run_backtest(SignalGymEnv(df, cfg, lookback=5, obs_stats=(mean, std)), Fixed())
    np.testing.assert_allclose(raw.equity_curve["equity"].to_numpy(),
                               nrm.equity_curve["equity"].to_numpy(), rtol=1e-9)


def test_fit_obs_stats_train_only(write_fixture):
    """A held-out frame with extreme features cannot shift the fitted stats."""
    cfg, d1 = _frame(write_fixture, ticker="TEST:TR1", n=60, seed=1)
    _, d2 = _frame(write_fixture, ticker="TEST:TR2", n=60, seed=2)
    _, dx = _frame(write_fixture, ticker="TEST:OUT", n=60, seed=3)
    feat = F.feature_columns(d1, cfg)
    dx[feat] = dx[feat] + 1000.0                # poison the held-out frame

    m_train, _ = fit_obs_stats([d1, d2], cfg)
    m_all, _ = fit_obs_stats([d1, d2, dx], cfg)
    assert not np.allclose(m_train, m_all)      # including poison shifts the mean
    assert (np.abs(m_train) < 100).all()        # train-only mean stays sane
