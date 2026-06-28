"""T5.1 — agent_rl: RLPolicy wraps PPO; smoke test on planted-edge synthetic ticker."""
from __future__ import annotations

import numpy as np
import pytest


# ─── RLPolicy

def test_rl_policy_act_in_unit_interval(write_fixture):
    """After minimal training, RLPolicy.act returns a float in [0,1]."""
    from stable_baselines3 import PPO

    from gym import features as F
    from gym.agent_rl import RLPolicy
    from gym.env import SignalGymEnv

    cfg, write = write_fixture
    rng = np.random.default_rng(77)
    n = 60
    c = list(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
    ts = [1_700_000_000 + i * 86_400 for i in range(n)]
    write("TEST:RL1", ts, c)

    vocab = F.build_vocab(["TEST:RL1"], cfg)
    df = F.build_ticker_frame("TEST:RL1", vocab, cfg)
    feat = F.feature_columns(df, cfg)
    df[feat] = df[feat].fillna(0.0)

    env = SignalGymEnv(df, cfg, lookback=5)
    model = PPO("MlpPolicy", env, n_steps=32, verbose=0, seed=0)
    model.learn(total_timesteps=100)

    policy = RLPolicy(model)
    obs, _ = env.reset()
    for _ in range(5):
        w = policy.act(obs)
        assert 0.0 <= w <= 1.0, f"weight {w} out of [0,1]"
        obs, _, done, _, _ = env.step(np.array([w], dtype=np.float32))
        if done:
            break


def test_rl_policy_reset_does_not_raise(write_fixture):
    from stable_baselines3 import PPO
    from gym import features as F
    from gym.agent_rl import RLPolicy
    from gym.env import SignalGymEnv

    cfg, write = write_fixture
    rng = np.random.default_rng(88)
    n = 50
    c = list(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
    write("TEST:RL2", [1_700_000_000 + i * 86_400 for i in range(n)], c)
    vocab = F.build_vocab(["TEST:RL2"], cfg)
    df = F.build_ticker_frame("TEST:RL2", vocab, cfg)
    df[F.feature_columns(df, cfg)] = df[F.feature_columns(df, cfg)].fillna(0.0)

    env = SignalGymEnv(df, cfg, lookback=5)
    model = PPO("MlpPolicy", env, n_steps=32, verbose=0, seed=0)
    model.learn(total_timesteps=64)
    policy = RLPolicy(model)
    policy.reset()   # must not raise


# ─── make_vec_env

def test_make_vec_env_builds_env(write_fixture):
    from gym import features as F
    from gym.agent_rl import make_vec_env
    from stable_baselines3.common.vec_env import VecEnv

    cfg, write = write_fixture
    dfs = []
    for i in range(2):
        rng = np.random.default_rng(i)
        n = 60
        c = list(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
        ts = [1_700_000_000 + j * 86_400 for j in range(n)]
        write(f"TEST:VEC{i}", ts, c)
        vocab = F.build_vocab([f"TEST:VEC{i}"], cfg)
        df = F.build_ticker_frame(f"TEST:VEC{i}", vocab, cfg)
        df[F.feature_columns(df, cfg)] = df[F.feature_columns(df, cfg)].fillna(0.0)
        dfs.append(df)

    vec = make_vec_env(dfs, cfg)
    assert isinstance(vec, VecEnv)
    obs = vec.reset()
    assert obs.shape[0] == 2            # 2 envs
