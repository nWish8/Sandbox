"""portfolio.py — the multi-asset FinRL spine (Signal Gym v2).

This is the canonical environment for v2: a long-only, multi-asset spot **portfolio
allocation** problem built on FinRL's ``StockPortfolioEnv``. Each bar the agent emits a raw
score per asset; a softmax turns it into portfolio weights (a point on the simplex — long-only,
sums to 1, cash is just "no over-weighting"). The realised return is the weighted basket return
over the *next* bar, so the decision at bar t only ever earns the t→t+1 move (causality).

Two things FinRL's stock-portfolio env leaves out, both fixed here:

  * **Turnover cost.** FinRL stores ``transaction_cost_pct`` but never charges it. Without a
    cost, churn is free and every reward degenerates. ``PortfolioEnv`` charges
    ``cost_pct · Σ|Δweight|`` against the bar's return.
  * **A pluggable per-step reward.** FinRL hard-codes ``reward = portfolio_value``. The whole
    point of v2 is to *investigate* the reward, so the per-step reward is a callable
    (``reward_fn(env) -> float``) read from the registry in ``rewards.py``. Default = the bar's
    net log-return.

The env also drops FinRL's matplotlib/``print`` side-effects on the terminal step and records a
clean per-bar memory (net return, equal-weight benchmark return, weights, turnover, value) for
evaluation in ``stats.py``.

Data: built on FinRL-standard **daily** bars (``pipeline.prepare_data``) so annualisation is a
correct √252 and the covariance observation matches FinRL's design. A causal lookback covariance
matrix is added per bar by :func:`add_cov_features` (past returns only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv

from pipeline import FinRLConfig, prepare_data, _make_callback
from rewards import DEFAULT_REWARD, get_reward


# ─────────────────────────────────────────── causal covariance data-prep

def add_cov_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Add a per-bar causal covariance matrix (``cov_list``) to a FinRL long df.

    For each bar i (from ``lookback`` onward) the covariance is computed over the *previous*
    ``lookback`` bars of per-ticker returns — strictly past information, no leakage. Bars before
    the warm-up are dropped. The returned df is re-indexed by day ordinal (the integer index
    ``StockPortfolioEnv`` steps over), with one row per (bar, tic).
    """
    df = df.sort_values(["date", "tic"]).copy()
    df.index = df["date"].factorize()[0]
    days = df.index.unique()
    if len(days) <= lookback:
        raise ValueError(f"need > lookback ({lookback}) bars; got {len(days)}")

    rows = []
    for i in range(lookback, len(days)):
        window = df.loc[i - lookback:i - 1, :]                       # PAST bars only [i-lookback, i-1]
        prices = window.pivot_table(index="date", columns="tic", values="close")
        rets = prices.pct_change().dropna()
        covs = rets.cov().values
        rows.append({"date": df.loc[i, "date"].values[0] if hasattr(df.loc[i, "date"], "values")
                     else df.loc[i, "date"], "cov_list": covs})
    df_cov = pd.DataFrame(rows)
    out = df.merge(df_cov, on="date")
    out = out.sort_values(["date", "tic"]).reset_index(drop=True)
    out.index = out["date"].factorize()[0]
    return out


def prepare_portfolio_data(cfg: FinRLConfig, lookback: int = 60, log=print
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``pipeline.prepare_data`` + causal covariance features, ready for ``PortfolioEnv``."""
    train_df, trade_df = prepare_data(cfg, log=log)
    log(f"[portfolio] adding causal cov (lookback={lookback})")
    return add_cov_features(train_df, lookback), add_cov_features(trade_df, lookback)


# ─────────────────────────────────────────── the environment

def _reward_logret(env: "PortfolioEnv") -> float:
    """Default per-step reward: the bar's net (after-cost) log-return."""
    return math.log(max(1e-9, 1.0 + env.returns_memory[-1]))


class PortfolioEnv(StockPortfolioEnv):
    """FinRL ``StockPortfolioEnv`` + turnover cost + pluggable per-step reward + clean memory."""

    def __init__(self, df: pd.DataFrame, *, stock_dim: int, tech_indicator_list: list[str],
                 cost_pct: float = 0.001, reward_fn=_reward_logret, reward_scaling: float = 1.0,
                 initial_amount: float = 1_000_000.0, hmax: int = 100,
                 turbulence_threshold=None, lookback: int = 60, day: int = 0):
        super().__init__(
            df=df, stock_dim=stock_dim, hmax=hmax, initial_amount=initial_amount,
            transaction_cost_pct=cost_pct, reward_scaling=reward_scaling,
            state_space=stock_dim, action_space=stock_dim,
            tech_indicator_list=tech_indicator_list,
            turbulence_threshold=turbulence_threshold, lookback=lookback, day=day,
        )
        self.cost_pct = cost_pct
        self.reward_fn = reward_fn
        self.tics = sorted(df["tic"].unique().tolist())
        self._init_portfolio_memory()

    # ---- memory
    def _init_portfolio_memory(self):
        eq = np.full(self.stock_dim, 1.0 / self.stock_dim)
        self.weights_memory = [eq]
        self.returns_memory = [0.0]            # net (after-cost) portfolio return per bar
        self.bench_returns_memory = [0.0]      # equal-weight benchmark return per bar
        self.turnover_memory = [0.0]
        self.reward_state: dict = {}           # scratch for stateful rewards (e.g. diff-Sharpe)

    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        self._init_portfolio_memory()
        self.reward = 0.0
        return state, info

    # ---- step (override: cost + causal return + pluggable reward, no side-effects)
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}

        weights = self.softmax_normalization(actions)
        prev_weights = self.weights_memory[-1]
        turnover = float(np.abs(weights - prev_weights).sum())
        cost = self.cost_pct * turnover
        last_data = self.data

        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[t].values.tolist() for t in self.tech_indicator_list], axis=0,
        )

        asset_ret = (self.data.close.values / last_data.close.values) - 1.0
        gross = float(np.sum(asset_ret * weights))
        net = gross - cost                                   # charge turnover cost
        bench = float(np.mean(asset_ret))                    # equal-weight benchmark, same bars

        self.portfolio_value *= (1.0 + net)
        self.weights_memory.append(weights)
        self.returns_memory.append(net)
        self.bench_returns_memory.append(bench)
        self.turnover_memory.append(turnover)
        self.asset_memory.append(self.portfolio_value)
        self.actions_memory.append(weights)
        self.portfolio_return_memory.append(net)
        self.date_memory.append(self.data.date.unique()[0])

        self.reward = self.reward_scaling * float(self.reward_fn(self))
        return self.state, self.reward, self.terminal, False, {}

    # ---- evaluation export
    def portfolio_history(self) -> pd.DataFrame:
        """Per-bar record for stats/backtest/replay: date, ret (net), bench_ret, value,
        turnover, plus one ``w_<tic>`` column per asset (the weights held that bar)."""
        hist = pd.DataFrame({
            "date": self.date_memory,
            "ret": self.returns_memory,
            "bench_ret": self.bench_returns_memory,
            "value": self.asset_memory,
            "turnover": self.turnover_memory,
        })
        weights = np.vstack(self.weights_memory)
        for i, tic in enumerate(self.tics):
            hist[f"w_{tic}"] = weights[:, i]
        return hist


# ─────────────────────────────────────────── env factory

def make_portfolio_env(df: pd.DataFrame, cfg: FinRLConfig, *, reward=DEFAULT_REWARD,
                       lookback: int = 60, turbulence_threshold=None) -> PortfolioEnv:
    """Build a ``PortfolioEnv`` from a cov-augmented FinRL long df + a config.

    ``reward`` may be a registry name (see ``rewards.REWARDS``) or a callable ``fn(env)->float``.
    """
    reward_fn = reward if callable(reward) else get_reward(reward)
    stock_dim = df.tic.nunique()
    tech = [c for c in cfg.indicators if c in df.columns]
    return PortfolioEnv(
        df, stock_dim=stock_dim, tech_indicator_list=tech, cost_pct=cfg.cost_pct,
        reward_fn=reward_fn, initial_amount=cfg.initial_amount, hmax=cfg.hmax,
        lookback=lookback, turbulence_threshold=turbulence_threshold,
    )


# ─────────────────────────────────────────── train + deterministic rollout

PORTFOLIO_ALGOS = ("ppo", "sac", "a2c")


def train_portfolio(train_df: pd.DataFrame, cfg: FinRLConfig, *, reward=DEFAULT_REWARD,
                    algo: str = "ppo", timesteps: int = 20_000, seed: int = 42,
                    lookback: int = 60, device: str = "cpu", progress_cb=None, stop=None,
                    recorder=None, log=print):
    """Train an SB3 policy (PPO / SAC / A2C) on the portfolio env under a given reward.

    ``recorder`` (see ``runlog.RunRecorder``) snapshots deterministic rollouts at
    checkpoints during training so the run can be replayed bar-by-bar later.
    """
    import stable_baselines3 as sb3
    from stable_baselines3.common.vec_env import DummyVecEnv

    if algo not in PORTFOLIO_ALGOS:
        raise ValueError(f"algo must be one of {PORTFOLIO_ALGOS}, got {algo!r}")
    cls = getattr(sb3, algo.upper())
    venv = DummyVecEnv([lambda: make_portfolio_env(train_df, cfg, reward=reward, lookback=lookback)])
    model = cls("MlpPolicy", venv, seed=seed, device=device, verbose=0)
    log(f"[portfolio] train algo={algo} reward={reward!s} steps={timesteps} "
        f"device={device} seed={seed}")
    callbacks = [_make_callback(timesteps, progress_cb, stop)]
    if recorder is not None:
        callbacks.append(recorder.sb3_callback())
    model.learn(total_timesteps=timesteps, callback=callbacks)
    if recorder is not None:
        recorder.finalize(model)
    return model


def run_portfolio(model, df: pd.DataFrame, cfg: FinRLConfig, *, reward=DEFAULT_REWARD,
                  lookback: int = 60, turbulence_threshold=None) -> pd.DataFrame:
    """Roll a trained model deterministically over ``df``; return its per-bar history."""
    env = make_portfolio_env(df, cfg, reward=reward, lookback=lookback,
                             turbulence_threshold=turbulence_threshold)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    return env.portfolio_history()
