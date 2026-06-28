"""env.py — the Signal Gym environment: one ticker, long-only spot, bar-by-bar.

A single instance plays through one ticker (an episode). The agent observes a live
window of features ending at the current bar, chooses a target weight in [0, 1], and the
position is rebalanced at the next bar's fill price. Reward is the differential Sharpe of
excess return over buy-and-hold, minus turnover cost (spec §5.4). The same stepper drives
the finplot replay.

Causality (the headline invariant): the observation at bar t contains only feature rows
<= t; the weight chosen at bar t is realised over the t -> t+1 interval using bar t+1's
price. The agent can never see a price it then trades on.

Tasks: T2.5 (mechanics), T2.6 (reward), T2.7 (causality tests).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from gym.config import DEFAULT_CONFIG, Config
from gym.features import feature_columns

DISCRETE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


class SignalGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ticker_df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG,
                 lookback: int | None = None, commission_bps: float | None = None,
                 reward_decay: float | None = None, continuous: bool | None = None,
                 obs_stats: tuple[np.ndarray, np.ndarray] | None = None):
        super().__init__()
        self.cfg = cfg
        self.lookback = lookback if lookback is not None else cfg.lookback
        self.commission = (commission_bps if commission_bps is not None
                           else cfg.commission_bps) / 1e4
        self.reward_decay = reward_decay if reward_decay is not None else cfg.reward_decay
        self.continuous = continuous if continuous is not None else cfg.continuous_action
        self.rebalance_threshold = float(getattr(cfg, "rebalance_threshold", 0.0))

        self.feat_cols = feature_columns(ticker_df, cfg)
        self.X = ticker_df[self.feat_cols].to_numpy(dtype=np.float32)
        # optional observation normalization (neural/PPO path only): standardize each
        # feature with TRAIN-fit (mean, std) so the MLP isn't dominated by raw-scale
        # features. The GBT and rule baselines read X RAW and never pass obs_stats.
        if obs_stats is not None:
            mean, std = obs_stats
            self.X = ((self.X - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)
        self.close = ticker_df["close"].to_numpy(dtype=np.float64)
        self.open_ = ticker_df["open"].to_numpy(dtype=np.float64)
        self.timestamp = ticker_df["timestamp"].to_numpy()
        self.n = len(ticker_df)
        if self.n <= self.lookback + 1:
            raise ValueError(f"ticker has too few bars ({self.n}) for lookback {self.lookback}")

        n_feat = self.X.shape[1]
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.lookback, n_feat), np.float32)
        if self.continuous:
            self.action_space = spaces.Box(0.0, 1.0, (1,), np.float32)
        else:
            self.action_space = spaces.Discrete(len(DISCRETE_WEIGHTS))

        self._first = self.lookback - 1          # first decision bar (full window exists)
        self._last = self.n - 2                  # last bar with a next-bar fill
        self.history: list[dict] = []

    # ----------------------------------------------------------------- helpers
    def _weight(self, action) -> float:
        if self.continuous:
            a = float(np.asarray(action).reshape(-1)[0])
        else:
            a = DISCRETE_WEIGHTS[int(action)]
        return float(np.clip(a, 0.0, 1.0))

    def _obs(self) -> np.ndarray:
        # copy so a consumer that retains the obs (e.g. SB3 buffers) is never aliased to
        # X, and so the causality guarantee holds even if X is later mutated.
        return self.X[self.i - self.lookback + 1: self.i + 1].copy()

    # -------------------------------------------------------------------- api
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = self._first
        self.weight = 0.0
        self.equity = 1.0
        self.bh_equity = 1.0
        self._peak_equity = 1.0    # running high-water mark for drawdown
        self._A = 0.0          # EMA of reward-signal return (differential-Sharpe state)
        self._B = 0.0          # EMA of reward-signal return^2
        self._steps = 0
        self.history = []
        return self._obs(), {"bar_idx": self.i}

    def _diff_sharpe(self, x: float) -> float:
        """Moeller differential Sharpe: marginal contribution of excess return x to the
        running Sharpe of excess returns. 0 during warmup; floored denominator."""
        if self._steps < self.lookback:
            self._A += self.reward_decay * (x - self._A)
            self._B += self.reward_decay * (x * x - self._B)
            return 0.0
        dA = x - self._A
        dB = x * x - self._B
        var = self._B - self._A * self._A
        denom = (var + self.cfg.reward_eps) ** 1.5
        d = (self._B * dA - 0.5 * self._A * dB) / denom
        self._A += self.reward_decay * dA
        self._B += self.reward_decay * dB
        return float(d)

    def step(self, action):
        w_prev = self.weight
        w = self._weight(action)
        # rebalance deadband: hold unless the target moved enough to be worth the cost
        if self.rebalance_threshold > 0.0 and abs(w - w_prev) < self.rebalance_threshold:
            w = w_prev
        i = self.i
        c_t, o_n, c_n = self.close[i], self.open_[i + 1], self.close[i + 1]

        # two-term fill accounting (spec §5.4): gap at old weight to the fill price, then
        # the rest of the bar at the new weight, minus turnover cost on the rebalance.
        gap = w_prev * (o_n / c_t - 1.0)
        intra = w * (c_n / o_n - 1.0)
        turn = self.commission * abs(w - w_prev)
        r = gap + intra - turn
        bh_r = c_n / c_t - 1.0
        excess = r - bh_r

        self.equity *= (1.0 + r)
        self.bh_equity *= (1.0 + bh_r)
        self._peak_equity = max(self._peak_equity, self.equity)
        drawdown = self.equity / self._peak_equity - 1.0   # <= 0

        # reward signal depends on the configured mode (eval metrics are independent of it)
        mode = getattr(self.cfg, "reward_mode", "excess")
        if mode == "excess":
            reward = self._diff_sharpe(excess)
        else:
            reward = self._diff_sharpe(r)                  # agent's OWN risk-adjusted return
            if mode == "absolute_dd":
                reward += self.cfg.drawdown_penalty * drawdown

        trade = None
        if w != w_prev:
            trade = {"bar_idx": i, "from": w_prev, "to": w, "price": o_n}
        self.weight = w
        self._steps += 1

        info = {
            "bar_idx": i, "weight": w, "position": w, "equity": self.equity,
            "bh_equity": self.bh_equity, "ret": r, "bh_ret": bh_r, "excess": excess,
            "drawdown": drawdown, "reward": reward, "trade": trade,
            "timestamp": int(self.timestamp[i]),
        }
        self.history.append(info)

        self.i += 1
        terminated = self.i > self._last
        if terminated:
            obs = self.X[self._last - self.lookback + 1: self._last + 1]
        else:
            obs = self._obs()
        return obs, reward, terminated, False, info

    def get_equity_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
