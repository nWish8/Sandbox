"""pipeline.py — a clean, reusable wrapper around FinRL's DRL flow.

This is the backend the PyQt control panel drives. It exposes a single config object and
four steps mirroring FinRL's example, but parameterised, cached, GPU/parallel-aware, and
streamable (progress + stop) so a GUI can configure, launch, watch, and cancel runs.

    cfg = FinRLConfig(...)
    train_df, trade_df = prepare_data(cfg)        # download + features + split (cached)
    model = train(cfg, progress_cb=..., stop=...) # build env + agent + learn
    acct, actions = backtest(cfg, model, trade_df)

Honest note on `n_envs`: FinRL's StockTradingEnv walks one deterministic trajectory, so
parallel copies mainly add rollout volume rather than wall-clock speed; it's wired for
generality. Device cpu/cuda is selectable — for this small single-env net CPU is usually
faster (see finrl_lab/requirements-lock.txt run notes).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

from marketdata import MarketData

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "trained_models"
AGENTS = ("a2c", "ddpg", "ppo", "td3", "sac")


# ─────────────────────────────────────────── config

@dataclass
class FinRLConfig:
    tickers: list[str]
    train_start: str = "2014-01-01"
    train_end: str = "2022-01-01"
    trade_start: str = "2022-01-01"
    trade_end: str = "2024-01-01"
    indicators: tuple = tuple(INDICATORS)
    use_vix: bool = True
    use_turbulence: bool = True

    agent: str = "ppo"
    timesteps: int = 20_000
    model_kwargs: dict = field(default_factory=dict)
    device: str = "cpu"            # "cpu" | "cuda"
    n_envs: int = 1

    hmax: int = 100
    initial_amount: float = 1_000_000.0
    cost_pct: float = 0.001
    reward_scaling: float = 1e-4
    turbulence_threshold: float = 70.0
    seed: int = 42

    def __post_init__(self):
        if self.agent not in AGENTS:
            raise ValueError(f"agent must be one of {AGENTS}, got {self.agent!r}")

    # cache key over the data-affecting fields only
    def _data_key(self) -> str:
        d = {k: getattr(self, k) for k in
             ("tickers", "train_start", "train_end", "trade_start", "trade_end",
              "indicators", "use_vix", "use_turbulence")}
        return hashlib.md5(json.dumps(d, default=list, sort_keys=True).encode()).hexdigest()[:10]


# ─────────────────────────────────────────── data

def prepare_data(cfg: FinRLConfig, log=print, force: bool = False
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download + FeatureEngineer + split, cached by data-config hash."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    key = cfg._data_key()
    ftrain = DATA_DIR / f"train_{key}.parquet"
    ftrade = DATA_DIR / f"trade_{key}.parquet"
    if ftrain.exists() and ftrade.exists() and not force:
        log(f"[data] cache hit {key}")
        return pd.read_parquet(ftrain), pd.read_parquet(ftrade)

    log(f"[data] fetching {len(cfg.tickers)} tickers {cfg.train_start}..{cfg.trade_end}")
    md = MarketData()
    raw = md.get_finrl(list(cfg.tickers), cfg.train_start, cfg.trade_end, log=log)
    # use_vix=False: FinRL's add_vix calls its broken YahooDownloader; we merge the same
    # 'vix' column from our own cached source below instead.
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=list(cfg.indicators),
                         use_vix=False, use_turbulence=cfg.use_turbulence,
                         user_defined_feature=False)
    processed = fe.preprocess_data(raw)
    if cfg.use_vix:
        vix = (md.get_finrl(["^VIX"], cfg.train_start, cfg.trade_end)[["date", "close"]]
               .rename(columns={"close": "vix"}))
        processed = processed.merge(vix, on="date")   # inner join, matching FinRL's add_vix
    # dense (date x tic) grid, fill gaps
    import itertools
    tics = processed["tic"].unique().tolist()
    dates = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
    full = (pd.DataFrame(list(itertools.product(dates, tics)), columns=["date", "tic"])
            .merge(processed, on=["date", "tic"], how="left"))
    full = full[full["date"].isin(processed["date"])].sort_values(["date", "tic"]).fillna(0)

    train_df = data_split(full, cfg.train_start, cfg.train_end)
    trade_df = data_split(full, cfg.trade_start, cfg.trade_end)
    train_df.to_parquet(ftrain); trade_df.to_parquet(ftrade)
    log(f"[data] train={len(train_df)} trade={len(trade_df)} cached as {key}")
    return train_df, trade_df


# ─────────────────────────────────────────── env + agent

def _env_kwargs(df: pd.DataFrame, cfg: FinRLConfig) -> dict:
    stock_dim = len(df.tic.unique())
    state_space = 1 + 2 * stock_dim + len(cfg.indicators) * stock_dim
    return {
        "hmax": cfg.hmax,
        "initial_amount": cfg.initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [cfg.cost_pct] * stock_dim,
        "sell_cost_pct": [cfg.cost_pct] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": list(cfg.indicators),
        "action_space": stock_dim,
        "reward_scaling": cfg.reward_scaling,
    }


def make_train_env(train_df: pd.DataFrame, cfg: FinRLConfig):
    """Return an SB3 vec env. n_envs>1 builds a DummyVecEnv of copies (in-process)."""
    kw = _env_kwargs(train_df, cfg)
    if cfg.n_envs <= 1:
        env, _ = StockTradingEnv(df=train_df, **kw).get_sb_env()
        return env
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv([(lambda: StockTradingEnv(df=train_df, **kw)) for _ in range(cfg.n_envs)])


def build_agent(vec_env, cfg: FinRLConfig):
    agent = DRLAgent(env=vec_env)
    mk = {**cfg.model_kwargs, "device": cfg.device}   # FinRL injects seed itself
    return agent, agent.get_model(cfg.agent, model_kwargs=mk, seed=cfg.seed)


# ─────────────────────────────────────────── progress callback

def _make_callback(total: int, progress_cb, stop):
    from stable_baselines3.common.callbacks import BaseCallback

    class _CB(BaseCallback):
        def _on_step(self) -> bool:
            if stop is not None and stop():
                return False                       # cancel training
            if progress_cb is not None and self.num_timesteps % 1000 < self.training_env.num_envs:
                rew = None
                buf = getattr(self.model, "ep_info_buffer", None)
                if buf:
                    import numpy as np
                    rew = float(np.mean([e["r"] for e in buf]))
                progress_cb(self.num_timesteps, total, rew)
            return True

    return _CB()


# ─────────────────────────────────────────── train + backtest

def train(cfg: FinRLConfig, train_df: pd.DataFrame | None = None,
          progress_cb=None, stop=None, log=print):
    """Build env+agent and train. progress_cb(steps,total,reward); stop()->bool cancels."""
    if train_df is None:
        train_df, _ = prepare_data(cfg, log=log)
    env = make_train_env(train_df, cfg)
    agent, model = build_agent(env, cfg)
    dev = next(model.policy.parameters()).device
    log(f"[train] agent={cfg.agent} device={dev} n_envs={cfg.n_envs} steps={cfg.timesteps}")
    t0 = time.time()
    model.learn(total_timesteps=cfg.timesteps, callback=_make_callback(cfg.timesteps, progress_cb, stop),
                tb_log_name=cfg.agent)
    log(f"[train] done in {time.time()-t0:.1f}s")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR / f"agent_{cfg.agent}"))
    return model


def backtest(cfg: FinRLConfig, model, trade_df: pd.DataFrame | None = None, log=print):
    """Run the trained model over the trade split → (account_value_df, actions_df)."""
    if trade_df is None:
        _, trade_df = prepare_data(cfg, log=log)
    kw = _env_kwargs(trade_df, cfg)
    e_trade = StockTradingEnv(df=trade_df, turbulence_threshold=cfg.turbulence_threshold,
                              risk_indicator_col="vix", **kw)
    acct, actions = DRLAgent.DRL_prediction(model=model, environment=e_trade)
    if len(acct):
        i, f = float(acct["account_value"].iloc[0]), float(acct["account_value"].iloc[-1])
        log(f"[backtest] {len(acct)} steps  return={100*(f/i-1):+.2f}%")
    return acct, actions


if __name__ == "__main__":
    # tiny smoke test (few tickers, short, few steps)
    cfg = FinRLConfig(tickers=["AAPL", "MSFT", "JPM"], train_start="2018-01-01",
                      train_end="2021-01-01", trade_start="2021-01-01",
                      trade_end="2022-01-01", agent="ppo", timesteps=3000, device="cpu")
    tr, td = prepare_data(cfg)
    m = train(cfg, tr, progress_cb=lambda s, t, r: print(f"  {s}/{t} reward={r}"))
    backtest(cfg, m, td)
    print("pipeline smoke OK")
