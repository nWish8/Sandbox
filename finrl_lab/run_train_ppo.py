"""Proof run: train PPO on FinRL's StockTradingEnv, comparing CPU vs GPU, then backtest.

Mirrors FinRL_StockTrading_2026_2_train.py but PPO-only + reduced timesteps, and times
CPU vs GPU so we have real numbers for whether the RTX 4050 helps this workload.
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import torch

import finrl_patch  # noqa: F401  (fixes yfinance downloader)
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

STEPS = 15000

train = pd.read_csv("train_data.csv")
train = train.set_index(train.columns[0])
train.index.names = [""]

stock_dim = len(train.tic.unique())
state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1_000_000,
    "num_stock_shares": [0] * stock_dim,
    "buy_cost_pct": [0.001] * stock_dim,
    "sell_cost_pct": [0.001] * stock_dim,
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dim,
    "reward_scaling": 1e-4,
}
print(f"stock_dim={stock_dim}  state_space={state_space}  indicators={len(INDICATORS)}")

PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}


def train_on(device: str) -> float:
    env_train, _ = StockTradingEnv(df=train, **env_kwargs).get_sb_env()
    agent = DRLAgent(env=env_train)
    model = agent.get_model("ppo", model_kwargs={**PPO_PARAMS, "device": device})
    dev = next(model.policy.parameters()).device
    t0 = time.time()
    agent.train_model(model=model, tb_log_name=f"ppo_{device}", total_timesteps=STEPS)
    dt = time.time() - t0
    print(f"  device={dev}  {STEPS} steps in {dt:.1f}s  ({STEPS/dt:.0f} steps/s)")
    return dt, model


print(f"\n=== PPO timing, {STEPS} steps ===")
print("GPU:")
gpu_dt, gpu_model = train_on("cuda")
print("CPU:")
cpu_dt, _ = train_on("cpu")
print(f"\nGPU vs CPU: {cpu_dt/gpu_dt:.2f}x  (>1 means GPU faster)")

# quick backtest of the GPU model on the trade split
trade = pd.read_csv("trade_data.csv")
trade = trade.set_index(trade.columns[0]); trade.index.names = [""]
e_trade = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)
df_account, df_actions = DRLAgent.DRL_prediction(model=gpu_model, environment=e_trade)
final = float(df_account["account_value"].iloc[-1])
init = float(df_account["account_value"].iloc[0])
print(f"\nBacktest (trade split): start={init:,.0f}  end={final:,.0f}  "
      f"return={100*(final/init-1):+.2f}%  ({len(df_account)} steps)")
gpu_model.save("trained_models/agent_ppo")
print("saved trained_models/agent_ppo")
