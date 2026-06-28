"""Central configuration for Signal Gym.

One ``Config`` dataclass holds every tunable default so runs are reproducible from a
single object. Paths are derived from this file's location so the package works
regardless of the caller's working directory.

Data layer (rev 4 — FinRL migration): OHLCV is downloaded from Yahoo (yfinance) on a
**1-hour** clock in the FinRL standard long format (``date, tic, open, high, low,
close, volume`` + technical indicators). The Prophets/TradingView signal suite and the
PPO/supervised agents were removed in this revision; the evolutionary population race is
the primary experience.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------- paths
GYM_DIR = Path(__file__).resolve().parent          # .../Sandbox/gym
ROOT = GYM_DIR.parent                              # .../Sandbox
GYM_DATA = GYM_DIR / "data"
RAW_DATA = GYM_DATA / "raw"                        # yfinance raw OHLCV parquets (FinRL long)
GYM_MODELS = GYM_DIR / "models"
GYM_REPORTS = GYM_DIR / "reports"
TICKERS_FILE = GYM_DIR / "config" / "tickers.txt"
SPLITS_FILE = GYM_DATA / "splits.json"

# Data acquisition (yfinance). 1h is the clock; Yahoo serves ~730 days of 1h bars.
INTERVAL = "1h"
PERIOD = "730d"
PRIMARY_TF = "1h"

# FinRL-standard technical indicators (the agent's observation columns). These are the
# only model inputs — purely price-derived, no Prophets signals (rev 4).
TECH_INDICATORS = [
    "ret_1", "ret_3", "ret_6",       # log-return lookbacks
    "vol_20",                         # trailing realised volatility
    "range_pos_20",                   # position within the 20-bar range
    "drawdown_20",                    # drawdown from the 20-bar high
    "atr_14",                         # average true range (close-normalised)
]

# Display-only columns: written to the processed parquet for charting but NEVER fed to
# the agent (excluded from feature_columns). Prefix convention: ``_display_``.
DISPLAY_COLS = ["_display_rsi14"]

# Asset-class buckets: 0=index, 1=equity, 2=commodity, 3=ETF. Only generalisable
# context for the stratified split — never a model feature.
ASSET_CLASS = {"index": 0, "equity": 1, "commodity": 2, "etf": 3}

# Per-ticker asset-class assignment, keyed on the Yahoo symbol (exact, upper-cased).
TICKER_ASSET_CLASS: dict[str, int] = {
    # indices / index futures
    "ES=F": 0, "^GSPC": 0, "^NDX": 0, "^DJI": 0, "^RUT": 0, "^TA125.TA": 0,
    # defence / aerospace / industrial equities
    "LMT": 1, "NOC": 1, "RTX": 1, "GD": 1, "LHX": 1, "LDOS": 1, "BA": 1, "PLTR": 1,
    # agriculture / fertiliser / machinery equities
    "NTR": 1, "MOS": 1, "CF": 1, "DE": 1, "ADM": 1, "BG": 1,
    # energy equities
    "CVX": 1, "XOM": 1, "OXY": 1,
    # commodity futures
    "CL=F": 2, "BZ=F": 2,
    # ETFs
    "ITA": 3, "XAR": 3, "PPA": 3, "DFND.L": 3, "ARMY.L": 3,
    "WEAT": 3, "CORN": 3, "SOYB": 3,
}


@dataclass
class Config:
    """All Signal Gym defaults. Pass an instance through the pipeline; serialise into
    meta.json for reproducibility."""
    # paths (overridable for tests)
    gym_data: Path = GYM_DATA
    raw_data: Path = RAW_DATA
    gym_models: Path = GYM_MODELS
    gym_reports: Path = GYM_REPORTS
    tickers_file: Path = TICKERS_FILE
    splits_file: Path = SPLITS_FILE

    # data / collection
    interval: str = INTERVAL
    period: str = PERIOD
    primary_tf: str = PRIMARY_TF
    tech_indicators: tuple = tuple(TECH_INDICATORS)

    # feature construction
    lookback: int = 30             # W — observation window length (now 30 one-hour bars)
    vol_window: int = 20           # trailing volatility / range / drawdown window
    atr_window: int = 14
    rsi_window: int = 14           # display RSI (chart only, not a feature)

    # environment / execution
    commission_bps: float = 10.0   # round-trip cost per unit weight change (basis pts)
    reward_decay: float = 0.02     # differential-Sharpe EMA decay (eta)
    reward_eps: float = 1e-8       # denominator floor for differential Sharpe
    continuous_action: bool = True  # False -> discrete {0,.25,.5,.75,1}
    # rebalance deadband: ignore weight changes smaller than this (don't trade on noise).
    # Essential on the 1h clock — without it a continuous sizer reweights every hour and
    # realistic costs ruin every agent over a ~130k-bar pass. Lets deliberate timing show.
    rebalance_threshold: float = 0.10
    # reward_mode selects the per-step env reward (the evolutionary objective is separate,
    # in stats.OBJECTIVES). "excess" = diff-Sharpe of return minus buy-and-hold;
    # "absolute" = diff-Sharpe of own return; "absolute_dd" subtracts drawdown_penalty*dd.
    reward_mode: str = "excess"
    drawdown_penalty: float = 0.0

    # splits
    split_ratios: tuple = (0.70, 0.15, 0.15)   # training / validation / test (tickers)
    stratify_col: str = "asset_class"
    seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


DEFAULT_CONFIG = Config()


def asset_class_of(symbol: str) -> int | None:
    """Map a Yahoo symbol to its asset-class ordinal, or None if unknown."""
    return TICKER_ASSET_CLASS.get(symbol.upper())


def load_tickers(path: Path = TICKERS_FILE) -> list[str]:
    """Read the ticker manifest, returning Yahoo symbols (comments/blanks stripped)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def slug(symbol: str) -> str:
    """Yahoo symbol -> filesystem-safe slug for parquet filenames.

    ``ES=F`` -> ``ES_F``, ``^GSPC`` -> ``GSPC``, ``^TA125.TA`` -> ``TA125_TA``,
    ``DFND.L`` -> ``DFND_L``. Deterministic and collision-free across the basket.
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", symbol).strip("_").upper()
