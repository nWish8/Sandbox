"""evo_replay.py — recorded replay of an evolutionary generation (the watchable dashboard).

Three panes (spec `evo_spec.md`, layout approved):
  - world          — stepped candles + Prophets signal markers (one shared ticker stream),
  - equity race    — every agent's threaded portfolio value on a shared axis, champion bold,
                     B&H dashed, ruin line marked; ruined curves end (NaN) where they died,
  - leaderboard    — agents ranked by the selected objective (printed + on-chart legend).

Split into a headless, unit-tested data layer (`GenerationReplay` + frame builders) and a
thin finplot drawing layer (`replay_evolution`) guarded by availability — the same pattern as
`replay.py` / `monitor.py`. The data layer is what the tests cover; the drawing reuses the
finplot calls proven in `replay.py`.

A generation is recorded during `evolve(..., record_path=...)` and replayed later by
`gym.run evo-replay`, so training stays headless and watching is a separate, scrubable step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gym.stats import objective_value

_FINPLOT_AVAILABLE = False
try:
    import finplot as fplt
    _FINPLOT_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────── recorded generation

@dataclass
class GenerationReplay:
    """A compact, serialisable snapshot of one generation's population pass.

    equity/weight are (n_agents, n_bars) with NaN after an agent's ruin; bh_equity is the
    shared (n_bars,) buy-and-hold reference; ruin_bar is -1 when an agent survived.
    """
    equity: np.ndarray
    weight: np.ndarray
    bh_equity: np.ndarray
    ruin_bar: np.ndarray
    fitness: np.ndarray
    ruined: np.ndarray
    names: list[str]
    champion_idx: int
    objective: str
    ordered_slugs: list[str]
    ticker_bounds: list                       # [(ticker, start, end), ...]
    stats: list[dict]

    # ---- construction from a live population pass
    @classmethod
    def from_results(cls, results, ordered_slugs, objective, names=None,
                     champion_idx=None):
        """Build from a list of `population.PassResult` (same ordered pass)."""
        n_bars = max(len(r.ec) for r in results)
        a = len(results)
        equity = np.full((a, n_bars), np.nan)
        weight = np.full((a, n_bars), np.nan)
        for i, r in enumerate(results):
            m = len(r.ec)
            equity[i, :m] = r.ec["equity"].to_numpy(dtype=np.float64)
            weight[i, :m] = r.ec["weight"].to_numpy(dtype=np.float64)
        # B&H is policy-independent; take the longest available, ignoring ruin truncation
        bh_src = max(results, key=lambda r: len(r.ec))
        bh = np.full(n_bars, np.nan)
        bh[: len(bh_src.ec)] = bh_src.ec["bh_equity"].to_numpy(dtype=np.float64)

        fitness = np.array([
            objective_value(r.stats, objective) if not r.ruined else float("-inf")
            for r in results
        ])
        ruined = np.array([bool(r.ruined) for r in results])
        ruin_bar = np.array([r.ruin_bar if r.ruin_bar is not None else -1 for r in results])
        if names is None:
            names = [f"agent {i:02d}" for i in range(a)]
        if champion_idx is None:
            champion_idx = int(np.argmax(np.where(np.isfinite(fitness), fitness, -np.inf)))

        return cls(equity=equity, weight=weight, bh_equity=bh, ruin_bar=ruin_bar,
                   fitness=fitness, ruined=ruined, names=list(names),
                   champion_idx=int(champion_idx), objective=objective,
                   ordered_slugs=list(ordered_slugs),
                   ticker_bounds=[list(b) for b in results[0].ticker_bounds],
                   stats=[r.stats for r in results])

    # ---- persistence (matrices in .npz, metadata in sibling .json)
    def save(self, path) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), equity=self.equity, weight=self.weight,
                 bh_equity=self.bh_equity, ruin_bar=self.ruin_bar,
                 fitness=self.fitness, ruined=self.ruined)
        meta = {
            "names": self.names, "champion_idx": self.champion_idx,
            "objective": self.objective, "ordered_slugs": self.ordered_slugs,
            "ticker_bounds": self.ticker_bounds, "stats": self.stats,
        }
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "GenerationReplay":
        path = Path(path).with_suffix(".npz")
        d = np.load(str(path))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(equity=d["equity"], weight=d["weight"], bh_equity=d["bh_equity"],
                   ruin_bar=d["ruin_bar"], fitness=d["fitness"], ruined=d["ruined"],
                   names=meta["names"], champion_idx=meta["champion_idx"],
                   objective=meta["objective"], ordered_slugs=meta["ordered_slugs"],
                   ticker_bounds=[tuple(b) for b in meta["ticker_bounds"]],
                   stats=meta["stats"])

    # ---- derived views (the tested data layer)
    @property
    def n_agents(self) -> int:
        return self.equity.shape[0]

    @property
    def n_bars(self) -> int:
        return self.equity.shape[1]

    def race_frame(self) -> pd.DataFrame:
        """Wide frame for the equity-race pane: one column per agent + 'bh', indexed by bar."""
        cols = {self.names[i]: self.equity[i] for i in range(self.n_agents)}
        cols["bh"] = self.bh_equity
        return pd.DataFrame(cols)

    def leaderboard(self) -> list[dict]:
        """Agents ranked by fitness (desc); ruined ones sink to the bottom (-inf)."""
        rows = []
        for i in range(self.n_agents):
            s = self.stats[i]
            rows.append({
                "rank": 0, "agent": self.names[i], "idx": i,
                "fitness": float(self.fitness[i]), "ruined": bool(self.ruined[i]),
                "is_champion": i == self.champion_idx,
                "exposure_pct": s.get("exposure_pct"), "weight_std": s.get("weight_std"),
                "turnover": s.get("turnover"), "total_return": s.get("total_return"),
                "max_drawdown": s.get("max_drawdown"),
            })
        rows.sort(key=lambda r: (r["fitness"] if np.isfinite(r["fitness"]) else -np.inf),
                  reverse=True)
        for rank, r in enumerate(rows, start=1):
            r["rank"] = rank
        return rows


# ─────────────────────────────────────────── world frame

def build_world_frame(ordered_dfs: list[pd.DataFrame], lookback: int) -> pd.DataFrame:
    """Concatenate the bars the env actually traded, in pass order, for the world pane.

    The env trades bars [lookback-1 .. n-2] of each ticker (`SignalGymEnv._first/_last`), so
    slicing each df to that window makes the world frame length match the threaded equity
    curve exactly (T = sum_i (n_i - lookback)). Carries OHLC, signal `_fired` columns, a
    `ticker` label, and timestamp.
    """
    parts = []
    for df in ordered_dfs:
        sl = df.iloc[lookback - 1: len(df) - 1].copy()
        parts.append(sl)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def rebase_world_ohlc(world: pd.DataFrame, ticker_bounds) -> np.ndarray:
    """Rebase each ticker segment's OHLC so it starts at 1.0.

    The stitched tickers span very different price levels (~$6 grains to ~52000 index);
    on one axis the cheap ones vanish. Dividing each segment by its first open puts every
    ticker on a common ~1.0 scale (relative moves), readable together. Returns an (N, 4)
    array of [open, high, low, close]; segment boundaries become a visible reset to 1.0.
    """
    o = world[["open", "high", "low", "close"]].to_numpy(dtype=np.float64).copy()
    for _tk, s, e in ticker_bounds:
        base = o[s, 0]
        if base > 0:
            o[s:e + 1] /= base
    return o


def rebase_segments_to_mean(arr, ticker_bounds) -> np.ndarray:
    """Rebase each ticker segment to mean 1.0 (used for volume, whose scale differs wildly
    across the basket — index futures vs a small ETF). Display-only normalisation."""
    out = np.asarray(arr, dtype=np.float64).copy()
    for _tk, s, e in ticker_bounds:
        seg = out[s:e + 1]
        m = float(np.nanmean(seg)) if seg.size else 0.0
        if m and np.isfinite(m):
            out[s:e + 1] = seg / m
    return out


# ─────────────────────────────────────────── drawing (guarded)

def _logify(ax) -> None:
    """Best-effort log y-scale across finplot/pyqtgraph variants; no-op if unsupported."""
    for call in (lambda: ax.setLogMode(False, True), lambda: ax.set_yscale("log")):
        try:
            call()
            return
        except Exception:  # noqa: BLE001
            continue


def replay_evolution(gen: GenerationReplay, ordered_dfs: list[pd.DataFrame],
                     lookback: int, speed: float = 1.0, ruin_frac: float = 0.45) -> None:
    """Render the recorded generation in four x-linked panes — world candles, volume,
    RSI(14), and the population equity race — plus a printed leaderboard. Requires finplot
    (drawing layer; not render-verified in CI). Post rev-4 the world is a TradingView-style
    price chart (no Prophets signal markers — yfinance data has no signals)."""
    if not _FINPLOT_AVAILABLE:
        raise ImportError("finplot is required for evo replay.\nInstall it: pip install finplot")

    board = gen.leaderboard()
    print(f"\n=== leaderboard (objective={gen.objective}) ===")
    for r in board[:12]:
        tag = "  <= champion" if r["is_champion"] else ("  (ruined)" if r["ruined"] else "")
        ws = r["weight_std"]
        print(f"  {r['rank']:>2}  {r['agent']:<10}  fit={r['fitness']:+.3f}  "
              f"wstd={ws:.3f}" + (f"  turnover={r['turnover']:.1f}" if r['turnover'] else "")
              + tag)

    world = build_world_frame(ordered_dfs, lookback)
    T = len(world)
    # Synthetic MONOTONIC index shared by all panes. Real timestamps jump backward at every
    # ticker boundary (stitched tickers) — that breaks finplot's candle time-smoothing AND,
    # because finplot links x-axes across rows, pushes the race curves off-screen. A uniform
    # daily index keeps every pane aligned and monotonic.
    idx = pd.date_range("2000-01-01", periods=T, freq="D")

    # Four stacked, x-linked panes (top→bottom): candles / volume / RSI / equity race.
    ax_world, ax_vol, ax_rsi, ax_race = fplt.create_plot(
        "Signal Gym — evolution replay", rows=4, init_zoom_periods=T)

    # ── world pane: rebase each ticker segment to start at 1.0 so all share one axis
    has_ohlc = all(c in world.columns for c in ("open", "high", "low", "close"))
    if has_ohlc:
        o = rebase_world_ohlc(world, gen.ticker_bounds)
        ohlcv = pd.DataFrame({"open": o[:, 0], "close": o[:, 3],
                              "high": o[:, 1], "low": o[:, 2]}, index=idx)
        fplt.candlestick_ochl(ohlcv, ax=ax_world)
    else:
        fplt.plot(pd.Series(world["close"].to_numpy(dtype=np.float64), index=idx),
                  ax=ax_world, legend="price")
    _logify(ax_world)   # rebased segments still span a wide multiplicative range

    # ── volume pane: per-segment rebased (mean=1) so wildly different volumes share an axis;
    # green/red by candle direction via finplot's volume_ocv (falls back to a line).
    if has_ohlc and "volume" in world.columns:
        vol = rebase_segments_to_mean(world["volume"].to_numpy(dtype=np.float64),
                                      gen.ticker_bounds)
        vdf = pd.DataFrame({"open": o[:, 0], "close": o[:, 3], "volume": vol}, index=idx)
        try:
            fplt.volume_ocv(vdf, ax=ax_vol)
        except Exception:  # noqa: BLE001
            fplt.plot(pd.Series(vol, index=idx), ax=ax_vol, color="#7f8c8d", legend="volume")

    # ── RSI(14) pane: line + 30/70 reference bands, fixed 0–100 axis
    if "_display_rsi14" in world.columns:
        rsi = world["_display_rsi14"].to_numpy(dtype=np.float64)
        fplt.plot(pd.Series(rsi, index=idx), ax=ax_rsi, color="#8e44ad", width=1.5,
                  legend="RSI(14)")
        fplt.plot(pd.Series(np.full(T, 70.0), index=idx), ax=ax_rsi, color="#2ecc71",
                  width=1, style="--")
        fplt.plot(pd.Series(np.full(T, 30.0), index=idx), ax=ax_rsi, color="#e74c3c",
                  width=1, style="--")
        try:
            fplt.set_y_range(0.0, 100.0, ax=ax_rsi)
        except Exception:  # noqa: BLE001
            pass

    # ── equity-race pane: every agent (thin grey / ruined red), champion bold, B&H dashed
    for i in range(gen.n_agents):
        if i == gen.champion_idx:
            continue
        color = "#c0392b" if gen.ruined[i] else "#9aa0a6"
        fplt.plot(pd.Series(gen.equity[i], index=idx), ax=ax_race, color=color, width=1)
    fplt.plot(pd.Series(gen.equity[gen.champion_idx], index=idx), ax=ax_race,
              color="#2962ff", width=3, legend="champion")
    fplt.plot(pd.Series(gen.bh_equity, index=idx), ax=ax_race,
              color="orange", width=1.5, style="--", legend="B&H")
    fplt.plot(pd.Series(np.full(T, ruin_frac), index=idx), ax=ax_race,
              color="#A32D2D", width=1, style="--", legend=f"ruin {ruin_frac}")
    _logify(ax_race)

    fplt.show()
