"""projections.py — ML return projections with confidence bands (Vision module 2, M2).

Gradient-boosted **quantile** regression: three models (q10 / q50 / q90) fit on causal
per-bar features (the screener's indicators + short return lags) pooled across the basket,
predicting the **forward h-day return**. The deliverable per ticker is a band, not a point:

    tic    close    q10%    q50%    q90%      (h-day forward return quantiles)

Honesty is built in: before the live projection, a chronological walk-forward split fits
on the earlier 70% of bars and measures **band coverage** on the held-out 30% — the
fraction of actual h-day returns that landed inside [q10, q90]. Nominal is 0.80; a
coverage far below that means the bands are overconfident and says so in the output.

Causality: features at bar t are trailing-only (tested in test_screener); the target at
bar t is close[t+h]/close[t]-1, used for *fitting* only — rows whose future isn't known
yet are never in the training set, and the live projection row is the latest bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketdata import MarketData
from screener import add_indicators

QUANTILES = (0.10, 0.50, 0.90)
FEATURES = ["rsi14", "macd_hist", "bb_pct_b", "bb_bandwidth", "atr_pct", "vol_z",
            "ret_20d", "off_52w_high", "ret_1", "ret_5"]


# ─────────────────────────────────────────── dataset

def build_dataset(px: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per-bar rows for ONE ticker: causal features + forward h-day return target.

    The last `horizon` rows have NaN target (future unknown) — kept, so the caller can
    use the final row for the live projection while excluding NaN targets from fitting.
    """
    ind = add_indicators(px)
    ind["ret_1"] = ind["close"].pct_change(1)
    ind["ret_5"] = ind["close"].pct_change(5)
    ind["target"] = ind["close"].shift(-horizon) / ind["close"] - 1.0
    return ind[FEATURES + ["target", "close"]]


def pooled_dataset(md: MarketData, tickers: list[str], start: str, end: str,
                   horizon: int, log=None) -> pd.DataFrame:
    """Stack per-ticker datasets, drop feature-warm-up rows, keep chronological order."""
    frames = []
    for tic in tickers:
        try:
            px = md.get_ohlcv(tic, start, end)
        except ValueError as e:
            if log is not None:
                log(f"[project] {tic}: skipped ({e})")
            continue
        d = build_dataset(px, horizon)
        d["tic"] = tic
        frames.append(d)
    if not frames:
        raise ValueError("no data for any requested ticker")
    out = pd.concat(frames)
    out = out.dropna(subset=FEATURES)             # warm-up rows out; NaN targets stay
    return out.sort_index(kind="stable")          # chronological across the pool


# ─────────────────────────────────────────── quantile models

def fit_quantiles(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """One small GBT per quantile."""
    from sklearn.ensemble import GradientBoostingRegressor
    models = {}
    for q in QUANTILES:
        m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=120,
                                      max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=seed)
        m.fit(X, y)
        models[q] = m
    return models


def predict_bands(models: dict, X: np.ndarray) -> np.ndarray:
    """(n, 3) array of [q10, q50, q90], sorted rowwise so bands can never cross."""
    preds = np.column_stack([models[q].predict(X) for q in QUANTILES])
    return np.sort(preds, axis=1)


# ─────────────────────────────────────────── the projection

def walkforward_coverage(data: pd.DataFrame, train_frac: float = 0.7,
                         seed: int = 42) -> dict:
    """Chronological 70/30 fit/eval: how often did actuals land inside [q10, q90] OOS?"""
    fit_rows = data.dropna(subset=["target"])
    cut_date = fit_rows.index.unique().sort_values()[int(len(fit_rows.index.unique()) * train_frac)]
    tr = fit_rows[fit_rows.index < cut_date]
    te = fit_rows[fit_rows.index >= cut_date]
    if len(tr) < 100 or len(te) < 30:
        return {"coverage": float("nan"), "n_test": int(len(te)), "median_abs_err": float("nan")}
    models = fit_quantiles(tr[FEATURES].to_numpy(), tr["target"].to_numpy(), seed=seed)
    bands = predict_bands(models, te[FEATURES].to_numpy())
    actual = te["target"].to_numpy()
    inside = (actual >= bands[:, 0]) & (actual <= bands[:, 2])
    return {"coverage": float(inside.mean()), "n_test": int(len(te)),
            "median_abs_err": float(np.median(np.abs(actual - bands[:, 1])))}


def project(tickers: list[str], horizon: int = 20, end: str | None = None,
            window_days: int = 1200, md: MarketData | None = None,
            seed: int = 42, log=None) -> tuple[pd.DataFrame, dict]:
    """Live h-day return bands per ticker + the OOS honesty check.

    Returns (table, honesty) where table has one row per ticker (close, q10/q50/q90 as %)
    and honesty is the walk-forward coverage dict for the same feature/model recipe.
    """
    md = md if md is not None else MarketData()
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start = (end_ts - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    data = pooled_dataset(md, tickers, start, end_ts.strftime("%Y-%m-%d"), horizon, log=log)

    honesty = walkforward_coverage(data, seed=seed)
    fit_rows = data.dropna(subset=["target"])
    models = fit_quantiles(fit_rows[FEATURES].to_numpy(), fit_rows["target"].to_numpy(),
                           seed=seed)

    rows = []
    for tic in data["tic"].unique():
        d = data[data["tic"] == tic]
        last = d.iloc[[-1]]
        band = predict_bands(models, last[FEATURES].to_numpy())[0]
        rows.append({"tic": tic, "date": d.index[-1].strftime("%Y-%m-%d"),
                     "close": float(last["close"].iloc[0]),
                     "q10_pct": round(100 * band[0], 2),
                     "q50_pct": round(100 * band[1], 2),
                     "q90_pct": round(100 * band[2], 2)})
    table = pd.DataFrame(rows).sort_values("q50_pct", ascending=False).reset_index(drop=True)
    return table, honesty


def format_projection(table: pd.DataFrame, honesty: dict, horizon: int) -> str:
    hdr = f"{'tic':<8} {'close':>9} | {horizon}d return band:   q10%     q50%     q90%"
    lines = [hdr, "-" * len(hdr)]
    for _, r in table.iterrows():
        lines.append(f"{r['tic']:<8} {r['close']:>9.2f} | "
                     f"{r['q10_pct']:>{18}.2f} {r['q50_pct']:>8.2f} {r['q90_pct']:>8.2f}")
    cov = honesty.get("coverage", float("nan"))
    note = ("n/a (too little history)" if not np.isfinite(cov) else
            f"{cov:.0%} of actuals fell inside [q10,q90] out-of-sample "
            f"(nominal 80%, n={honesty['n_test']})"
            + ("  — bands look OVERCONFIDENT, treat them as narrow" if cov < 0.70 else ""))
    lines.append(f"\nhonesty check: {note}")
    return "\n".join(lines)
