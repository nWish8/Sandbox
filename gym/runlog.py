"""runlog.py — structured recording of a training run for later playback (Vision M2/M3).

Every recorded run lands in ``gym/runs/<run_id>/`` as three artifacts:

    manifest.json   config, algo, reward, seed, timesteps, tickers, checkpoint steps
    record.npz      per-checkpoint deterministic rollouts over a fixed window:
                    equity (C,T), returns (C,T), turnover (C,T), weights (C,T,N),
                    bench_ret (T,), dates (T,), steps (C,)
    model.zip       the final trained SB3 model

so the playback visualizer can scrub through training progress ("generations") and step
bar-by-bar WITHOUT retraining. The recorder rolls the current policy deterministically over
one fixed window (usually the training window) at training start (checkpoint 0 = the
untrained policy), every ``every`` timesteps, and at the end.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"


class RunRecorder:
    """Attach to ``portfolio.train_portfolio(recorder=...)`` to record a replayable run."""

    def __init__(self, record_df: pd.DataFrame, cfg, *, reward: str, algo: str = "ppo",
                 lookback: int = 60, every: int = 5_000, seed: int = 42,
                 timesteps: int | None = None, run_id: str | None = None,
                 runs_dir: Path | str | None = None, log=print):
        self.record_df = record_df
        self.cfg = cfg
        self.reward = reward
        self.algo = algo
        self.lookback = lookback
        self.every = max(1, int(every))
        self.seed = seed
        self.timesteps = timesteps
        self.log = log
        self.run_id = run_id or f"{datetime.now():%Y%m%d_%H%M%S}_{algo}_{reward}"
        self.run_dir = Path(runs_dir if runs_dir is not None else RUNS_DIR) / self.run_id

        self.steps: list[int] = []
        self.histories: list[pd.DataFrame] = []
        self._next_mark = self.every

    # ---- rollout capture
    def _rollout(self, model, step: int) -> None:
        from portfolio import run_portfolio     # lazy: avoids a circular import
        hist = run_portfolio(model, self.record_df, self.cfg,
                             reward=self.reward, lookback=self.lookback)
        self.steps.append(int(step))
        self.histories.append(hist)
        self.log(f"[runlog] checkpoint @ {step} steps "
                 f"(final equity {hist['value'].iloc[-1] / hist['value'].iloc[0]:.4f})")

    def sb3_callback(self):
        """An SB3 callback that snapshots a rollout at checkpoint marks."""
        from stable_baselines3.common.callbacks import BaseCallback
        rec = self

        class _RecorderCallback(BaseCallback):
            def _on_training_start(self) -> None:
                rec._rollout(self.model, 0)                    # generation 0: untrained

            def _on_step(self) -> bool:
                if self.num_timesteps >= rec._next_mark:
                    rec._rollout(self.model, self.num_timesteps)
                    while rec._next_mark <= self.num_timesteps:
                        rec._next_mark += rec.every
                return True

        return _RecorderCallback()

    # ---- persistence
    def finalize(self, model) -> Path:
        """Record the final state (if not already at the last mark), save everything."""
        if model is not None:
            last = model.num_timesteps
            if not self.steps or self.steps[-1] < last:
                self._rollout(model, last)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        tics = sorted(self.record_df["tic"].unique().tolist())
        w_cols = [f"w_{t}" for t in tics]
        h0 = self.histories[0]
        dates = h0["date"].astype(str).to_numpy()
        np.savez_compressed(
            self.run_dir / "record.npz",
            steps=np.array(self.steps, dtype=np.int64),
            dates=dates.astype("U10"),
            bench_ret=h0["bench_ret"].to_numpy(),
            equity=np.stack([(h["value"] / h["value"].iloc[0]).to_numpy()
                             for h in self.histories]),
            returns=np.stack([h["ret"].to_numpy() for h in self.histories]),
            turnover=np.stack([h["turnover"].to_numpy() for h in self.histories]),
            weights=np.stack([h[w_cols].to_numpy() for h in self.histories]),
        )
        from dataclasses import asdict, is_dataclass
        cfg_d = asdict(self.cfg) if is_dataclass(self.cfg) else dict(self.cfg)
        cfg_d["indicators"] = list(cfg_d.get("indicators", []))
        manifest = {
            "run_id": self.run_id, "algo": self.algo, "reward": self.reward,
            "seed": self.seed, "timesteps": self.timesteps, "every": self.every,
            "lookback": self.lookback, "tics": tics, "n_checkpoints": len(self.steps),
            "created": datetime.now().isoformat(timespec="seconds"), "config": cfg_d,
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        if model is not None:
            model.save(str(self.run_dir / "model"))
        self.log(f"[runlog] saved {len(self.steps)} checkpoints -> {self.run_dir}")
        return self.run_dir


# ─────────────────────────────────────────── loading

class RunRecord:
    """A recorded run, loaded for playback / promotion."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        z = np.load(self.run_dir / "record.npz", allow_pickle=False)
        self.steps = z["steps"]              # (C,)
        self.dates = z["dates"]              # (T,)
        self.bench_ret = z["bench_ret"]      # (T,)
        self.equity = z["equity"]            # (C, T)
        self.returns = z["returns"]          # (C, T)
        self.turnover = z["turnover"]        # (C, T)
        self.weights = z["weights"]          # (C, T, N)
        self.tics = list(self.manifest["tics"])

    @property
    def n_checkpoints(self) -> int:
        return int(self.equity.shape[0])

    @property
    def n_bars(self) -> int:
        return int(self.equity.shape[1])

    @property
    def bench_equity(self) -> np.ndarray:
        return np.cumprod(1.0 + self.bench_ret)

    def model_path(self) -> Path | None:
        p = self.run_dir / "model.zip"
        return p if p.exists() else None

    @staticmethod
    def load(run_id_or_dir: str | Path, runs_dir: Path | str | None = None) -> "RunRecord":
        p = Path(run_id_or_dir)
        if not p.is_dir():
            p = Path(runs_dir if runs_dir is not None else RUNS_DIR) / str(run_id_or_dir)
        return RunRecord(p)


def list_runs(runs_dir: Path | str | None = None) -> list[str]:
    """Run ids under the runs dir (newest first) that have a complete record."""
    base = Path(runs_dir if runs_dir is not None else RUNS_DIR)
    if not base.is_dir():
        return []
    out = [d.name for d in base.iterdir()
           if d.is_dir() and (d / "record.npz").exists() and (d / "manifest.json").exists()]
    return sorted(out, reverse=True)
