"""evo.py — ES / neuroevolution trainer for the visual gym (the population learner).

A population of small MLP policies is evolved by truncation selection + Gaussian mutation —
no gradients. Each generation:

  1. evaluate every genome on a TRAIN pass (shared, re-rolled order; capital threaded;
     ruined agents get -inf fitness),
  2. keep the top `elite_frac` by train fitness (elitism), breed the rest as mutated elites,
  3. evaluate the generation's best genome on a VAL pass and record it.

The champion is the generation with the best VAL objective (so the *search* is driven by
train fitness but final selection is val-gated — val is never the breeding target, which
would just overfit to it). Test is touched once on the champion.

This mirrors the train/val/test discipline of the PPO loop (`train.py`) but with a
population you can watch survive and get culled. The fitness is `stats.objective_value`
on the threaded pass, so any backtesting.py-style metric can drive selection.

Public API
----------
EvoPolicy            — numpy MLP, act(obs)->weight in [0,1]; genome <-> flat vector.
evolve(...)          — run the generational loop, return EvoResult.
save_champion/load_champion — persist/restore the winning genome + arch.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gym.config import Config, DEFAULT_CONFIG, GYM_MODELS
from gym.features import feature_columns
from gym.population import run_pass, run_population, shuffled_order
from gym.stats import objective_value

log = logging.getLogger(__name__)


# ─────────────────────────────────────────── policy

class EvoPolicy:
    """A small MLP mapping the current bar's feature vector to a target weight in [0,1].

    Architecture: input -> tanh(hidden) -> sigmoid(1). The genome is a flat float vector
    (all weights + biases) so evolution can mutate/recombine it directly.

    window_mode:
      "last"    — input is obs[-1] (current bar's features, n_feat dims). Default; keeps the
                  search dimensionality low (no gradients) and the decision interpretable.
      "flatten" — input is the whole lookback window flattened (lookback*n_feat dims).
    """

    def __init__(self, n_feat: int, hidden: int = 32, window_mode: str = "last",
                 lookback: int = 30, genome: np.ndarray | None = None,
                 rng: np.random.Generator | None = None, init_scale: float = 0.5):
        if window_mode not in ("last", "flatten"):
            raise ValueError(f"window_mode must be 'last' or 'flatten', got {window_mode!r}")
        self.n_feat = int(n_feat)
        self.hidden = int(hidden)
        self.window_mode = window_mode
        self.lookback = int(lookback)
        self.n_in = self.n_feat if window_mode == "last" else self.n_feat * self.lookback

        self._shapes = [(self.n_in, self.hidden), (self.hidden,), (self.hidden, 1), (1,)]
        self._sizes = [int(np.prod(s)) for s in self._shapes]
        self.n_params = int(sum(self._sizes))

        if genome is None:
            rng = rng if rng is not None else np.random.default_rng()
            genome = rng.normal(0.0, init_scale, self.n_params)
        self.set_genome(genome)

    def set_genome(self, genome: np.ndarray) -> None:
        g = np.asarray(genome, dtype=np.float64).reshape(-1)
        if g.size != self.n_params:
            raise ValueError(f"genome size {g.size} != n_params {self.n_params}")
        self.genome = g
        parts, i = [], 0
        for sz, shp in zip(self._sizes, self._shapes):
            parts.append(g[i:i + sz].reshape(shp))
            i += sz
        self.W1, self.b1, self.W2, self.b2 = parts

    def act(self, obs: np.ndarray) -> float:
        x = obs[-1] if self.window_mode == "last" else np.asarray(obs, dtype=np.float64).reshape(-1)
        h = np.tanh(x @ self.W1 + self.b1)
        y = float((h @ self.W2 + self.b2)[0])
        y = float(np.clip(y, -30.0, 30.0))            # sigmoid overflow guard
        return 1.0 / (1.0 + np.exp(-y))

    def reset(self) -> None:
        pass

    @classmethod
    def from_genome(cls, genome: np.ndarray, n_feat: int, hidden: int = 32,
                    window_mode: str = "last", lookback: int = 30) -> "EvoPolicy":
        return cls(n_feat, hidden=hidden, window_mode=window_mode, lookback=lookback,
                   genome=genome)


# ─────────────────────────────────────────── config + results

@dataclass
class EvoConfig:
    pop_size: int = 64
    n_generations: int = 40
    elite_frac: float = 0.25
    mutation_sigma: float = 0.1
    sigma_decay: float = 1.0          # *= each generation (1.0 = constant)
    hidden: int = 32
    window_mode: str = "last"
    ruin_frac: float = 0.45
    # default to timing-vs-own-average-exposure-twin: selects for genuine trade timing,
    # not de-risking (a constant-exposure agent scores 0). See stats.OBJECTIVES.
    objective: str = "timing_sortino"
    early_stop_patience: int = 8      # generations w/o val improvement -> stop
    min_trades: int = 3
    init_scale: float = 0.5
    seed: int = 42


@dataclass
class GenerationRecord:
    generation: int
    best_train_fitness: float
    mean_train_fitness: float
    n_survivors: int                  # non-ruined this generation
    val_fitness: float
    val_stats: dict = field(default_factory=dict)
    champion_genome: np.ndarray | None = field(default=None, repr=False)


@dataclass
class EvoResult:
    n_feat: int
    hidden: int
    window_mode: str
    lookback: int
    objective: str
    generations: list[GenerationRecord]
    best_generation_idx: int
    champion_genome: np.ndarray
    val_fitness: float
    test_stats: dict | None
    elapsed_s: float
    evo_cfg: EvoConfig


# ─────────────────────────────────────────── persistence

def save_champion(result: EvoResult, path=None):
    path = path if path is not None else GYM_MODELS / "evo_champion.npz"
    GYM_MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), genome=result.champion_genome,
             n_feat=result.n_feat, hidden=result.hidden,
             window_mode=result.window_mode, lookback=result.lookback,
             objective=result.objective)
    return path


def load_champion(path=None) -> EvoPolicy | None:
    from pathlib import Path
    path = path if path is not None else GYM_MODELS / "evo_champion.npz"
    if not Path(path).exists():
        return None
    d = np.load(str(path), allow_pickle=True)
    return EvoPolicy.from_genome(
        d["genome"], int(d["n_feat"]), hidden=int(d["hidden"]),
        window_mode=str(d["window_mode"]), lookback=int(d["lookback"]),
    )


# ─────────────────────────────────────────── the generational loop

def evolve(train_dfs: list[pd.DataFrame], val_dfs: list[pd.DataFrame],
           test_dfs: list[pd.DataFrame] | None = None, cfg: Config = DEFAULT_CONFIG,
           evo_cfg: EvoConfig | None = None, monitor=None, record_path=None) -> EvoResult:
    """Evolve a population through the universe; return the val-selected champion.

    If `record_path` is given, the final generation's population pass is saved there (a
    `GenerationReplay` .npz + .json) so `evo-replay` can render the race afterwards.
    """
    if not train_dfs:
        raise ValueError("train_dfs is empty")
    if evo_cfg is None:
        evo_cfg = EvoConfig()

    from gym.normalize import fit_obs_stats   # train-fit obs normalization (ES needs it too)

    t0 = time.time()
    feat_cols = feature_columns(train_dfs[0], cfg)
    n_feat = len(feat_cols)
    lookback = cfg.lookback
    obs_stats = fit_obs_stats(train_dfs, cfg)
    rng = np.random.default_rng(evo_cfg.seed)

    obj = evo_cfg.objective
    n_elite = max(1, int(round(evo_cfg.pop_size * evo_cfg.elite_frac)))
    sigma = evo_cfg.mutation_sigma

    # a probe policy gives the parameter count for this architecture
    probe = EvoPolicy(n_feat, hidden=evo_cfg.hidden, window_mode=evo_cfg.window_mode,
                      lookback=lookback, rng=rng, init_scale=evo_cfg.init_scale)
    n_params = probe.n_params
    population = [rng.normal(0.0, evo_cfg.init_scale, n_params)
                 for _ in range(evo_cfg.pop_size)]
    names = [f"agent {i:02d}" for i in range(evo_cfg.pop_size)]

    # a fixed val order so val fitness is comparable across generations
    val_order = shuffled_order(val_dfs, seed=evo_cfg.seed) if val_dfs else []

    log.info("Evolve: pop=%d  gens=%d  elite=%d  n_params=%d  n_feat=%d  objective=%s",
             evo_cfg.pop_size, evo_cfg.n_generations, n_elite, n_params, n_feat, obj)

    records: list[GenerationRecord] = []
    best_val = float("-inf")
    patience = evo_cfg.early_stop_patience
    last_results = None              # final generation's population pass (for recording)
    last_order_dfs: list = []
    last_best_idx = 0

    def _policy(genome):
        return EvoPolicy.from_genome(genome, n_feat, hidden=evo_cfg.hidden,
                                     window_mode=evo_cfg.window_mode, lookback=lookback)

    for g in range(evo_cfg.n_generations):
        # 1) evaluate population on a re-rolled TRAIN pass
        train_order = shuffled_order(train_dfs, seed=evo_cfg.seed + g + 1)
        policies = [_policy(genome) for genome in population]
        results = run_population(policies, train_order, cfg, ruin_frac=evo_cfg.ruin_frac,
                                 obs_stats=obs_stats, objective=obj)
        fitnesses = np.array([
            objective_value(r.stats, obj, min_trades=evo_cfg.min_trades)
            if not r.ruined else float("-inf")
            for r in results
        ])
        n_survivors = int(np.sum([not r.ruined for r in results]))

        order = np.argsort(fitnesses)[::-1]          # best first
        elite_idx = order[:n_elite]
        elite_genomes = [population[i] for i in elite_idx]
        best_genome = population[int(order[0])]
        last_results, last_order_dfs, last_best_idx = results, train_order, int(order[0])
        finite = fitnesses[np.isfinite(fitnesses)]
        best_train = float(fitnesses[int(order[0])])
        mean_train = float(np.mean(finite)) if finite.size else float("-inf")

        # 2) val-gate the generation's best genome (threaded val pass)
        if val_order:
            val_res = run_pass(_policy(best_genome), val_order, cfg,
                               ruin_frac=evo_cfg.ruin_frac, obs_stats=obs_stats, objective=obj)
            val_fit = (objective_value(val_res.stats, obj, min_trades=evo_cfg.min_trades)
                       if not val_res.ruined else float("-inf"))
            val_stats = val_res.stats
        else:
            val_fit, val_stats, val_res = float("nan"), {}, None

        records.append(GenerationRecord(
            generation=g, best_train_fitness=best_train, mean_train_fitness=mean_train,
            n_survivors=n_survivors, val_fitness=val_fit, val_stats=val_stats,
            champion_genome=best_genome.copy(),
        ))
        log.info("Gen %2d | train best=%.4f mean=%.4f  survivors=%d/%d  val=%.4f  sigma=%.3f",
                 g, best_train, mean_train, n_survivors, evo_cfg.pop_size, val_fit, sigma)

        if monitor is not None and hasattr(monitor, "emit"):
            ev = {"type": "checkpoint", "agent": "evo", "timestep": g,
                  "val_sharpe": val_fit, "beat_rate": float("nan")}
            if val_res is not None and not val_res.ruined and "equity" in val_res.ec:
                ev["equity"] = val_res.ec["equity"].dropna().tolist()
                ev["bh_equity"] = val_res.ec["bh_equity"].tolist()[: len(ev["equity"])]
            monitor.emit(ev)
            monitor.emit({"type": "rollout", "timestep": g, "ret": best_train})
            # rich per-generation snapshot for the live evo dashboard (EvoMonitor)
            from gym.evo_monitor import build_generation_event
            monitor.emit(build_generation_event(
                g, results, fitnesses, names, int(order[0]), evo_cfg.ruin_frac,
                best_train, mean_train, val_fit))

        # 3) early stop on val improvement
        if np.isfinite(val_fit) and val_fit > best_val:
            best_val = val_fit
            patience = evo_cfg.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                log.info("Early stop at gen %d (no val improvement)", g)
                break

        # 4) breed next generation: elites carried forward, rest = mutated elites
        next_pop = [eg.copy() for eg in elite_genomes]               # elitism
        while len(next_pop) < evo_cfg.pop_size:
            parent = elite_genomes[rng.integers(len(elite_genomes))]
            child = parent + rng.normal(0.0, sigma, n_params)
            next_pop.append(child)
        population = next_pop
        sigma *= evo_cfg.sigma_decay

    # champion = generation with best val fitness (fall back to train if no val)
    key = (lambda r: r.val_fitness) if val_order else (lambda r: r.best_train_fitness)
    best_idx = max(range(len(records)),
                   key=lambda i: (key(records[i]) if np.isfinite(key(records[i])) else float("-inf")))
    champion_genome = records[best_idx].champion_genome
    champion_val = records[best_idx].val_fitness

    test_stats = None
    if test_dfs:
        test_order = shuffled_order(test_dfs, seed=evo_cfg.seed)
        test_res = run_pass(_policy(champion_genome), test_order, cfg,
                            ruin_frac=evo_cfg.ruin_frac, obs_stats=obs_stats, objective=obj)
        test_stats = test_res.stats
        log.info("Test | objective(%s)=%.4f  ruined=%s  return=%.4f  maxDD=%.4f",
                 obj, objective_value(test_stats, obj, min_trades=evo_cfg.min_trades),
                 test_stats.get("ruined"), test_stats.get("total_return", float("nan")),
                 test_stats.get("max_drawdown", float("nan")))

    # record the final generation's race for evo-replay
    if record_path is not None and last_results is not None:
        from gym.evo_replay import GenerationReplay
        slugs = [str(df["ticker"].iloc[0]) if "ticker" in df.columns else f"T{i}"
                 for i, df in enumerate(last_order_dfs)]
        GenerationReplay.from_results(last_results, slugs, obj,
                                      champion_idx=last_best_idx).save(record_path)
        log.info("Recorded final-generation race to %s", record_path)

    return EvoResult(
        n_feat=n_feat, hidden=evo_cfg.hidden, window_mode=evo_cfg.window_mode,
        lookback=lookback, objective=obj, generations=records,
        best_generation_idx=best_idx, champion_genome=champion_genome,
        val_fitness=champion_val, test_stats=test_stats,
        elapsed_s=time.time() - t0, evo_cfg=evo_cfg,
    )
