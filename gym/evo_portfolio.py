"""evo_portfolio.py — GPU-batched population evolution on a multi-stock portfolio.

A *population* of small MLP policies is evolved (no gradients) to allocate a long-only
portfolio across the basket. Each agent, every bar, observes the market features and its
own current weights and emits a softmax allocation over {tickers, cash}. The WHOLE
population forward-passes as one batched matmul (torch.bmm) on the GPU — this is where the
RTX 4050 actually helps (vs FinRL's single-agent DRL where it was slower).

Why portfolio-allocation (softmax weights), not FinRL's share-trading action: weights ARE
the "variable allocation values" we want to visualise, they're naturally long-only and
bounded, and softmax(≈0) starts at an equal-weight portfolio — a stable, sane init that
sidesteps the churn-to-ruin problem bang-bang policies hit.

Pipeline: load DOW-30 daily panel → time-split train/val/test → evolve on train,
val-gate the champion, touch test once → record per-bar prices/allocations/portfolio value
for the stepped replay.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
TRADING_DAYS = 252


# ─────────────────────────────────────────── data panel

@dataclass
class Panel:
    dates: np.ndarray            # (T,) datetime64
    tics: list[str]              # N
    returns: np.ndarray          # (T, N) simple return from bar t -> t+1 (last row 0)
    close: np.ndarray            # (T, N) close price (for the ticker pane)
    feats: np.ndarray            # (T, F) z-scored market features (causal)
    feat_names: list[str]

    @property
    def T(self): return self.returns.shape[0]

    @property
    def N(self): return self.returns.shape[1]

    @property
    def F(self): return self.feats.shape[1]


def load_panel(csv_path: str | Path, indicators: list[str] | None = None) -> Panel:
    """Build a Panel from a FinRL long CSV (date,tic,close,<indicators>...)."""
    df = pd.read_csv(csv_path)
    df = df[df.columns.drop([c for c in df.columns if c.lower().startswith("unnamed")])]
    df["date"] = pd.to_datetime(df["date"])
    if indicators is None:
        from finrl.config import INDICATORS
        indicators = list(INDICATORS)
    indicators = [c for c in indicators if c in df.columns]

    tics = sorted(df["tic"].unique().tolist())
    piv_close = df.pivot(index="date", columns="tic", values="close").sort_index()
    piv_close = piv_close[tics].ffill().bfill()
    dates = piv_close.index.values
    close = piv_close.to_numpy(dtype=np.float64)                      # (T, N)
    rets = np.zeros_like(close)
    rets[:-1] = close[1:] / close[:-1] - 1.0                          # t -> t+1

    # per-ticker indicator panels, flattened to (T, N*K) market features
    feat_panels, feat_names = [], []
    for ind in indicators:
        p = df.pivot(index="date", columns="tic", values=ind).sort_index()[tics]
        feat_panels.append(p.ffill().bfill().to_numpy(dtype=np.float64))
        feat_names += [f"{ind}:{t}" for t in tics]
    feats = np.concatenate(feat_panels, axis=1) if feat_panels else np.zeros((len(dates), 0))
    return Panel(dates=dates, tics=tics, returns=rets, close=close,
                 feats=feats, feat_names=feat_names)


# ─────────────────────────────────────────── config + result

@dataclass
class EvoPortfolioConfig:
    pop_size: int = 128
    n_generations: int = 30
    hidden: int = 64
    elite_frac: float = 0.25
    mutation_sigma: float = 0.1
    init_scale: float = 0.1
    cost: float = 0.001            # turnover cost per unit |Δweight|
    smooth: float = 0.20           # realized = smooth*target + (1-smooth)*prev (eases churn)
    objective: str = "sharpe"      # sharpe | return | sortino
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15         # rest is test


@dataclass
class GenRecord:
    generation: int
    best_train: float
    mean_train: float
    val_fitness: float


@dataclass
class EvoPortfolioResult:
    cfg: EvoPortfolioConfig
    tics: list[str]
    generations: list[GenRecord]
    champion_genome: np.ndarray
    n_params: int
    split_idx: tuple[int, int]     # (train_end, val_end) row indices
    elapsed_s: float


# ─────────────────────────────────────────── the engine

class PortfolioEvo:
    def __init__(self, panel: Panel, cfg: EvoPortfolioConfig):
        self.panel = panel
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.N = panel.N
        self.in_dim = panel.F + self.N + 1          # market feats + own weights (+cash)
        self.out_dim = self.N + 1                    # tickers + cash
        H = cfg.hidden
        self._shapes = [(self.in_dim, H), (H,), (H, self.out_dim), (self.out_dim,)]
        self._sizes = [int(np.prod(s)) for s in self._shapes]
        self.n_params = int(sum(self._sizes))

        # z-score features on TRAIN rows only (causal, no leakage)
        tr_end = int(panel.T * cfg.train_frac)
        f = panel.feats
        mu = f[:tr_end].mean(0); sd = f[:tr_end].std(0)
        sd[(sd == 0) | ~np.isfinite(sd)] = 1.0
        self.feats = torch.tensor(((f - mu) / sd), dtype=torch.float32, device=self.dev)
        self.rets = torch.tensor(panel.returns, dtype=torch.float32, device=self.dev)
        self.tr_end = tr_end
        self.val_end = int(panel.T * (cfg.train_frac + cfg.val_frac))

    # genome tensor (P, n_params) -> per-agent weight matrices
    def _unpack(self, g: torch.Tensor):
        P = g.shape[0]; i = 0; parts = []
        for sz, shp in zip(self._sizes, self._shapes):
            parts.append(g[:, i:i + sz].reshape(P, *shp)); i += sz
        return parts                                  # W1,b1,W2,b2

    def _alloc(self, packs, x):                        # x: (P, in_dim) -> (P, out_dim)
        W1, b1, W2, b2 = packs
        h = torch.tanh(torch.bmm(x.unsqueeze(1), W1).squeeze(1) + b1)
        logits = torch.bmm(h.unsqueeze(1), W2).squeeze(1) + b2
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def evaluate(self, genomes: torch.Tensor, t0: int, t1: int, record: bool = False):
        """Run the whole population over bars [t0,t1). Returns fitness (P,) and, if
        record, also value_hist (P, S) and weight_hist (P, S, N+1)."""
        P = genomes.shape[0]
        packs = self._unpack(genomes)
        w = torch.full((P, self.out_dim), 1.0 / self.out_dim, device=self.dev)  # equal-weight init
        pv = torch.ones(P, device=self.dev)
        rets_log = []
        v_hist, w_hist = [], []
        for t in range(t0, t1):
            x = torch.cat([self.feats[t].unsqueeze(0).expand(P, -1), w], dim=1)
            target = self._alloc(packs, x)
            nw = self.cfg.smooth * target + (1.0 - self.cfg.smooth) * w          # eased rebalance
            port_ret = (nw[:, :self.N] * self.rets[t].unsqueeze(0)).sum(1)       # cash earns 0
            turn = self.cfg.cost * (nw - w).abs().sum(1)
            step = port_ret - turn
            pv = pv * (1.0 + step)
            rets_log.append(step)
            w = nw
            if record:
                v_hist.append(pv.clone()); w_hist.append(nw.clone())
        R = torch.stack(rets_log, dim=1)              # (P, S)
        fit = self._fitness(R, pv)
        if record:
            return fit, torch.stack(v_hist, 1).cpu().numpy(), torch.stack(w_hist, 1).cpu().numpy()
        return fit

    def _fitness(self, R: torch.Tensor, pv: torch.Tensor) -> torch.Tensor:
        obj = self.cfg.objective
        if obj == "return":
            return pv - 1.0
        mean = R.mean(1)
        if obj == "sortino":
            dn = torch.sqrt((R.clamp(max=0.0) ** 2).mean(1)) + 1e-9
            return mean / dn * (TRADING_DAYS ** 0.5)
        std = R.std(1) + 1e-9                          # default: Sharpe
        return mean / std * (TRADING_DAYS ** 0.5)

    def evolve(self, log=print, record_path: str | Path | None = None,
               gen_cb=None, stop=None) -> EvoPortfolioResult:
        cfg = self.cfg
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        pop = (torch.randn(cfg.pop_size, self.n_params, generator=g) * cfg.init_scale).to(self.dev)
        n_elite = max(1, int(round(cfg.pop_size * cfg.elite_frac)))
        recs: list[GenRecord] = []
        best_val = -1e18; champ = pop[0].clone()
        t0 = time.time()
        log(f"[evo] pop={cfg.pop_size} gens={cfg.n_generations} params={self.n_params} "
            f"device={self.dev} N={self.N} F={self.panel.F} obj={cfg.objective}")

        for gen in range(cfg.n_generations):
            if stop is not None and stop():
                log(f"[evo] stopped at gen {gen}"); break
            fit = self.evaluate(pop, 0, self.tr_end)               # train
            order = torch.argsort(fit, descending=True)
            elite = pop[order[:n_elite]]
            best_g = pop[order[0]].clone()
            # val-gate the generation's best
            vfit = float(self.evaluate(best_g.unsqueeze(0), self.tr_end, self.val_end)[0])
            recs.append(GenRecord(gen, float(fit[order[0]]), float(fit.mean()), vfit))
            if vfit > best_val:
                best_val = vfit; champ = best_g.clone()
            log(f"  gen {gen:2d}  train best={float(fit[order[0]]):+.3f} "
                f"mean={float(fit.mean()):+.3f}  val={vfit:+.3f}")
            if gen_cb is not None:
                gen_cb(gen, float(fit[order[0]]), float(fit.mean()), vfit)
            # breed: elites + mutated elites
            children = [elite]
            while sum(c.shape[0] for c in children) < cfg.pop_size:
                k = cfg.pop_size - sum(c.shape[0] for c in children)
                idx = torch.randint(0, n_elite, (min(k, n_elite),), device=self.dev)
                children.append(elite[idx] + torch.randn_like(elite[idx]) * cfg.mutation_sigma)
            pop = torch.cat(children, 0)[:cfg.pop_size]

        res = EvoPortfolioResult(cfg=cfg, tics=self.panel.tics, generations=recs,
                                 champion_genome=champ.cpu().numpy(), n_params=self.n_params,
                                 split_idx=(self.tr_end, self.val_end), elapsed_s=time.time() - t0)
        if record_path is not None:
            self.record_run(pop, champ, record_path, log=log)
        log(f"[evo] done {res.elapsed_s:.1f}s  best_val={best_val:+.3f}")
        return res

    @torch.no_grad()
    def record_run(self, final_pop: torch.Tensor, champion: torch.Tensor,
                   path: str | Path, log=print):
        """Record the final population over the WHOLE timeline for the stepped replay:
        per-bar portfolio values (all agents) + champion allocations + ticker prices."""
        path = Path(path).with_suffix(".npz")
        _, val_hist, w_hist = self.evaluate(final_pop, 0, self.panel.T - 1, record=True)
        # champion index within final_pop (best train fitness)
        fit = self.evaluate(final_pop, 0, self.tr_end)
        champ_idx = int(torch.argmax(fit))
        S = val_hist.shape[1]
        np.savez(str(path),
                 dates=self.panel.dates[:S].astype("datetime64[s]").astype(np.int64),
                 tics=np.array(self.panel.tics),
                 close=self.panel.close[:S],                 # (S, N)
                 values=val_hist,                            # (P, S)
                 champ_weights=w_hist[champ_idx],            # (S, N+1)
                 champ_idx=champ_idx,
                 split=np.array(self.split_idx if hasattr(self, "split_idx") else
                                [self.tr_end, self.val_end]))
        log(f"[evo] recorded replay -> {path}  (agents={val_hist.shape[0]}, bars={S})")
        return path


def run(csv_path="train_data.csv", **kw):
    """Convenience: load panel + evolve + record. Returns the result."""
    panel = load_panel(HERE / csv_path)
    cfg = EvoPortfolioConfig(**kw)
    evo = PortfolioEvo(panel, cfg)
    evo.split_idx = (evo.tr_end, evo.val_end)
    return evo.evolve(record_path=HERE / "evo_portfolio_record.npz")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    run(pop_size=64, n_generations=8)
