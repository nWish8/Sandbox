"""evo_monitor.py — live dashboard for evolutionary training (per-generation refresh).

Watch the population race *during* training. Granularity is per generation (not per bar —
per-bar live rendering would throttle training): each generation finishes and the view
redraws three panels:
  A — population equity race (champion bold, ruined red, B&H dashed, ruin line),
  B — skill curve (best + mean train fitness, and val fitness, per generation),
  C — leaderboard (top agents by the objective).

The pure payload builder (`build_generation_event`, `downsample`) is import-safe and unit
tested; the pyqtgraph window mirrors `monitor.py` (background thread + thread-safe emit queue)
and degrades to a no-op when pyqtgraph is unavailable. `evolve(..., monitor=EvoMonitor())`
emits one event per generation.
"""
from __future__ import annotations

import logging
import queue
import threading

import numpy as np

log = logging.getLogger(__name__)

_PYQTGRAPH_AVAILABLE = False
try:
    import pyqtgraph as pg
    # import Qt THROUGH pyqtgraph so we always match its binding — mixing PyQt5 with a
    # PyQt6-bound pyqtgraph segfaults at widget construction (both are installed here).
    from pyqtgraph.Qt import QtCore, QtWidgets
    try:
        _DASH = QtCore.Qt.PenStyle.DashLine        # PyQt6 scoped enum
    except AttributeError:
        _DASH = QtCore.Qt.DashLine                 # PyQt5 flat enum
    _PYQTGRAPH_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────── pure payload layer (tested)

def downsample(arr, n_points: int):
    """Evenly subsample a 1-D array to at most n_points, preserving NaNs (ruin truncation)."""
    a = np.asarray(arr, dtype=np.float64)
    if len(a) <= n_points:
        return a
    idx = np.linspace(0, len(a) - 1, n_points).astype(int)
    return a[idx]


def build_generation_event(generation: int, results, fitnesses, names, champion_idx: int,
                           ruin_frac: float, best_train: float, mean_train: float,
                           val_fit: float, n_points: int = 200, top_n: int = 12) -> dict:
    """Build a compact, picklable per-generation event from a population pass.

    `results` is a list of `population.PassResult`; `fitnesses` the matching scalars. Equity
    curves are downsampled for cheap live transport over the monitor queue.
    """
    fit = np.asarray(fitnesses, dtype=np.float64)
    equity = [downsample(r.ec["equity"].to_numpy(), n_points).tolist() for r in results]
    bh = downsample(results[0].ec["bh_equity"].to_numpy(), n_points).tolist()
    ruined = [bool(r.ruined) for r in results]
    order = sorted(range(len(results)),
                   key=lambda i: fit[i] if np.isfinite(fit[i]) else -np.inf, reverse=True)
    leaderboard = [{
        "name": names[i], "fitness": float(fit[i]), "ruined": ruined[i],
        "weight_std": results[i].stats.get("weight_std"),
        "turnover": results[i].stats.get("turnover"),
    } for i in order[:top_n]]
    return {
        "type": "generation", "generation": int(generation),
        "equity": equity, "bh": bh, "ruined": ruined,
        "champion_idx": int(champion_idx), "ruin_frac": float(ruin_frac),
        "best_train": float(best_train), "mean_train": float(mean_train),
        "val_fit": float(val_fit), "leaderboard": leaderboard, "names": list(names),
    }


# ─────────────────────────────────────────── no-op fallback

class _NoOpEvoMonitor:
    def emit(self, event) -> None:
        if event and event.get("type") == "generation":
            log.info("Gen %s | best=%.4f mean=%.4f val=%.4f",
                     event.get("generation"), event.get("best_train", float("nan")),
                     event.get("mean_train", float("nan")), event.get("val_fit", float("nan")))

    def run(self, worker_callable) -> None:
        """No GUI: just run the worker in the calling thread."""
        worker_callable()


# ─────────────────────────────────────────── real monitor

if _PYQTGRAPH_AVAILABLE:

    class _EvoWindow(QtWidgets.QMainWindow):
        _new_event = QtCore.pyqtSignal(dict)

        def __init__(self):
            super().__init__()
            self.setWindowTitle("Signal Gym — Evolution Monitor")
            self.resize(1280, 800)
            cw = pg.GraphicsLayoutWidget()
            self.setCentralWidget(cw)

            # Panel A: population equity race (spans both columns)
            self._race = cw.addPlot(row=0, col=0, colspan=2, title="A: Population equity race")
            self._race.setLabel("bottom", "Bar (downsampled)")
            self._race.setLogMode(False, True)   # equity spans 1 -> ~100x; log keeps it legible
            self._race.addLegend()

            # Panel B: skill curve
            self._skill = cw.addPlot(row=1, col=0, title="B: Skill curve (fitness / generation)")
            self._skill.setLabel("bottom", "Generation")
            self._skill.addLegend()
            self._line_best = self._skill.plot([], [], pen=pg.mkPen("limegreen", width=2),
                                                name="best train")
            self._line_mean = self._skill.plot([], [], pen=pg.mkPen("gray", width=1),
                                                name="mean train")
            self._line_val = self._skill.plot([], [], pen=pg.mkPen("deepskyblue", width=2),
                                               name="val")
            self._skill.addLine(y=0, pen=pg.mkPen("dimgray", style=_DASH))

            # Panel C: leaderboard bars
            self._board = cw.addPlot(row=1, col=1, title="C: Leaderboard (objective)")

            self._best, self._mean, self._val = [], [], []
            self._new_event.connect(self._on_event)

        def post(self, event: dict) -> None:
            self._new_event.emit(event)

        @QtCore.pyqtSlot(dict)
        def _on_event(self, event: dict) -> None:
            if event.get("type") != "generation":
                return

            # Panel A: redraw all agent curves for this generation
            self._race.clear()
            champ = event["champion_idx"]
            ruined = event["ruined"]
            for i, eq in enumerate(event["equity"]):
                if i == champ:
                    continue
                color = (200, 60, 50) if ruined[i] else (150, 155, 160, 120)
                self._race.plot(eq, pen=pg.mkPen(color, width=1))
            self._race.plot(event["bh"], pen=pg.mkPen("orange", width=2,
                                                       style=_DASH))
            self._race.plot(event["equity"][champ], pen=pg.mkPen("deepskyblue", width=3))
            n = len(event["bh"])
            self._race.plot([0, n - 1], [event["ruin_frac"]] * 2,
                            pen=pg.mkPen((160, 45, 45), width=1, style=_DASH))

            # Panel B: append skill-curve points
            self._best.append(event["best_train"])
            self._mean.append(event["mean_train"])
            self._val.append(event["val_fit"])
            xs = list(range(len(self._best)))
            self._line_best.setData(xs, self._best)
            self._line_mean.setData(xs, self._mean)
            self._line_val.setData(xs, self._val)

            # Panel C: leaderboard bars (top agents by fitness)
            board = event["leaderboard"]
            self._board.clear()
            vals = [r["fitness"] for r in board][::-1]
            names = [r["name"] for r in board][::-1]
            y = list(range(len(vals)))
            bars = pg.BarGraphItem(x0=0, y=y, height=0.7, width=vals, brush="mediumpurple")
            self._board.addItem(bars)
            self._board.getAxis("left").setTicks([list(zip(y, names))])

    class EvoMonitor:
        """Live evolution monitor — Qt event loop on the MAIN thread, training on a worker.

        Qt GUI objects must live on the main thread; running the QApplication on a background
        thread crashes on Windows/PyQt6. So `run(worker)` owns the main thread (QApplication +
        window + drain timer) and spawns `worker` in the background; `worker` runs evolve() and
        calls emit() per generation, then emits None to close the window when training ends.
        """

        def __init__(self):
            self._q: "queue.Queue" = queue.Queue()
            self._win = None
            self._app = None

        def emit(self, event) -> None:
            """Thread-safe: called from the worker thread; the main-thread timer drains it."""
            self._q.put(event)

        def run(self, worker_callable) -> None:
            import sys
            self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
            self._win = _EvoWindow()
            self._win.show()
            worker = threading.Thread(target=worker_callable, daemon=True)
            worker.start()
            timer = QtCore.QTimer()
            timer.timeout.connect(self._drain)
            timer.start(150)
            self._app.exec()                       # blocks the main thread (PyQt6: exec())
            worker.join(timeout=5.0)

        def _drain(self) -> None:
            while not self._q.empty():
                ev = self._q.get_nowait()
                if ev is None:                     # sentinel: training finished
                    self._app.quit()
                    return
                if self._win:
                    self._win.post(ev)

else:
    EvoMonitor = _NoOpEvoMonitor  # type: ignore[misc]
