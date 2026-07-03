"""run_replay_panel.py — bar-by-bar playback of a recorded training run (Vision M3).

Loads a ``gym/runs/<run_id>/`` recording (see runlog.py) and replays it:

  * equity pane — the selected checkpoint's equity revealed bar-by-bar (green), the
    equal-weight benchmark (amber dashed), and earlier checkpoints as grey ghosts so
    improvement across training is visible at a glance;
  * weights pane — the portfolio weight per asset at every bar (the agent's decisions);
  * skill pane — final equity per checkpoint ("generations"), current one highlighted.

Controls: a **checkpoint scrubber** (slide through training progress), a bar cursor,
play/pause with speed, all without retraining anything.

Standalone:  .venv/Scripts/python gym/run_replay_panel.py [run_id]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from runlog import RunRecord, list_runs   # noqa: E402


# ─────────────────────────────────────────── pure playback logic (tested headlessly)

def playback_frame(rec: RunRecord, ci: int, t: int) -> dict:
    """Everything the panel needs to draw checkpoint ``ci`` revealed up to bar ``t``."""
    ci = int(np.clip(ci, 0, rec.n_checkpoints - 1))
    t = int(np.clip(t, 0, rec.n_bars - 1))
    return {
        "ci": ci, "t": t,
        "step": int(rec.steps[ci]),
        "date": str(rec.dates[t]),
        "equity": rec.equity[ci, :t + 1],
        "bench": rec.bench_equity[:t + 1],
        "weights": rec.weights[ci, :t + 1],          # (t+1, N)
        "ret": float(rec.returns[ci, t]),
        "turnover": float(rec.turnover[ci, t]),
        "final_equity": float(rec.equity[ci, -1]),
        "skill": rec.equity[:, -1],                  # final equity per checkpoint
    }


def weight_stack(weights: np.ndarray) -> np.ndarray:
    """(T, N) weights → (T, N+1) stacked-area bounds: column k is the cumulative weight of
    the first k assets, so asset k's band lives between columns k and k+1."""
    T, N = weights.shape
    out = np.zeros((T, N + 1))
    out[:, 1:] = np.cumsum(weights, axis=1)
    return out


def regime_spans(labels) -> list[tuple[int, int, str]]:
    """Contiguous [start, end) runs of equal labels, for background shading."""
    spans: list[tuple[int, int, str]] = []
    if len(labels) == 0:
        return spans
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            spans.append((start, i, str(labels[start])))
            start = i
    return spans


# ─────────────────────────────────────────── Qt panel

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
import pyqtgraph as pg                                # noqa: E402
from pyqtgraph.Qt import QtCore, QtWidgets            # noqa: E402

BG, FG = "#0b0e11", "#c9d1d9"
GREEN, AMBER, GREY = "#00d26a", "#ffb000", "#3a4149"
ASSET_COLORS = ["#00d26a", "#ffb000", "#58a6ff", "#f85149", "#d2a8ff",
                "#76e3ea", "#f0883e", "#7ee787", "#ff7b72", "#a5d6ff"]
_DASH = QtCore.Qt.PenStyle.DashLine if hasattr(QtCore.Qt, "PenStyle") else QtCore.Qt.DashLine


class RunReplayWidget(QtWidgets.QWidget):
    """3-pane scrubber/playback for one RunRecord."""

    def __init__(self, rec: RunRecord, parent=None):
        super().__init__(parent)
        self.rec = rec
        self.ci = rec.n_checkpoints - 1
        self.t = rec.n_bars - 1
        self._speed = 4

        pg.setConfigOption("background", BG)
        pg.setConfigOption("foreground", FG)
        lay = QtWidgets.QVBoxLayout(self)

        m = rec.manifest
        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet(
            f"color: {AMBER}; font-family: Consolas, monospace; font-size: 12px;")
        self.info.setMinimumHeight(20)
        lay.addWidget(self.info)

        pw = pg.GraphicsLayoutWidget()
        lay.addWidget(pw, stretch=1)
        self.p_eq = pw.addPlot(row=0, col=0)
        self.p_eq.setLabel("left", "equity (×)")
        self.p_eq.showGrid(x=True, y=True, alpha=0.15)
        self.p_w = pw.addPlot(row=1, col=0)
        self.p_w.setLabel("left", "weights")
        self.p_w.setYRange(0, 1)
        self.p_w.showGrid(x=True, y=True, alpha=0.15)
        self.p_skill = pw.addPlot(row=2, col=0)
        self.p_skill.setLabel("left", "final equity")
        self.p_skill.setLabel("bottom", "training steps at checkpoint")
        pw.ci.layout.setRowStretchFactor(0, 3)
        pw.ci.layout.setRowStretchFactor(1, 2)
        pw.ci.layout.setRowStretchFactor(2, 1)

        # regime shading behind the equity curves (causal labels from the benchmark)
        self.regime_labels = None
        if rec.n_bars >= 70:
            from regime import label_regimes
            self.regime_labels = label_regimes(rec.bench_ret)
            shade = {"bull": (0, 210, 106, 20), "bear": (248, 81, 73, 22),
                     "choppy": (120, 130, 140, 10)}
            for s, e, lab in regime_spans(self.regime_labels):
                reg = pg.LinearRegionItem(values=(s, e), movable=False,
                                          brush=pg.mkBrush(*shade[lab]),
                                          pen=pg.mkPen((0, 0, 0, 0)))
                reg.setZValue(-10)
                self.p_eq.addItem(reg)

        # static content: ghosts + benchmark + skill curve
        eq_legend = self.p_eq.addLegend(offset=(4, 4))
        eq_legend.setLabelTextColor(FG)
        for i in range(rec.n_checkpoints):
            self.p_eq.plot(rec.equity[i], pen=pg.mkPen(GREY, width=1))
        self.p_eq.plot(rec.bench_equity, pen=pg.mkPen(AMBER, width=1, style=_DASH),
                       name="equal-weight")
        self.cur_eq = self.p_eq.plot(pen=pg.mkPen(GREEN, width=2), name="agent")

        # stacked-area weights: N+1 boundary curves with fills between neighbours
        n_assets = len(rec.tics)
        self.w_bounds = [self.p_w.plot(pen=pg.mkPen((0, 0, 0, 0)))]      # floor at 0
        w_legend = self.p_w.addLegend(offset=(4, 4))
        w_legend.setLabelTextColor(FG)
        for i, tic in enumerate(rec.tics):
            color = ASSET_COLORS[i % len(ASSET_COLORS)]
            self.w_bounds.append(self.p_w.plot(pen=pg.mkPen(color, width=1), name=tic))
            fill_color = pg.mkColor(color)
            fill_color.setAlpha(90)
            self.p_w.addItem(pg.FillBetweenItem(self.w_bounds[i], self.w_bounds[i + 1],
                                                brush=pg.mkBrush(fill_color)))
        self.p_skill.plot(rec.steps, rec.equity[:, -1], pen=pg.mkPen(FG, width=1),
                          symbol="o", symbolSize=5, symbolBrush=GREY)
        self.skill_marker = self.p_skill.plot([], [], pen=None, symbol="o",
                                              symbolSize=9, symbolBrush=GREEN)

        # controls
        ctl = QtWidgets.QHBoxLayout()
        lay.addLayout(ctl)
        self.btn = QtWidgets.QPushButton("Play")
        self.btn.clicked.connect(self._toggle)
        ctl.addWidget(self.btn)
        ctl.addWidget(QtWidgets.QLabel("speed"))
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(["1×", "4×", "16×"])
        self.speed.setCurrentIndex(1)
        self.speed.currentIndexChanged.connect(
            lambda i: setattr(self, "_speed", [1, 4, 16][i]))
        ctl.addWidget(self.speed)
        ctl.addWidget(QtWidgets.QLabel("checkpoint"))
        self.s_ci = QtWidgets.QSlider(QtCore.Qt.Horizontal if not hasattr(QtCore.Qt, "Orientation")
                                      else QtCore.Qt.Orientation.Horizontal)
        self.s_ci.setMaximum(rec.n_checkpoints - 1)
        self.s_ci.setValue(self.ci)
        self.s_ci.valueChanged.connect(self.set_checkpoint)
        ctl.addWidget(self.s_ci, stretch=1)
        ctl.addWidget(QtWidgets.QLabel("bar"))
        self.s_t = QtWidgets.QSlider(QtCore.Qt.Horizontal if not hasattr(QtCore.Qt, "Orientation")
                                     else QtCore.Qt.Orientation.Horizontal)
        self.s_t.setMaximum(rec.n_bars - 1)
        self.s_t.setValue(self.t)
        self.s_t.valueChanged.connect(self.set_bar)
        ctl.addWidget(self.s_t, stretch=2)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self._redraw()

    # ---- state
    def set_checkpoint(self, ci: int):
        self.ci = int(np.clip(ci, 0, self.rec.n_checkpoints - 1))
        self._redraw()

    def set_bar(self, t: int):
        self.t = int(np.clip(t, 0, self.rec.n_bars - 1))
        self._redraw()

    def _toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn.setText("Play")
        else:
            if self.t >= self.rec.n_bars - 1:
                self.t = 0
            self.timer.start()
            self.btn.setText("Pause")

    def _tick(self):
        self.t = min(self.t + self._speed, self.rec.n_bars - 1)
        self.s_t.blockSignals(True)
        self.s_t.setValue(self.t)
        self.s_t.blockSignals(False)
        self._redraw()
        if self.t >= self.rec.n_bars - 1:
            self._toggle()

    def _redraw(self):
        f = playback_frame(self.rec, self.ci, self.t)
        x = np.arange(f["t"] + 1)
        self.cur_eq.setData(x, f["equity"])
        stack = weight_stack(f["weights"])
        for k, bound in enumerate(self.w_bounds):
            bound.setData(x, stack[:, k])
        self.skill_marker.setData([self.rec.steps[f["ci"]]], [f["final_equity"]])
        m = self.rec.manifest
        regime = (f"   regime {self.regime_labels[f['t']]}"
                  if self.regime_labels is not None else "")
        self.info.setText(
            f"run {m['run_id']}   algo={m['algo']} reward={m['reward']}   "
            f"checkpoint {f['ci'] + 1}/{self.rec.n_checkpoints} @ {f['step']} steps   "
            f"bar {f['t'] + 1}/{self.rec.n_bars} ({f['date']})   "
            f"equity {f['equity'][-1]:.4f}  final {f['final_equity']:.4f}{regime}")


def main(run_id: str | None = None):
    runs = list_runs()
    if run_id is None:
        if not runs:
            sys.exit("No recorded runs under gym/runs/. Train with a RunRecorder first.")
        run_id = runs[0]
    rec = RunRecord.load(run_id)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from pyqtgraph.Qt import QtGui
    app.setFont(QtGui.QFont("Consolas", 9))
    w = RunReplayWidget(rec)
    w.setWindowTitle(f"Vision — run replay [{run_id}]")
    w.resize(1100, 800)
    w.setStyleSheet(f"background-color: {BG}; color: {FG};")
    w.show()
    app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
