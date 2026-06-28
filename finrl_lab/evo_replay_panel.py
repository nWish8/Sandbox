"""evo_replay_panel.py — 3-pane stepped replay of a population-evolution run.

Plays back evo_portfolio_record.npz, revealing one candle at a time:

  A · Tickers      — every basket member as a rebased line (start = 1.0), overlaid.
  B · Allocation   — the champion's weights as a STACKED AREA over time (each colour band
                     a ticker/cash; band thickness = capital there; total = 1.0).
  C · Portfolios   — every agent's portfolio value (faint), champion bold, equal-weight
                     buy & hold dashed; labelled dashed verticals for train|val|test.

X-axis is real dates. Controls: Play/Pause, a speed slider (ms/candle), "To end".

Headless snapshot:  QT_QPA_PLATFORM=offscreen python evo_replay_panel.py --shot
  → writes PNGs of a few frames so the chart can be inspected without a display.

Run (live):  .venv/Scripts/python finrl_lab/evo_replay_panel.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

pg.setConfigOptions(antialias=True, background="k", foreground="w")   # high-contrast axes/text

HERE = Path(__file__).resolve().parent
RECORD = HERE / "evo_portfolio_record.npz"

FAST_MS, SLOW_MS, DEFAULT_MS = 10, 400, 40      # speed-slider range + default


def _load(path=RECORD):
    d = np.load(path, allow_pickle=True)
    return {
        "dates": d["dates"].astype(np.int64).astype(float),   # epoch seconds (S,)
        "close": d["close"].astype(float),                    # (S, N)
        "values": d["values"].astype(float),                  # (P, S)
        "weights": d["champ_weights"].astype(float),          # (S, N+1)
        "champ_idx": int(d["champ_idx"]),
        "tics": [str(t) for t in d["tics"]],
        "split": [int(x) for x in d["split"]],
    }


def _date_axis():
    ax = pg.DateAxisItem(orientation="bottom")
    return ax


class ReplayWindow(QtWidgets.QMainWindow):
    def __init__(self, rec: dict, interval_ms: int = DEFAULT_MS):
        super().__init__()
        self.setWindowTitle("Signal Gym — population replay (DOW-30)")
        self.resize(1240, 920)
        self.rec = rec
        self.dates = rec["dates"]
        self.close_r = rec["close"] / rec["close"][0]              # (S, N)
        self.values = rec["values"]                                # (P, S)
        self.weights = rec["weights"]                              # (S, N+1)
        self.cum = np.cumsum(self.weights, axis=1)                 # (S, N+1) stacked tops
        self.eqw = self.close_r.mean(axis=1)                       # equal-weight B&H (S,)
        self.S, self.N = self.close_r.shape
        self.labels = rec["tics"] + ["CASH"]
        self.champ = rec["champ_idx"]
        self.t = 2

        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)
        v.addLayout(self._controls())
        gl = pg.GraphicsLayoutWidget(); v.addWidget(gl, 1)

        # ── Pane A: tickers overlaid
        self.pA = gl.addPlot(row=0, col=0, title="A · Tickers (rebased to 1.0)",
                             axisItems={"bottom": _date_axis()})
        self.pA.showGrid(x=True, y=True, alpha=0.15)
        self.tick_curves = [self.pA.plot([], [], pen=pg.mkPen(pg.intColor(i, self.N), width=1))
                            for i in range(self.N)]

        # ── Pane B: champion allocation as a stacked area
        self.pB = gl.addPlot(row=1, col=0, title="B · Champion allocation (stacked = 100% of capital)",
                             axisItems={"bottom": _date_axis()})
        self.pB.setXLink(self.pA); self.pB.setYRange(0, 1)
        self.pB.addLegend(offset=(10, 10), labelTextSize="7pt", colCount=4)
        self._base = self.pB.plot([], [], pen=None)                # zero baseline
        self.cum_curves, self.fills = [], []
        for k in range(self.N + 1):
            col = pg.intColor(k, self.N + 1)
            cur = self.pB.plot([], [], pen=pg.mkPen(col, width=0.6), name=self.labels[k])
            lower = self._base if k == 0 else self.cum_curves[k - 1]
            fb = pg.FillBetweenItem(cur, lower, brush=pg.mkBrush(col.red(), col.green(), col.blue(), 150))
            self.pB.addItem(fb)
            self.cum_curves.append(cur); self.fills.append(fb)

        # ── Pane C: portfolios (faint agents as one item, champion bold, B&H dashed)
        self.pC = gl.addPlot(row=2, col=0,
                             title="C · Portfolio value — agents (grey), champion (blue), buy&hold (orange dash)",
                             axisItems={"bottom": _date_axis()})
        self.pC.setXLink(self.pA); self.pC.showGrid(x=True, y=True, alpha=0.15)
        self.agents_item = pg.PlotCurveItem(pen=pg.mkPen((150, 155, 160, 60), width=1), connect="finite")
        self.pC.addItem(self.agents_item)
        self._others = [i for i in range(self.values.shape[0]) if i != self.champ]
        self.champ_curve = self.pC.plot([], [], pen=pg.mkPen("#2962ff", width=3), name="champion")
        self.bh_curve = self.pC.plot([], [], pen=pg.mkPen("orange", width=1.7, style=QtCore.Qt.DashLine),
                                     name="buy & hold")

        # ── labelled split markers on every pane
        tr, va = self.rec["split"]
        marks = [(tr, "#9aa0a6", "validation ▶"), (va, "#c0392b", "test ▶")]
        for p in (self.pA, self.pB, self.pC):
            for idx, col, txt in marks:
                ln = pg.InfiniteLine(pos=self.dates[idx], angle=90,
                                     pen=pg.mkPen(col, width=1.4, style=QtCore.Qt.DashLine))
                if p is self.pA:
                    ln.label = pg.InfLineLabel(ln, txt, position=0.92, color=col,
                                               movable=False, fill=(0, 0, 0, 120))
                p.addItem(ln)

        # dates only on the bottom pane (cleaner), readable font + room for labels
        for p in (self.pA, self.pB):
            p.getAxis("bottom").setStyle(showValues=False)
        self.pC.getAxis("bottom").setStyle(tickFont=QtGui.QFont("", 9))
        self.pC.getAxis("bottom").setHeight(34)

        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self._tick)
        self.timer.start(interval_ms)
        self._redraw()

    # ── controls
    def _controls(self):
        h = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Pause"); self.btn_play.clicked.connect(self._toggle)
        h.addWidget(self.btn_play)
        h.addWidget(QtWidgets.QLabel("fast"))
        self.speed = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.speed.setRange(FAST_MS, SLOW_MS)
        self.speed.setValue(DEFAULT_MS); self.speed.setFixedWidth(200)
        self.speed.valueChanged.connect(lambda v: self.timer.setInterval(v)); h.addWidget(self.speed)
        h.addWidget(QtWidgets.QLabel("slow"))
        self.btn_end = QtWidgets.QPushButton("To end"); self.btn_end.clicked.connect(self._to_end)
        h.addWidget(self.btn_end)
        self.lbl = QtWidgets.QLabel(""); h.addWidget(self.lbl); h.addStretch(1)
        return h

    def _toggle(self):
        if self.timer.isActive():
            self.timer.stop(); self.btn_play.setText("Play")
        else:
            self.timer.start(self.speed.value()); self.btn_play.setText("Pause")

    def _to_end(self):
        self.t = self.S; self._redraw()

    def _tick(self):
        if self.t >= self.S:
            self.timer.stop(); self.btn_play.setText("Play"); return
        self.t += 1; self._redraw()

    # ── draw frame
    def _redraw(self):
        t = max(2, self.t); x = self.dates[:t]
        for i, c in enumerate(self.tick_curves):
            c.setData(x, self.close_r[:t, i])
        self._base.setData(x, np.zeros(t))
        for k, cur in enumerate(self.cum_curves):
            cur.setData(x, self.cum[:t, k])
        # faint agents as one NaN-separated polyline
        P = len(self._others)
        xb = np.full((P, t + 1), np.nan); xb[:, :t] = x
        yb = np.full((P, t + 1), np.nan); yb[:, :t] = self.values[self._others, :t]
        self.agents_item.setData(xb.ravel(), yb.ravel())
        self.champ_curve.setData(x, self.values[self.champ, :t])
        self.bh_curve.setData(x, self.eqw[:t])
        import datetime as _dt
        when = _dt.datetime.utcfromtimestamp(self.dates[t - 1]).strftime("%Y-%m-%d")
        self.lbl.setText(f"{when}   bar {t}/{self.S}   champ PV={self.values[self.champ, t-1]:.2f}   "
                         f"B&H={self.eqw[t-1]:.2f}")


def render_snapshots(frames=None, prefix="snap", size=(1280, 940)):
    """Render the window to PNGs at a few frames (offscreen) for visual inspection."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = ReplayWindow(_load(), interval_ms=DEFAULT_MS); win.timer.stop()
    win.resize(*size); win.show(); app.processEvents()
    S = win.S
    frames = frames or [int(S * 0.45), int(S * 0.8), S]
    out = []
    for f in frames:
        win.t = f; win._redraw()
        for _ in range(3):
            app.processEvents()
        p = HERE / f"{prefix}_{f}.png"
        win.grab().save(str(p))
        out.append(p)
    return out


def main():
    if not RECORD.exists():
        sys.exit(f"No recording at {RECORD}. Run: .venv/Scripts/python finrl_lab/evo_portfolio.py")
    if "--shot" in sys.argv:
        for p in render_snapshots():
            print("snapshot:", p)
        return
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = ReplayWindow(_load()); win.show()
    app.exec_()


if __name__ == "__main__":
    main()
