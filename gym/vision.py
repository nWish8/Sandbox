"""vision.py — the Vision terminal shell (M5): one dark window, five modules.

    ┌ VISION ────────────────────────────────────────────────┐
    │ [Monitor] [Screener] [RL Lab] [Replay]                  │
    │  … active module …                                      │
    └─────────────────────────────────────────────────────────┘

  * **Monitor** — READ-ONLY portfolio tracking from ``gym/portfolio_state.json``
    (positions you edit by hand): live-ish prices via the cached marketdata layer,
    P&L, weights, and a 60-day return-correlation heatmap. No order routing, ever —
    the project is research-only.
  * **Screener** — the technical snapshot table + quantile return projections.
  * **RL Lab** — the existing FinRL control panel, embedded whole.
  * **Replay** — bar-by-bar playback of recorded training runs (checkpoint scrubber).

Theme: dark, high-contrast, monospace, terminal green/amber. No light mode.

Run:  .venv/Scripts/python gym/vision.py        (or: python -m gym.run vision)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from marketdata import MarketData                     # noqa: E402

STATE_FILE = HERE / "portfolio_state.json"

BG, PANEL, FG = "#0b0e11", "#11161c", "#c9d1d9"
GREEN, AMBER, RED, GREY = "#00d26a", "#ffb000", "#f85149", "#3a4149"

THEME_QSS = f"""
* {{ font-family: Consolas, 'Cascadia Mono', monospace; font-size: 12px; }}
QMainWindow, QWidget {{ background-color: {BG}; color: {FG}; }}
QTabWidget::pane {{ border: 1px solid {GREY}; }}
QTabBar::tab {{ background: {PANEL}; color: {FG}; padding: 6px 14px; border: 1px solid {GREY}; }}
QTabBar::tab:selected {{ color: {GREEN}; border-bottom: 2px solid {GREEN}; }}
QPushButton {{ background: {PANEL}; border: 1px solid {GREY}; padding: 5px 12px; color: {AMBER}; }}
QPushButton:hover {{ border-color: {AMBER}; }}
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {{ background: {PANEL}; border: 1px solid {GREY};
    color: {FG}; selection-background-color: {GREY}; }}
QTableWidget {{ background: {PANEL}; gridline-color: {GREY}; }}
QHeaderView::section {{ background: {BG}; color: {AMBER}; border: 1px solid {GREY}; padding: 4px; }}
QSlider::groove:horizontal {{ background: {GREY}; height: 4px; }}
QSlider::handle:horizontal {{ background: {GREEN}; width: 10px; margin: -4px 0; }}
QLabel#title {{ color: {GREEN}; font-size: 14px; font-weight: bold; }}
"""


# ─────────────────────────────────────────── pure logic (unit-tested)

def load_portfolio_state(path: Path | str = STATE_FILE) -> list[dict]:
    """Positions from the hand-edited JSON: [{tic, shares, cost_basis}, ...]."""
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("positions", [])


def build_positions_table(positions: list[dict], last: dict, prev: dict) -> dict:
    """Rows + totals for the monitor. ``last``/``prev`` map tic -> price (prev = the bar
    before last, for day change). Pure math — no I/O."""
    rows, tv, tc = [], 0.0, 0.0
    for p in positions:
        tic, sh, cb = p["tic"], float(p["shares"]), float(p["cost_basis"])
        px = float(last.get(tic, float("nan")))
        value = sh * px
        cost = sh * cb
        day = (px / float(prev[tic]) - 1.0) if tic in prev and prev[tic] else float("nan")
        rows.append({"tic": tic, "shares": sh, "cost_basis": cb, "last": px,
                     "value": value, "pnl": value - cost,
                     "pnl_pct": (value / cost - 1.0) if cost else float("nan"),
                     "day_pct": day})
        if np.isfinite(value):
            tv += value
            tc += cost
    for r in rows:
        r["weight"] = (r["value"] / tv) if tv and np.isfinite(r["value"]) else float("nan")
    return {"rows": rows, "total_value": tv, "total_cost": tc,
            "total_pnl": tv - tc,
            "total_pnl_pct": (tv / tc - 1.0) if tc else float("nan")}


def returns_correlation(md: MarketData, tics: list[str], end: str | None = None,
                        days: int = 90) -> pd.DataFrame:
    """Correlation matrix of daily returns over a trailing window (for the heatmap)."""
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start = (end_ts - pd.Timedelta(days=days + 40)).strftime("%Y-%m-%d")
    closes = {}
    for tic in tics:
        try:
            closes[tic] = md.get_ohlcv(tic, start, end_ts.strftime("%Y-%m-%d"))["close"]
        except ValueError:
            continue
    px = pd.DataFrame(closes).dropna()
    return px.pct_change().dropna().tail(days).corr()


# ─────────────────────────────────────────── Qt shell

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
import pyqtgraph as pg                                 # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets      # noqa: E402

pg.setConfigOption("background", BG)
pg.setConfigOption("foreground", FG)


def _fmt_pct(x, signed=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100 * x:+.2f}%" if signed else f"{100 * x:.2f}%"


class MonitorTab(QtWidgets.QWidget):
    """Read-only positions + P&L + correlation heatmap. No order routing."""

    def __init__(self, md: MarketData | None = None, parent=None):
        super().__init__(parent)
        self.md = md or MarketData()
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("PORTFOLIO MONITOR  ·  read-only")
        title.setObjectName("title")
        top.addWidget(title)
        top.addStretch(1)
        self.status = QtWidgets.QLabel("")
        top.addWidget(self.status)
        btn = QtWidgets.QPushButton("Refresh")
        btn.clicked.connect(self.refresh)
        top.addWidget(btn)
        lay.addLayout(top)

        split = QtWidgets.QHBoxLayout()
        lay.addLayout(split, stretch=1)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        split.addWidget(self.table, stretch=3)

        right = QtWidgets.QVBoxLayout()
        split.addLayout(right, stretch=2)
        right.addWidget(QtWidgets.QLabel("60-day return correlation"))
        self.heat = pg.GraphicsLayoutWidget()
        self.heat_plot = self.heat.addPlot()
        self.heat_plot.setAspectLocked(True)
        self.img = pg.ImageItem()
        cmap = pg.ColorMap([0.0, 0.5, 1.0],
                           [(248, 81, 73), (11, 14, 17), (0, 210, 106)])
        self.img.setLookupTable(cmap.getLookupTable(nPts=256))
        self.heat_plot.addItem(self.img)
        right.addWidget(self.heat, stretch=1)

    def refresh(self):
        positions = load_portfolio_state()
        if not positions:
            self.status.setText(f"no positions — edit {STATE_FILE.name}")
            return
        tics = [p["tic"] for p in positions]
        last, prev = {}, {}
        end = pd.Timestamp.today().normalize()
        start = (end - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        for tic in tics:
            try:
                c = self.md.get_ohlcv(tic, start, end.strftime("%Y-%m-%d"))["close"]
                last[tic] = float(c.iloc[-1])
                if len(c) > 1:
                    prev[tic] = float(c.iloc[-2])
            except ValueError:
                pass
        t = build_positions_table(positions, last, prev)

        cols = ["tic", "shares", "cost", "last", "value", "day %", "P&L $", "P&L %", "weight"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(t["rows"]) + 1)
        for i, r in enumerate(t["rows"]):
            vals = [r["tic"], f"{r['shares']:g}", f"{r['cost_basis']:.2f}",
                    f"{r['last']:.2f}", f"{r['value']:,.0f}", _fmt_pct(r["day_pct"]),
                    f"{r['pnl']:+,.0f}", _fmt_pct(r["pnl_pct"]),
                    _fmt_pct(r["weight"], signed=False)]
            for j, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v)
                if j in (5, 6, 7):
                    good = not str(v).startswith("-")
                    item.setForeground(QtGui.QColor(GREEN if good else RED))
                self.table.setItem(i, j, item)
        totals = ["TOTAL", "", "", "", f"{t['total_value']:,.0f}", "",
                  f"{t['total_pnl']:+,.0f}", _fmt_pct(t["total_pnl_pct"]), "100%"]
        for j, v in enumerate(totals):
            item = QtWidgets.QTableWidgetItem(v)
            item.setForeground(QtGui.QColor(AMBER))
            self.table.setItem(len(t["rows"]), j, item)
        self.table.resizeColumnsToContents()

        corr = returns_correlation(self.md, tics)
        if len(corr):
            self.img.setImage(corr.to_numpy(), levels=(-1.0, 1.0))
            ticks = [(i + 0.5, tic) for i, tic in enumerate(corr.columns)]
            self.heat_plot.getAxis("bottom").setTicks([ticks])
            self.heat_plot.getAxis("left").setTicks([ticks])
        self.status.setText(f"{len(t['rows'])} positions · value {t['total_value']:,.0f}")


class ScreenerTab(QtWidgets.QWidget):
    """Technical screen + quantile projections, rendered as terminal tables."""

    def __init__(self, md: MarketData | None = None, parent=None):
        super().__init__(parent)
        self.md = md or MarketData()
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("SCREENER & PROJECTIONS")
        title.setObjectName("title")
        row.addWidget(title)
        row.addWidget(QtWidgets.QLabel("  tickers:"))
        self.edit = QtWidgets.QLineEdit("AAPL MSFT JPM XOM CAT PG JNJ WMT")
        row.addWidget(self.edit, stretch=1)
        b1 = QtWidgets.QPushButton("Screen")
        b1.clicked.connect(self.run_screen)
        row.addWidget(b1)
        b2 = QtWidgets.QPushButton("Project 20d")
        b2.clicked.connect(self.run_project)
        row.addWidget(b2)
        lay.addLayout(row)
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        lay.addWidget(self.out, stretch=1)

    def _tics(self):
        return [t for t in self.edit.text().replace(",", " ").split() if t]

    def run_screen(self):
        from screener import format_screen, screen
        self.out.appendPlainText("\n$ screen " + " ".join(self._tics()))
        try:
            table = screen(self._tics(), md=self.md, log=lambda m: self.out.appendPlainText(m))
            self.out.appendPlainText(format_screen(table))
        except Exception as e:  # noqa: BLE001
            self.out.appendPlainText(f"error: {e}")

    def run_project(self):
        from projections import format_projection, project
        self.out.appendPlainText("\n$ project --horizon 20 " + " ".join(self._tics()))
        try:
            table, honesty = project(self._tics(), horizon=20, md=self.md,
                                     log=lambda m: self.out.appendPlainText(m))
            self.out.appendPlainText(format_projection(table, honesty, 20))
        except Exception as e:  # noqa: BLE001
            self.out.appendPlainText(f"error: {e}")


class ReplayTab(QtWidgets.QWidget):
    """Pick a recorded run, embed the scrubber panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from runlog import list_runs
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("RUN REPLAY")
        title.setObjectName("title")
        row.addWidget(title)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(list_runs() or ["<no recorded runs>"])
        row.addWidget(self.combo, stretch=1)
        b = QtWidgets.QPushButton("Load")
        b.clicked.connect(self.load_run)
        row.addWidget(b)
        lay.addLayout(row)
        self.holder = QtWidgets.QVBoxLayout()
        lay.addLayout(self.holder, stretch=1)
        self.panel = None
        if list_runs():
            self.load_run()

    def load_run(self):
        from run_replay_panel import RunReplayWidget
        from runlog import RunRecord, list_runs
        if not list_runs():
            return
        if self.panel is not None:
            self.holder.removeWidget(self.panel)
            self.panel.deleteLater()
        self.panel = RunReplayWidget(RunRecord.load(self.combo.currentText()))
        self.holder.addWidget(self.panel)


class VisionWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISION — trading intelligence terminal (research only)")
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)
        md = MarketData()
        self.monitor = MonitorTab(md)
        tabs.addTab(self.monitor, "Monitor")
        tabs.addTab(ScreenerTab(md), "Screener")
        try:
            from control_panel import ControlPanel
            tabs.addTab(ControlPanel(), "RL Lab")
        except Exception as e:  # noqa: BLE001 — lab is optional if FinRL env is broken
            err = QtWidgets.QLabel(f"RL Lab unavailable: {e}")
            tabs.addTab(err, "RL Lab")
        tabs.addTab(ReplayTab(), "Replay")
        self.resize(1280, 860)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(THEME_QSS)
    app.setFont(QtGui.QFont("Consolas", 9))   # pyqtgraph axis text inherits this too
    w = VisionWindow()
    w.show()
    app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    main()
