"""control_panel.py — PyQt desktop panel to configure, launch, watch and stop FinRL runs.

A single window:
  * left  — configuration form (ticker preset/list, dates, agent, timesteps, device, n_envs)
  * right — live training log, progress bar, a reward-vs-timesteps plot, and (after a run)
            the backtest equity curve.

Training runs on a background thread; the pipeline streams log lines and progress through
thread-safe Qt signals. Stop cancels mid-training via a stop flag the SB3 callback checks.

Run:  .venv/Scripts/python finrl_lab/control_panel.py
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# Both PyQt5 and PyQt6 exist in this venv; force pyqtgraph onto the clean PyQt5 binding
# and import Qt THROUGH pyqtgraph so everything matches (mixing bindings segfaults).
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import finrl_patch  # noqa: F401,E402
from finrl.config import INDICATORS  # noqa: E402
from pipeline import AGENTS, FinRLConfig, backtest, prepare_data, train  # noqa: E402


def _our_basket() -> list[str]:
    """Yahoo symbols from the gym project's ticker manifest, if present."""
    f = HERE.parent / "gym" / "config" / "tickers.txt"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _dow30() -> list[str]:
    try:
        from finrl import config_tickers
        return list(config_tickers.DOW_30_TICKER)
    except Exception:  # noqa: BLE001
        return ["AAPL", "MSFT", "JPM", "XOM", "CAT", "BA", "CVX", "KO"]


PRESETS = {"DOW-30": _dow30, "Our basket": _our_basket,
           "Quick (AAPL/MSFT/JPM)": lambda: ["AAPL", "MSFT", "JPM"]}


class Bridge(QtCore.QObject):
    """Thread-safe signal hub: worker thread emits, GUI thread receives."""
    log = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int, float)   # steps, total, reward (nan if n/a)
    equity = QtCore.pyqtSignal(list)
    done = QtCore.pyqtSignal(str)
    # evolution tab
    evo_log = QtCore.pyqtSignal(str)
    evo_gen = QtCore.pyqtSignal(int, float, float, float)   # gen, best_train, mean_train, val
    evo_done = QtCore.pyqtSignal(str)


class ControlPanel(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Gym — Control Panel")
        self.resize(1220, 820)
        self.bridge = Bridge()
        self.bridge.log.connect(self._on_log)
        self.bridge.progress.connect(self._on_progress)
        self.bridge.equity.connect(self._on_equity)
        self.bridge.done.connect(self._on_done)
        self.bridge.evo_log.connect(lambda s: self.evo_logbox.appendPlainText(s))
        self.bridge.evo_gen.connect(self._on_evo_gen)
        self.bridge.evo_done.connect(self._on_evo_done)
        self._stop = threading.Event()
        self._evo_stop = threading.Event()
        self._reward_x: list[int] = []
        self._reward_y: list[float] = []
        self._evo_best: list[float] = []
        self._evo_mean: list[float] = []
        self._evo_val: list[float] = []
        self._replay_win = None

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)
        finrl_tab = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(finrl_tab)
        fl.addLayout(self._build_form(), 0)
        fl.addLayout(self._build_view(), 1)
        tabs.addTab(finrl_tab, "FinRL DRL (single agent)")
        tabs.addTab(self._build_evo_tab(), "Evolution (GPU population)")

    # ---- left: config form
    def _build_form(self):
        form = QtWidgets.QFormLayout()

        self.preset = QtWidgets.QComboBox(); self.preset.addItems(PRESETS.keys())
        self.preset.currentTextChanged.connect(self._apply_preset)
        form.addRow("Preset", self.preset)

        self.tickers = QtWidgets.QPlainTextEdit(); self.tickers.setFixedHeight(70)
        form.addRow("Tickers", self.tickers)

        self.train_start = QtWidgets.QLineEdit("2014-01-01")
        self.train_end = QtWidgets.QLineEdit("2022-01-01")
        self.trade_start = QtWidgets.QLineEdit("2022-01-01")
        self.trade_end = QtWidgets.QLineEdit("2024-01-01")
        for lbl, w in (("Train start", self.train_start), ("Train end", self.train_end),
                       ("Trade start", self.trade_start), ("Trade end", self.trade_end)):
            form.addRow(lbl, w)

        self.agent = QtWidgets.QComboBox(); self.agent.addItems(AGENTS)
        self.agent.setCurrentText("ppo"); form.addRow("Agent", self.agent)

        self.timesteps = QtWidgets.QSpinBox()
        self.timesteps.setRange(1000, 5_000_000); self.timesteps.setSingleStep(5000)
        self.timesteps.setValue(20000); form.addRow("Timesteps", self.timesteps)

        self.device = QtWidgets.QComboBox(); self.device.addItems(["cpu", "cuda"])
        form.addRow("Device", self.device)

        self.n_envs = QtWidgets.QSpinBox(); self.n_envs.setRange(1, 16)
        form.addRow("Parallel envs", self.n_envs)

        self.btn_run = QtWidgets.QPushButton("Run (data → train → backtest)")
        self.btn_run.clicked.connect(self._start)
        self.btn_stop = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self._stop.set())
        form.addRow(self.btn_run); form.addRow(self.btn_stop)

        self._apply_preset(self.preset.currentText())
        wrap = QtWidgets.QVBoxLayout(); wrap.addLayout(form); wrap.addStretch(1)
        return wrap

    # ---- right: log + plots
    def _build_view(self):
        v = QtWidgets.QVBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        v.addWidget(self.progress)

        self.reward_plot = pg.PlotWidget(title="Training — mean episode reward")
        self.reward_plot.setLabel("bottom", "timesteps")
        self.reward_curve = self.reward_plot.plot([], [], pen=pg.mkPen("limegreen", width=2),
                                                  symbol="o", symbolSize=5)
        v.addWidget(self.reward_plot, 2)

        self.equity_plot = pg.PlotWidget(title="Backtest — portfolio value")
        self.equity_plot.setLabel("bottom", "step")
        self.equity_curve = self.equity_plot.plot([], [], pen=pg.mkPen("cyan", width=2))
        v.addWidget(self.equity_plot, 2)

        self.logbox = QtWidgets.QPlainTextEdit(); self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(2000)
        v.addWidget(self.logbox, 1)
        return v

    # ---- config
    def _apply_preset(self, name):
        syms = PRESETS.get(name, lambda: [])()
        self.tickers.setPlainText(", ".join(syms))

    def _collect_cfg(self) -> FinRLConfig:
        syms = [s.strip() for s in self.tickers.toPlainText().replace("\n", ",").split(",") if s.strip()]
        device = self.device.currentText()
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self._on_log("[warn] CUDA not available — falling back to CPU")
                    device = "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
        return FinRLConfig(
            tickers=syms, train_start=self.train_start.text(), train_end=self.train_end.text(),
            trade_start=self.trade_start.text(), trade_end=self.trade_end.text(),
            indicators=tuple(INDICATORS), agent=self.agent.currentText(),
            timesteps=int(self.timesteps.value()), device=device, n_envs=int(self.n_envs.value()),
        )

    # ---- run lifecycle
    def _start(self):
        try:
            cfg = self._collect_cfg()
        except Exception as e:  # noqa: BLE001
            self._on_log(f"[error] bad config: {e}"); return
        if not cfg.tickers:
            self._on_log("[error] no tickers"); return
        self._stop.clear()
        self._reward_x.clear(); self._reward_y.clear()
        self.reward_curve.setData([], []); self.equity_curve.setData([], [])
        self.btn_run.setEnabled(False); self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        threading.Thread(target=self._run, args=(cfg,), daemon=True).start()

    def _run(self, cfg: FinRLConfig):
        emit = self.bridge.log.emit
        try:
            train_df, trade_df = prepare_data(cfg, log=emit)
            model = train(cfg, train_df,
                          progress_cb=lambda s, t, r: self.bridge.progress.emit(
                              s, t, float("nan") if r is None else float(r)),
                          stop=self._stop.is_set, log=emit)
            if self._stop.is_set():
                emit("[run] stopped"); self.bridge.done.emit("stopped"); return
            acct, _ = backtest(cfg, model, trade_df, log=emit)
            if len(acct):
                self.bridge.equity.emit(acct["account_value"].astype(float).tolist())
            self.bridge.done.emit("done")
        except Exception as e:  # noqa: BLE001
            import traceback
            emit("[error] " + str(e)); emit(traceback.format_exc())
            self.bridge.done.emit("error")

    # ---- slots (GUI thread)
    def _on_log(self, s):
        self.logbox.appendPlainText(s)

    def _on_progress(self, steps, total, reward):
        self.progress.setMaximum(total); self.progress.setValue(min(steps, total))
        import math
        if not math.isnan(reward):
            self._reward_x.append(steps); self._reward_y.append(reward)
            self.reward_curve.setData(self._reward_x, self._reward_y)

    def _on_equity(self, vals):
        self.equity_curve.setData(list(range(len(vals))), vals)

    def _on_done(self, status):
        self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False)
        self._on_log(f"[run] {status}")

    # ════════════════════════════════ Evolution tab ════════════════════════════════
    def _build_evo_tab(self):
        import torch
        w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w)
        form = QtWidgets.QFormLayout()

        self.evo_pop = QtWidgets.QSpinBox(); self.evo_pop.setRange(8, 1024); self.evo_pop.setValue(128)
        form.addRow("Population", self.evo_pop)
        self.evo_gens = QtWidgets.QSpinBox(); self.evo_gens.setRange(2, 500); self.evo_gens.setValue(20)
        form.addRow("Generations", self.evo_gens)
        self.evo_obj = QtWidgets.QComboBox(); self.evo_obj.addItems(["sharpe", "sortino", "return"])
        form.addRow("Objective", self.evo_obj)
        self.evo_smooth = QtWidgets.QDoubleSpinBox(); self.evo_smooth.setRange(0.0, 1.0)
        self.evo_smooth.setSingleStep(0.05); self.evo_smooth.setValue(0.20)
        form.addRow("Rebalance smooth", self.evo_smooth)
        self.evo_device = QtWidgets.QComboBox(); self.evo_device.addItems(["cuda", "cpu"])
        self.evo_device.setCurrentText("cuda" if torch.cuda.is_available() else "cpu")
        form.addRow("Device", self.evo_device)
        self.evo_seed = QtWidgets.QSpinBox(); self.evo_seed.setRange(0, 99999); self.evo_seed.setValue(42)
        form.addRow("Seed", self.evo_seed)

        self.btn_evo = QtWidgets.QPushButton("Run evolution"); self.btn_evo.clicked.connect(self._start_evo)
        self.btn_evo_stop = QtWidgets.QPushButton("Stop"); self.btn_evo_stop.setEnabled(False)
        self.btn_evo_stop.clicked.connect(lambda: self._evo_stop.set())
        self.btn_replay = QtWidgets.QPushButton("Open 3-pane replay"); self.btn_replay.clicked.connect(self._open_replay)
        form.addRow(self.btn_evo); form.addRow(self.btn_evo_stop); form.addRow(self.btn_replay)

        left = QtWidgets.QVBoxLayout(); left.addLayout(form); left.addStretch(1)
        h.addLayout(left, 0)

        right = QtWidgets.QVBoxLayout()
        self.skill_plot = pg.PlotWidget(title="Generations — fitness per generation")
        self.skill_plot.addLegend(); self.skill_plot.setLabel("bottom", "generation")
        self.skill_plot.addLine(y=0, pen=pg.mkPen("gray", style=QtCore.Qt.DashLine))
        self.skill_best = self.skill_plot.plot([], [], pen=pg.mkPen("limegreen", width=2),
                                               symbol="o", symbolSize=5, name="best train")
        self.skill_mean = self.skill_plot.plot([], [], pen=pg.mkPen("gray", width=1), name="mean train")
        self.skill_val = self.skill_plot.plot([], [], pen=pg.mkPen("deepskyblue", width=2),
                                              symbol="t", symbolSize=5, name="val (gates champion)")
        right.addWidget(self.skill_plot, 2)
        self.evo_logbox = QtWidgets.QPlainTextEdit(); self.evo_logbox.setReadOnly(True)
        self.evo_logbox.setMaximumBlockCount(2000); right.addWidget(self.evo_logbox, 1)
        h.addLayout(right, 1)
        return w

    def _start_evo(self):
        self._evo_stop.clear()
        self._evo_best.clear(); self._evo_mean.clear(); self._evo_val.clear()
        self.skill_best.setData([], []); self.skill_mean.setData([], []); self.skill_val.setData([], [])
        self.btn_evo.setEnabled(False); self.btn_evo_stop.setEnabled(True)
        threading.Thread(target=self._run_evo, daemon=True).start()

    def _run_evo(self):
        emit = self.bridge.evo_log.emit
        try:
            from evo_portfolio import EvoPortfolioConfig, PortfolioEvo, load_panel, HERE as EH
            csv = EH / "train_data.csv"
            if not csv.exists():
                emit("[data] preparing DOW-30 data (one-time)…")
                from pipeline import FinRLConfig, prepare_data
                tr, _ = prepare_data(FinRLConfig(tickers=_dow30()), log=emit)
                tr.to_csv(csv)
            panel = load_panel(csv)
            cfg = EvoPortfolioConfig(
                pop_size=int(self.evo_pop.value()), n_generations=int(self.evo_gens.value()),
                objective=self.evo_obj.currentText(), smooth=float(self.evo_smooth.value()),
                device=self.evo_device.currentText(), seed=int(self.evo_seed.value()))
            evo = PortfolioEvo(panel, cfg); evo.split_idx = (evo.tr_end, evo.val_end)
            evo.evolve(log=emit, record_path=EH / "evo_portfolio_record.npz",
                       gen_cb=lambda g, b, m, v: self.bridge.evo_gen.emit(g, b, m, v),
                       stop=self._evo_stop.is_set)
            self.bridge.evo_done.emit("stopped" if self._evo_stop.is_set() else "done")
        except Exception as e:  # noqa: BLE001
            import traceback
            emit("[error] " + str(e)); emit(traceback.format_exc())
            self.bridge.evo_done.emit("error")

    def _open_replay(self):
        from evo_portfolio import HERE as EH
        rec = EH / "evo_portfolio_record.npz"
        if not rec.exists():
            self.evo_logbox.appendPlainText("[replay] no recording yet — run evolution first"); return
        import evo_replay_panel as erp
        self._replay_win = erp.ReplayWindow(erp._load(rec))   # keep ref so it isn't GC'd
        self._replay_win.show()

    def _on_evo_gen(self, gen, best, mean, val):
        self._evo_best.append(best); self._evo_mean.append(mean); self._evo_val.append(val)
        xs = list(range(len(self._evo_best)))
        self.skill_best.setData(xs, self._evo_best)
        self.skill_mean.setData(xs, self._evo_mean)
        self.skill_val.setData(xs, self._evo_val)

    def _on_evo_done(self, status):
        self.btn_evo.setEnabled(True); self.btn_evo_stop.setEnabled(False)
        self.btn_replay.setEnabled(True)
        self.evo_logbox.appendPlainText(f"[evo] {status} — click 'Open 3-pane replay' to watch")


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = ControlPanel(); win.show()
    app.exec_()


if __name__ == "__main__":
    main()
