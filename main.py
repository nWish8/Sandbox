"""Entry point for running a simple backtest with a PyQt interface."""

import sys
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from backtesting.backtest_engine import BacktestEngine
from strategies.moving_average_cross import MovingAverageCross
from data_pipeline.data_fetcher import load_csv


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    def run_backtest():
        data = load_csv("BTCUSDT", "1h")
        strategy = MovingAverageCross()
        engine = BacktestEngine(data, strategy)
        equity, trades = engine.run()
        window.plotter.plot_results(data, equity, trades)

    window.run_btn.clicked.connect(run_backtest)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
