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
        try:
            window.status_label.setText("Loading data...")
            window.run_btn.setEnabled(False)
            QApplication.processEvents()  # Update GUI
            
            data = load_csv("BTCUSDT_1m_2025-06-08_to_2025-07-08", "")
            
            window.status_label.setText("Running backtest...")
            QApplication.processEvents()  # Update GUI
            
            strategy = MovingAverageCross()
            engine = BacktestEngine(data, strategy)
            equity, trades = engine.run()
            
            window.status_label.setText("Plotting results...")
            QApplication.processEvents()  # Update GUI
            
            window.plotter.plot_results(data, equity, trades)
            
            window.status_label.setText(f"Backtest complete! {len(trades)} trades executed.")
            window.run_btn.setEnabled(True)
            
        except Exception as e:
            window.status_label.setText(f"Error: {str(e)}")
            window.run_btn.setEnabled(True)
            print(f"Backtest error: {e}")
            import traceback
            traceback.print_exc()

    window.run_btn.clicked.connect(run_backtest)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
