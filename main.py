"""Entry point for running a simple backtest with a PyQt interface."""

import sys
from PyQt6.QtWidgets import QApplication
import finplot as fplt

from ui.main_window import MainWindow
from backtesting.backtest_engine import BacktestEngine
from strategies.moving_average_cross import MovingAverageCross
from data_pipeline.data_fetcher import load_csv
from backtesting.evaluation import compute_metrics


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    def run_backtest():
        try:
            window.status_label.setText("Loading data...")
            window.run_btn.setEnabled(False)
            QApplication.processEvents()  # Update GUI
            
            # Get parameters from UI
            dataset = window.data_combo.currentText()
            initial_capital = window.initial_capital_spin.value()
            commission = window.commission_spin.value() / 100  # Convert percentage to decimal
            
            # Load data
            data = load_csv(dataset, "")
            
            window.status_label.setText("Running backtest...")
            QApplication.processEvents()  # Update GUI
            
            # Create strategy with parameters from UI
            strategy = MovingAverageCross(
                short_win=window.short_ma_spin.value(),
                long_win=window.long_ma_spin.value()
            )
            
            # Run backtest
            engine = BacktestEngine(data, strategy, initial_capital, commission)
            equity, trades = engine.run()
            
            window.status_label.setText("Computing metrics...")
            QApplication.processEvents()  # Update GUI
            
            # Compute metrics
            metrics = compute_metrics(equity, trades)
            
            window.status_label.setText("Updating charts...")
            QApplication.processEvents()  # Update GUI
            
            # Update the advanced interface
            window.update_charts(data, equity, trades)
            window.update_performance_metrics(metrics)
            window.update_trade_log(trades)
            
            # Also update the old plotter for backward compatibility
            window.plotter.plot_results(data, equity, trades)
            
            window.status_label.setText(f"Backtest complete! {len(trades)} trades executed.")
            window.run_btn.setEnabled(True)
            
        except Exception as e:
            window.status_label.setText(f"Error: {str(e)}")
            window.run_btn.setEnabled(True)
            print(f"Backtest error: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_results():
        """Clear all results from the interface."""
        window.status_label.setText("Results cleared.")
        window.metrics_text.clear()
        window.trade_table.setRowCount(0)
        # Clear finplot price chart
        window.price_ax.clear()
        # Clear matplotlib equity chart
        window.equity_ax.clear()
        window.equity_canvas.draw()
        fplt.refresh()

    # Connect buttons
    window.run_btn.clicked.connect(run_backtest)
    window.clear_btn.clicked.connect(clear_results)
    
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
