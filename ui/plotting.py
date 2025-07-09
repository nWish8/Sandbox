import finplot as fplt
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class BacktestPlotter(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        ax = fplt.create_plot_widget(self)
        layout.addWidget(ax.ax_widget)
        self.ax = ax

    def plot_results(self, data, equity_curve, trades):
        fplt.candlestick_ochl(data[["open", "close", "high", "low"]])
        if equity_curve:
            eq_times, eq_vals = zip(*equity_curve)
            self.ax.plot(eq_times, eq_vals, color="blue", legend="Equity")
        for trade in trades:
            color = "green" if trade["action"] == "BUY" else "red"
            fplt.add_line((trade["time"], trade["price"]),
                          (trade["time"], trade["price"] * 1.01),
                          color=color, ax=self.ax)
        fplt.refresh()
