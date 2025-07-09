from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton

from .plotting import BacktestPlotter


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.plotter = BacktestPlotter()
        self.run_btn = QPushButton("Run Backtest")
        layout = QVBoxLayout()
        layout.addWidget(self.plotter)
        layout.addWidget(self.run_btn)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
