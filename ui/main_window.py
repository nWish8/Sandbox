from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt6.QtCore import QThread, pyqtSignal

from .plotting import BacktestPlotter


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.plotter = BacktestPlotter()
        self.run_btn = QPushButton("Run Backtest")
        self.status_label = QLabel("Ready to run backtest")
        
        layout = QVBoxLayout()
        layout.addWidget(self.plotter)
        layout.addWidget(self.status_label)
        layout.addWidget(self.run_btn)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        self.setWindowTitle("Backtesting Engine")
        self.resize(900, 600)
