from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
                             QTabWidget, QTextEdit, QSplitter, QScrollArea, QApplication)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont
import finplot as fplt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from .plotting import BacktestPlotter


class AdvancedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize finplot
        fplt.autoviewrestore()
        
        self.setWindowTitle("Advanced Backtesting Engine")
        
        # Center the window and make it shorter
        self.resize(1200, 700)
        self.center_window()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create horizontal splitter for main layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_widget_layout = QHBoxLayout(main_widget)
        main_widget_layout.addWidget(main_splitter)
        
        # Left panel for controls
        left_panel = self.create_control_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel for charts and results
        right_panel = self.create_chart_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions (30% left, 70% right)
        main_splitter.setSizes([400, 1000])
        
        # Initialize data
        self.current_data = None
        self.current_equity = None
        self.current_trades = None
        
    def center_window(self):
        """Center the window on the screen."""
        from PyQt6.QtWidgets import QApplication
        
        # Get the primary screen
        screen = QApplication.primaryScreen()
        if screen is not None:
            screen_geometry = screen.geometry()
            
            # Calculate center position
            x = (screen_geometry.width() - self.width()) // 2
            y = (screen_geometry.height() - self.height()) // 2
            
            self.move(x, y)
        else:
            # Fallback to default position
            self.move(200, 100)
    
    def create_control_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Strategy Selection Group
        strategy_group = QGroupBox("Strategy Selection")
        strategy_layout = QFormLayout(strategy_group)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Moving Average Cross",
            "RSI Strategy", 
            "Bollinger Bands",
            "MACD Strategy"
        ])
        strategy_layout.addRow("Strategy:", self.strategy_combo)
        
        # Strategy Parameters Group
        params_group = QGroupBox("Strategy Parameters")
        params_layout = QFormLayout(params_group)
        self.params_layout = params_layout  # Store reference for visibility update
        
        # Moving Average parameters
        self.short_ma_spin = QSpinBox()
        self.short_ma_spin.setRange(1, 100)
        self.short_ma_spin.setValue(20)
        params_layout.addRow("Short MA Period:", self.short_ma_spin)
        
        self.long_ma_spin = QSpinBox()
        self.long_ma_spin.setRange(1, 200)
        self.long_ma_spin.setValue(50)
        params_layout.addRow("Long MA Period:", self.long_ma_spin)
        
        # RSI parameters
        self.rsi_period_spin = QSpinBox()
        self.rsi_period_spin.setRange(2, 50)
        self.rsi_period_spin.setValue(14)
        params_layout.addRow("RSI Period:", self.rsi_period_spin)
        
        self.rsi_oversold_spin = QSpinBox()
        self.rsi_oversold_spin.setRange(10, 40)
        self.rsi_oversold_spin.setValue(30)
        params_layout.addRow("RSI Oversold:", self.rsi_oversold_spin)
        
        self.rsi_overbought_spin = QSpinBox()
        self.rsi_overbought_spin.setRange(60, 90)
        self.rsi_overbought_spin.setValue(70)
        params_layout.addRow("RSI Overbought:", self.rsi_overbought_spin)
        
        # Bollinger Bands parameters
        self.bb_period_spin = QSpinBox()
        self.bb_period_spin.setRange(5, 50)
        self.bb_period_spin.setValue(20)
        params_layout.addRow("BB Period:", self.bb_period_spin)
        
        self.bb_std_spin = QDoubleSpinBox()
        self.bb_std_spin.setRange(0.5, 4.0)
        self.bb_std_spin.setValue(2.0)
        self.bb_std_spin.setSingleStep(0.1)
        params_layout.addRow("BB Std Dev:", self.bb_std_spin)
        
        # MACD parameters
        self.macd_fast_spin = QSpinBox()
        self.macd_fast_spin.setRange(5, 30)
        self.macd_fast_spin.setValue(12)
        params_layout.addRow("MACD Fast:", self.macd_fast_spin)
        
        self.macd_slow_spin = QSpinBox()
        self.macd_slow_spin.setRange(15, 50)
        self.macd_slow_spin.setValue(26)
        params_layout.addRow("MACD Slow:", self.macd_slow_spin)
        
        self.macd_signal_spin = QSpinBox()
        self.macd_signal_spin.setRange(5, 20)
        self.macd_signal_spin.setValue(9)
        params_layout.addRow("MACD Signal:", self.macd_signal_spin)
        
        # Connect strategy combo to parameter visibility
        self.strategy_combo.currentTextChanged.connect(self.update_parameter_visibility)
        
        # Initialize parameter visibility
        self.update_parameter_visibility()
        
        # Backtesting Parameters Group
        backtest_group = QGroupBox("Backtesting Parameters")
        backtest_layout = QFormLayout(backtest_group)
        
        self.initial_capital_spin = QDoubleSpinBox()
        self.initial_capital_spin.setRange(1000, 1000000)
        self.initial_capital_spin.setValue(10000)
        self.initial_capital_spin.setSuffix(" USD")
        backtest_layout.addRow("Initial Capital:", self.initial_capital_spin)
        
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0.0, 0.1)
        self.commission_spin.setValue(0.001)
        self.commission_spin.setDecimals(4)
        self.commission_spin.setSuffix(" %")
        backtest_layout.addRow("Commission:", self.commission_spin)
        
        # Data Selection Group
        data_group = QGroupBox("Data Selection")
        data_layout = QFormLayout(data_group)
        
        self.data_combo = QComboBox()
        self.data_combo.addItems([
            "BTCUSDT_1m_2025-06-08_to_2025-07-08",
            "BTCUSDT_1h"
        ])
        data_layout.addRow("Dataset:", self.data_combo)
        
        # Control Buttons
        button_layout = QVBoxLayout()
        
        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.run_btn.clicked.connect(self.run_backtest)
        
        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.clear_btn.clicked.connect(self.clear_results)
        
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.clear_btn)
        
        # Status Label
        self.status_label = QLabel("Ready to run backtest")
        self.status_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        
        # Add all groups to layout
        layout.addWidget(strategy_group)
        layout.addWidget(params_group)
        layout.addWidget(backtest_group)
        layout.addWidget(data_group)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        layout.addStretch()
        
        return panel
    
    def create_chart_panel(self):
        """Create the right chart and results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Charts Tab
        charts_tab = self.create_charts_tab()
        self.tab_widget.addTab(charts_tab, "Charts")
        
        # Performance Tab
        performance_tab = self.create_performance_tab()
        self.tab_widget.addTab(performance_tab, "Performance")
        
        # Trade Log Tab
        trade_log_tab = self.create_trade_log_tab()
        self.tab_widget.addTab(trade_log_tab, "Trade Log")
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    def create_charts_tab(self):
        """Create the charts tab with candlestick and equity charts."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create vertical splitter for charts
        charts_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Create finplot candlestick chart
        candlestick_widget = QWidget()
        candlestick_layout = QVBoxLayout(candlestick_widget)
        
        # Create single finplot chart for candlesticks
        self.price_ax = fplt.create_plot('Price Chart', init_zoom_periods=500)
        if isinstance(self.price_ax, list):
            self.price_ax = self.price_ax[0]
        candlestick_layout.addWidget(self.price_ax.vb.win)
        
        # Create matplotlib equity chart
        equity_widget = QWidget()
        equity_layout = QVBoxLayout(equity_widget)
        
        # Create matplotlib figure for equity curve
        self.equity_figure = Figure(figsize=(10, 3))
        self.equity_canvas = FigureCanvas(self.equity_figure)
        self.equity_ax = self.equity_figure.add_subplot(111)
        
        equity_layout.addWidget(QLabel("Portfolio Equity Curve:"))
        equity_layout.addWidget(self.equity_canvas)
        
        # Add both widgets to splitter
        charts_splitter.addWidget(candlestick_widget)
        charts_splitter.addWidget(equity_widget)
        
        # Set splitter proportions (60% candlestick, 40% equity)
        charts_splitter.setSizes([600, 400])
        
        layout.addWidget(charts_splitter)
        
        return tab
    
    def create_performance_tab(self):
        """Create the performance metrics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Performance metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setFont(QFont("Consolas", 10))
        
        layout.addWidget(QLabel("Performance Metrics:"))
        layout.addWidget(self.metrics_text)
        
        return tab
    
    def create_trade_log_tab(self):
        """Create the trade log tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Trade log table
        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(5)
        self.trade_table.setHorizontalHeaderLabels([
            "Time", "Action", "Price", "Quantity", "P&L"
        ])
        
        layout.addWidget(QLabel("Trade Log:"))
        layout.addWidget(self.trade_table)
        
        return tab
    
    def update_charts(self, data, equity_curve, trades):
        """Update the finplot charts with new data."""
        try:
            # Clear previous plots
            self.price_ax.clear()
            self.equity_ax.clear()
            
            # Prepare data for finplot
            if hasattr(data, 'reset_index'):
                plot_data = data.reset_index()
            else:
                plot_data = data.copy()
            
            # Convert time to datetime if needed
            if 'time' in plot_data.columns:
                plot_data['time'] = pd.to_datetime(plot_data['time'])
                plot_data = plot_data.set_index('time')
            
            # Plot candlestick chart
            if all(col in plot_data.columns for col in ['open', 'high', 'low', 'close']):
                fplt.candlestick_ochl(plot_data[['open', 'close', 'high', 'low']], ax=self.price_ax)
            
            # Plot trade markers on candlestick chart
            if trades:
                buy_trades = [t for t in trades if t['action'] == 'BUY']
                sell_trades = [t for t in trades if t['action'] == 'SELL']
                
                if buy_trades:
                    buy_times = [pd.to_datetime(t['time']) for t in buy_trades]
                    buy_prices = [t['price'] for t in buy_trades]
                    fplt.plot(buy_times, buy_prices, ax=self.price_ax, color='#00FF00', 
                             style='^', width=3, legend='Buy')
                
                if sell_trades:
                    sell_times = [pd.to_datetime(t['time']) for t in sell_trades]
                    sell_prices = [t['price'] for t in sell_trades]
                    fplt.plot(sell_times, sell_prices, ax=self.price_ax, color='#FF0000', 
                             style='v', width=3, legend='Sell')
            
            # Refresh finplot
            fplt.refresh()
            
            # Update matplotlib equity chart
            if equity_curve:
                eq_times, eq_vals = zip(*equity_curve)
                eq_times = pd.to_datetime(eq_times)
                
                self.equity_ax.plot(eq_times, eq_vals, color='#2E8B57', linewidth=2, label='Portfolio Value')
                self.equity_ax.set_title('Portfolio Equity Curve', fontsize=12)
                self.equity_ax.set_ylabel('Value ($)', fontsize=10)
                self.equity_ax.set_xlabel('Time', fontsize=10)
                self.equity_ax.grid(True, alpha=0.3)
                self.equity_ax.legend()
                
                # Format x-axis timestamps to save space
                self.equity_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                self.equity_ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
                
                # Rotate x-axis labels for better readability
                plt.setp(self.equity_ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Adjust layout to prevent label cutoff
                self.equity_figure.tight_layout()
                self.equity_canvas.draw()
            
        except Exception as e:
            print(f"Chart update error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_parameter_visibility(self):
        """Update which parameters are visible based on selected strategy."""
        strategy = self.strategy_combo.currentText()
        
        # Show/hide parameters based on strategy
        ma_visible = strategy == "Moving Average Cross"
        rsi_visible = strategy == "RSI Strategy"
        bb_visible = strategy == "Bollinger Bands"
        macd_visible = strategy == "MACD Strategy"
        
        # MA parameters
        self.short_ma_spin.setVisible(ma_visible)
        self.long_ma_spin.setVisible(ma_visible)
        
        # RSI parameters
        self.rsi_period_spin.setVisible(rsi_visible)
        self.rsi_oversold_spin.setVisible(rsi_visible)
        self.rsi_overbought_spin.setVisible(rsi_visible)
        
        # BB parameters
        self.bb_period_spin.setVisible(bb_visible)
        self.bb_std_spin.setVisible(bb_visible)
        
        # MACD parameters
        self.macd_fast_spin.setVisible(macd_visible)
        self.macd_slow_spin.setVisible(macd_visible)
        self.macd_signal_spin.setVisible(macd_visible)
    
    def run_backtest(self):
        """Execute the backtest with selected parameters."""
        try:
            self.status_label.setText("Running backtest...")
            self.run_btn.setEnabled(False)
            
            # Import strategies
            from strategies.moving_average_cross import MovingAverageCross
            from strategies.rsi_strategy import RSIStrategy
            from strategies.bollinger_bands_strategy import BollingerBandsStrategy
            from strategies.macd_strategy import MACDStrategy
            
            # Load data
            import os
            
            dataset = self.data_combo.currentText()
            if dataset == "BTCUSDT_1m_2025-06-08_to_2025-07-08":
                data_path = "market_data/BTCUSDT_1m_2025-06-08_to_2025-07-08.csv"
            else:
                data_path = "market_data/BTCUSDT_1h.csv"
            
            # Load CSV data
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
            else:
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Create strategy based on selection
            strategy_name = self.strategy_combo.currentText()
            if strategy_name == "Moving Average Cross":
                strategy = MovingAverageCross(
                    short_win=self.short_ma_spin.value(),
                    long_win=self.long_ma_spin.value()
                )
            elif strategy_name == "RSI Strategy":
                strategy = RSIStrategy(
                    period=self.rsi_period_spin.value(),
                    oversold=self.rsi_oversold_spin.value(),
                    overbought=self.rsi_overbought_spin.value()
                )
            elif strategy_name == "Bollinger Bands":
                strategy = BollingerBandsStrategy(
                    period=self.bb_period_spin.value(),
                    std_dev=self.bb_std_spin.value()
                )
            elif strategy_name == "MACD Strategy":
                strategy = MACDStrategy(
                    fast_period=self.macd_fast_spin.value(),
                    slow_period=self.macd_slow_spin.value(),
                    signal_period=self.macd_signal_spin.value()
                )
            
            # Run backtest
            from backtesting.backtest_engine import BacktestEngine
            engine = BacktestEngine(
                data=data,
                strategy=strategy,
                initial_capital=self.initial_capital_spin.value(),
                commission=self.commission_spin.value() / 100
            )
            
            equity_curve, trades = engine.run()
            
            # Update displays
            self.update_charts(data, equity_curve, trades)
            self.update_performance_metrics(equity_curve, trades, engine.portfolio)
            self.update_trade_log(trades)
            
            # Store current results
            self.current_data = data
            self.current_equity = equity_curve
            self.current_trades = trades
            
            self.status_label.setText(f"Backtest completed - {len(trades)} trades executed")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.run_btn.setEnabled(True)

    def clear_results(self):
        """Clear all results and charts."""
        try:
            # Clear finplot chart
            self.price_ax.clear()
            fplt.refresh()
            
            # Clear equity chart
            self.equity_ax.clear()
            self.equity_canvas.draw()
            
            # Clear text displays
            self.metrics_text.clear()
            
            # Clear trade table
            self.trade_table.setRowCount(0)
            
            # Reset stored data
            self.current_data = None
            self.current_equity = None
            self.current_trades = None
            
            self.status_label.setText("Results cleared")
            
        except Exception as e:
            self.status_label.setText(f"Error clearing results: {str(e)}")

    def update_performance_metrics(self, equity_curve, trades, portfolio):
        """Update the performance metrics display."""
        try:
            from backtesting.evaluation import compute_metrics
            
            # Convert equity curve to format expected by compute_metrics
            if equity_curve:
                equity_series = pd.Series(
                    [eq[1] for eq in equity_curve],
                    index=[eq[0] for eq in equity_curve]
                )
                
                metrics = compute_metrics(equity_series, trades)
                
                # Format metrics for display
                metrics_text = "PERFORMANCE METRICS\n"
                metrics_text += "=" * 50 + "\n\n"
                
                metrics_text += f"Total Return: {metrics['total_return']:.2f}%\n"
                metrics_text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                metrics_text += f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
                metrics_text += f"Win Rate: {metrics['win_rate']:.2f}%\n"
                metrics_text += f"Total Trades: {len(trades)}\n"
                metrics_text += f"Final Portfolio Value: ${equity_curve[-1][1]:.2f}\n"
                metrics_text += f"Final Cash: ${portfolio.cash:.2f}\n"
                metrics_text += f"Final Position: {portfolio.position} shares\n"
                
                self.metrics_text.setText(metrics_text)
                
        except Exception as e:
            self.metrics_text.setText(f"Error computing metrics: {str(e)}")

    def update_trade_log(self, trades):
        """Update the trade log table."""
        try:
            self.trade_table.setRowCount(len(trades))
            
            for i, trade in enumerate(trades):
                self.trade_table.setItem(i, 0, QTableWidgetItem(str(trade['time'])))
                self.trade_table.setItem(i, 1, QTableWidgetItem(trade['action']))
                self.trade_table.setItem(i, 2, QTableWidgetItem(f"${trade['price']:.2f}"))
                self.trade_table.setItem(i, 3, QTableWidgetItem("N/A"))  # Quantity not stored
                self.trade_table.setItem(i, 4, QTableWidgetItem("N/A"))  # P&L not calculated
                
        except Exception as e:
            print(f"Error updating trade log: {str(e)}")

# Keep the old MainWindow for backward compatibility
class MainWindow(AdvancedMainWindow):
    def __init__(self):
        super().__init__()
        # Add the old plotter for backward compatibility
        self.plotter = BacktestPlotter()
