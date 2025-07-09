import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

class BacktestPlotter(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Create subplots
        self.ax1 = self.figure.add_subplot(211)  # Price chart
        self.ax2 = self.figure.add_subplot(212)  # Equity curve

    def plot_results(self, data, equity_curve, trades):
        try:
            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()
            
            # Convert data for plotting
            if hasattr(data, 'reset_index'):
                data = data.reset_index()
            
            # Plot price data (downsample for performance)
            if 'open' in data.columns and 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                if 'time' in data.columns:
                    data['time'] = pd.to_datetime(data['time'])
                    
                    # Downsample to every 60th point for minute data (hourly)
                    if len(data) > 1000:
                        sampled_data = data.iloc[::60]  # Every 60 minutes
                    else:
                        sampled_data = data
                    
                    self.ax1.plot(sampled_data['time'], sampled_data['close'], 
                                 label='Close Price', color='blue', linewidth=1)
                    self.ax1.set_title('Price Chart')
                    self.ax1.set_ylabel('Price')
                    self.ax1.legend()
            
            # Plot equity curve (downsample for performance)
            if equity_curve:
                eq_times, eq_vals = zip(*equity_curve)
                eq_times = pd.to_datetime(eq_times)
                
                # Downsample equity curve if too many points
                if len(equity_curve) > 1000:
                    step = len(equity_curve) // 1000
                    eq_times = eq_times[::step]
                    eq_vals = eq_vals[::step]
                
                self.ax2.plot(eq_times, eq_vals, label='Equity Curve', color='green', linewidth=2)
                self.ax2.set_title('Portfolio Value')
                self.ax2.set_ylabel('Value')
                self.ax2.set_xlabel('Time')
                self.ax2.legend()
            
            # Add trade markers (subsample for performance)
            if trades:
                # Only plot every 100th trade to avoid freezing
                sampled_trades = trades[::100] if len(trades) > 1000 else trades
                
                buy_trades = [t for t in sampled_trades if t['action'] == 'BUY']
                sell_trades = [t for t in sampled_trades if t['action'] == 'SELL']
                
                if buy_trades:
                    buy_times = [pd.to_datetime(t['time']) for t in buy_trades]
                    buy_prices = [t['price'] for t in buy_trades]
                    self.ax1.scatter(buy_times, buy_prices, color='green', marker='^', 
                                   s=50, label=f"BUY ({len(buy_trades)})", alpha=0.7)
                
                if sell_trades:
                    sell_times = [pd.to_datetime(t['time']) for t in sell_trades]
                    sell_prices = [t['price'] for t in sell_trades]
                    self.ax1.scatter(sell_times, sell_prices, color='red', marker='v', 
                                   s=50, label=f"SELL ({len(sell_trades)})", alpha=0.7)
            
            # Format x-axis
            self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            self.ax1.xaxis.set_major_locator(mdates.HourLocator(interval=24))
            self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            self.ax2.xaxis.set_major_locator(mdates.HourLocator(interval=24))
            
            # Rotate x-axis labels
            plt.setp(self.ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(self.ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Adjust layout and refresh
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Plotting error: {e}")
            import traceback
            traceback.print_exc()
