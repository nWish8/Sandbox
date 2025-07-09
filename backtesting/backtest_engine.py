# backtest_engine.py
# Core backtesting logic

import os
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.*=false'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import finplot as fplt
from PyQt6.QtWidgets import QApplication
from PyQt6 import QtCore
from PyQt6.QtCore import QTimer, Qt

from data_pipeline.market_data import main as fetch_data

# --- TradingSimulator class for import/use in main.py ---
class TradingSimulator:
    def __init__(self, config, agent=None):
        # --- Data Configuration ---
        self.config = config
        self.ASSET_TYPE = config['ASSET_TYPE']
        self.SYMBOL = config['SYMBOL']
        self.TIMEFRAME = config['TIMEFRAME']
        self.START_DATE = config['START_DATE']
        self.END_DATE = config['END_DATE']
        self.SIM_TIMEFRAME = config['SIM_TIMEFRAME']
        self.WINDOW_SIZE = config['WINDOW_SIZE']
        self.INTERVAL = config['INTERVAL']
        # --- Data Loading ---
        os.makedirs('market_data', exist_ok=True)
        csv_name = self.SYMBOL.replace('/', '') + f'_{self.TIMEFRAME}_{self.START_DATE}_to_{self.END_DATE}.csv' if self.ASSET_TYPE == 'crypto' else self.SYMBOL + f'_{self.TIMEFRAME}_{self.START_DATE}_to_{self.END_DATE}.csv'
        csv_path = os.path.join('market_data', csv_name)
        if not os.path.exists(csv_path):
            fetch_data(self.ASSET_TYPE, self.SYMBOL, self.TIMEFRAME, self.START_DATE, self.END_DATE)
            if os.path.exists(csv_name):
                os.replace(csv_name, csv_path)
        self.df = pd.read_csv(csv_path)
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.rename(columns={'timestamp': 'time'}, inplace=True)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df.set_index('time', inplace=True)
        # --- Timeframe Mapping ---
        PANDAS_FREQ_MAP = {
            '1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1d',
        }
        self.pandas_freq = PANDAS_FREQ_MAP.get(self.SIM_TIMEFRAME, self.SIM_TIMEFRAME)
        # --- Resample Data ---
        self.df_resampled = self.resample_df(self.df, self.pandas_freq)
        # --- Plot Setup ---
        self.ax = fplt.create_plot(f"{self.SYMBOL} {self.SIM_TIMEFRAME} Sandbox", maximize=False)
        # Move the sandbox window to the top middle of the screen
        self.ax.vb.win.setWindowTitle(f"{self.SYMBOL} {self.SIM_TIMEFRAME} Sandbox")
        screen_geometry = QApplication.primaryScreen().geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        self.ax.vb.win.move(screen_width // 2 - 400, 0)
        self.ax.vb.win.resize(800, 300)  # Set a reasonable width and height

        # --- Portfolio Visualization ---
        plt.ion()
        self.fig, self.ax_portfolio = plt.subplots(figsize=(12, 3))  # Wide, short
        self.portfolio_line, = self.ax_portfolio.plot([], [], color='#00aaff', label='Portfolio Value')
        self.ax_portfolio.set_title('Portfolio Value')
        self.ax_portfolio.set_xlabel('Time')
        self.ax_portfolio.set_ylabel('Value')
        self.ax_portfolio.legend()
        self.fig.autofmt_xdate()
        # Move the agent window just below the sandbox window
        mgr = plt.get_current_fig_manager()
        try:
            mgr.window.move(screen_width // 2 - 400, 300)
            mgr.window.resize(800, 250)
        except Exception:
            pass  # Not all backends support window.move
        self.fig.show()
        # --- Simulation State ---
        self.portfolio_values = []
        self.portfolio_times = []
        self.window_size = self.WINDOW_SIZE
        self.current_idx = int(self.WINDOW_SIZE/2) * (1 if self.SIM_TIMEFRAME == '1m' else int(pd.Timedelta(self.pandas_freq).total_seconds() // 60))
        self.plot = None
        self.one_minute_df = self.df[['open', 'high', 'low', 'close', 'volume']].copy()
        # --- Agent ---
        if agent is not None:
            self.agent = agent
        else:
            from agent import Agent
            self.agent = Agent(input_dim=5, initial_balance=1000, cuda=True)
        # --- Timer ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_chart)
        self.timer.setInterval(self.INTERVAL)
        # --- Speed Control ---
        class SpeedControl(QtCore.QObject):
            def __init__(self, timer, interval):
                super().__init__()
                self.timer = timer
                self.interval = interval
            def set_speed(self, new_interval):
                self.interval = max(10, min(2000, new_interval))
                self.timer.setInterval(self.interval)
                print(f"Playback speed: {self.interval} ms per candle")
            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.Type.KeyPress:
                    if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
                        self.set_speed(self.interval - 50)
                        return True
                    elif event.key() == Qt.Key.Key_Minus:
                        self.set_speed(self.interval + 50)
                        return True
                return False
        self.speed_control = SpeedControl(self.timer, self.INTERVAL)

    def resample_df(self, df, freq):
        if self.SIM_TIMEFRAME == '1m':
            return df
        ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        return df.resample(freq).agg(ohlc_dict).dropna()

    def get_resampled_candles(self, idx):
        partial = self.one_minute_df.iloc[:idx+1]
        if self.SIM_TIMEFRAME != '1m':
            ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            resampled = partial.resample(self.pandas_freq).agg(ohlc_dict).dropna(how='all')
            last_bin = partial.resample(self.pandas_freq).agg(ohlc_dict).tail(1)
            if not last_bin.index[-1] in resampled.index:
                resampled = pd.concat([resampled, last_bin])
            return resampled
        return partial

    def update_chart(self):
        if self.current_idx >= len(self.one_minute_df):
            self.timer.stop()
            return
        shown = self.get_resampled_candles(self.current_idx)[['open', 'close', 'high', 'low']]
        shown = shown.iloc[:self.window_size] if self.current_idx <= self.window_size else shown.iloc[-self.window_size:]
        if len(shown) < 2:
            return
        if self.plot is not None:
            self.plot.update_data(shown)
        else:
            self.plot = fplt.candlestick_ochl(shown, ax=self.ax)
        t = self.current_idx - 1
        if t < len(self.df):
            obs = self.df.iloc[t][['open','high','low','close','volume']].values.astype(np.float32)
            action = self.agent.act(obs)
            price = obs[3]
            self.portfolio_values.append(self.agent.balance + self.agent.holdings * price)
            self.portfolio_times.append(self.df.index[t])
            self.portfolio_line.set_data(self.portfolio_times, self.portfolio_values)
            self.ax_portfolio.relim()
            self.ax_portfolio.autoscale_view()
            # Format x-axis with readable date strings
            from matplotlib.dates import DateFormatter, AutoDateLocator, date2num, num2date
            self.ax_portfolio.xaxis.set_major_locator(AutoDateLocator())
            # Improved custom formatter: use matplotlib's num2date for correct conversion
            def custom_time_formatter(x, pos=None):
                dt = num2date(x)
                if dt.hour == 0 and dt.minute == 0:
                    return dt.strftime('%Y/%m/%d')
                else:
                    return dt.strftime('%H:%M')
            self.ax_portfolio.xaxis.set_major_formatter(plt.FuncFormatter(custom_time_formatter))
            if self.portfolio_times:
                xdata = [date2num(pd.Timestamp(x)) for x in self.portfolio_times]
                self.portfolio_line.set_xdata(xdata)
                if len(set(xdata)) > 1:
                    self.ax_portfolio.set_xlim(min(xdata), max(xdata))
            self.fig.autofmt_xdate()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        fplt.refresh()
        self.current_idx += 1

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def install_speed_control(self, app):
        app.installEventFilter(self.speed_control)
