# main.py
import datetime as dt
import pandas as pd
import multiprocessing
import backtesting
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from data_pipeline.manager import DataManager

def SMA(values, n):
    """Simple Moving Average."""
    return pd.Series(values).rolling(n).mean()

class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()

if __name__ == '__main__':
    # Enable multiprocessing for optimization
    backtesting.Pool = multiprocessing.Pool

    # Initialize data manager and specify market data parameters
    manager = DataManager()
    data = manager.get_dataset(
        "BTC/USDT",
        start=dt.datetime(2023, 1, 1),
        end=dt.datetime(2023, 2, 1),
        timeframe="1h",   # Output timeframe
        source="ccxt",
        exchange="binance"
    )

    # Set up and run backtest
    bt = Backtest(
        data,
        SmaCross,
        cash=10_000,
        commission=.002
    )

    stats = bt.run()
    print(stats)
    bt.plot(filename="results/results.html")

    # Parameter optimization
    stats_optimized = bt.optimize(
        n1=range(5, 30, 5),
        n2=range(10, 70, 5),
        maximize='Equity Final [$]',
        constraint=lambda param: param.n1 < param.n2
    )
    print("Optimized parameters:", stats_optimized._strategy)
