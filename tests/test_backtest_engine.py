import unittest
import pandas as pd
from backtesting.backtest_engine import BacktestEngine
from strategies.moving_average_cross import MovingAverageCross


class TestBacktestEngine(unittest.TestCase):
    def test_run(self):
        data = pd.DataFrame({
            'open': [1, 2, 3, 4],
            'high': [1, 2, 3, 4],
            'low': [1, 2, 3, 4],
            'close': [1, 2, 3, 4],
            'volume': [1, 1, 1, 1]
        }, index=pd.date_range('2020-01-01', periods=4, freq='1h'))
        strat = MovingAverageCross(short_win=1, long_win=2)
        engine = BacktestEngine(data, strat, initial_capital=10)
        equity, trades = engine.run()
        self.assertTrue(len(equity) == 4)
        self.assertTrue(isinstance(trades, list))


if __name__ == '__main__':
    unittest.main()
