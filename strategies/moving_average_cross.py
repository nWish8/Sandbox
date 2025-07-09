from .strategy_base import StrategyBase
import numpy as np

class MovingAverageCross(StrategyBase):
    def __init__(self, short_win: int = 50, long_win: int = 200):
        self.short_win = short_win
        self.long_win = long_win
        self.prices: list[float] = []

    def on_bar(self, bar):
        self.prices.append(bar.close)
        if len(self.prices) >= self.long_win:
            short_ma = np.mean(self.prices[-self.short_win:])
            long_ma = np.mean(self.prices[-self.long_win:])
            if short_ma > long_ma:
                return "BUY"
            elif short_ma < long_ma:
                return "SELL"
        return "HOLD"
