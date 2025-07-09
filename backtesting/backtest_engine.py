from __future__ import annotations

import pandas as pd

from .portfolio import Portfolio

class BacktestEngine:
    """Simple offline backtesting engine."""
    def __init__(self, data: pd.DataFrame, strategy, initial_capital: float = 10000.0, commission: float = 0.001):
        self.data = data
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)
        self.commission = commission
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.trade_log: list[dict] = []

    def run(self):
        for bar in self.data.itertuples():
            action = self.strategy.on_bar(bar)
            price = bar.close
            if action == "BUY":
                qty = 1
                self.portfolio.buy(qty, price, fee=self.commission)
                self.trade_log.append({"action": "BUY", "price": price, "time": bar.Index})
            elif action == "SELL":
                qty = self.portfolio.position
                self.portfolio.sell(qty, price, fee=self.commission)
                self.trade_log.append({"action": "SELL", "price": price, "time": bar.Index})
            equity = self.portfolio.equity(price)
            self.equity_curve.append((bar.Index, equity))
        return self.equity_curve, self.trade_log
