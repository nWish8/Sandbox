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
            # Pass current portfolio position to strategy
            action = self.strategy.on_bar(bar, current_position=self.portfolio.position)
            price = bar.close  # Keep as pandas scalar for now
            
            # Execute trades based on signals
            if action == "BUY" and self.portfolio.position == 0:
                # Calculate quantity based on available cash
                cost_per_share = float(price) * (1 + self.commission)
                qty = int(self.portfolio.cash / cost_per_share)
                if qty > 0:
                    self.portfolio.buy(qty, float(price), fee=self.commission)
                    self.trade_log.append({"action": "BUY", "price": float(price), "time": bar.Index})
                    
            elif action == "SELL" and self.portfolio.position > 0:
                # Sell all current position
                qty = self.portfolio.position
                if qty > 0:
                    self.portfolio.sell(qty, float(price), fee=self.commission)
                    self.trade_log.append({"action": "SELL", "price": float(price), "time": bar.Index})
            
            # Calculate and store equity for this bar
            equity = self.portfolio.equity(float(price))
            self.equity_curve.append((bar.Index, equity))
            
        return self.equity_curve, self.trade_log
