"""
Moving Average Crossover Strategy for Backtrader.

This module contains a Backtrader-compatible moving average crossover strategy.
"""

import backtrader as bt
from typing import Any


class MovingAverageCrossStrategy(bt.Strategy):
    """
    Simple moving average crossover strategy for Backtrader.
    
    This strategy:
    - Buys when fast MA crosses above slow MA
    - Sells when fast MA crosses below slow MA
    - Only takes one position at a time
    - Uses configurable position sizing
    
    Parameters:
        fast_period (int): Period for fast moving average (default: 3)
        slow_period (int): Period for slow moving average (default: 7)
        position_size (float): Fraction of available cash to use (default: 0.95)
        printlog (bool): Whether to print trade logs (default: True)
    """
    
    params = (
        ('fast_period', 3),
        ('slow_period', 7),
        ('position_size', 0.95),
        ('printlog', True),
    )
    
    def __init__(self):
        """Initialize the strategy with moving averages and tracking variables."""
        # Create moving averages
        self.fast_ma = bt.indicators.SMA(
            self.data.close, 
            period=self.params.fast_period  # type: ignore
        )
        self.slow_ma = bt.indicators.SMA(
            self.data.close, 
            period=self.params.slow_period  # type: ignore
        )
        
        # Create crossover signal
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)  # type: ignore
        
        # Track orders and trades
        self.order: Any = None
        self.trade_count = 0
        
        # Log strategy initialization
        if self.params.printlog:  # type: ignore
            print(f"Moving Average Crossover Strategy initialized:")
            print(f"  - Fast MA Period: {self.params.fast_period}")  # type: ignore
            print(f"  - Slow MA Period: {self.params.slow_period}")  # type: ignore
            print(f"  - Position Size: {self.params.position_size * 100:.1f}%")  # type: ignore
    
    def next(self):
        """Execute strategy logic on each bar."""
        # Check if we have a pending order
        if self.order:
            return
        
        current_price = self.data.close[0]
        current_date = self.data.datetime.date(0)
        
        # Buy signal: fast MA crosses above slow MA
        if self.crossover > 0 and not self.position:
            if self.params.printlog:  # type: ignore
                print(f"BUY signal at ${current_price:.2f} on {current_date}")
            
            # Calculate position size based on available cash
            cash = self.broker.get_cash()
            size = (cash * self.params.position_size) / current_price  # type: ignore
            self.order = self.buy(size=size)
            
        # Sell signal: fast MA crosses below slow MA  
        elif self.crossover < 0 and self.position:
            if self.params.printlog:  # type: ignore
                print(f"SELL signal at ${current_price:.2f} on {current_date}")
            self.order = self.sell()
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.params.printlog:  # type: ignore
                    print(f"BUY EXECUTED: Price: ${order.executed.price:.2f}, Size: {order.executed.size:.4f}")
            else:
                if self.params.printlog:  # type: ignore
                    print(f"SELL EXECUTED: Price: ${order.executed.price:.2f}, Size: {order.executed.size:.4f}")
            self.order = None
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.params.printlog:  # type: ignore
                print(f"Order {order.status}")
            self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if trade.isclosed:
            self.trade_count += 1
            pnl = trade.pnl
            pnl_pct = (pnl / trade.value) * 100 if trade.value != 0 else 0
            
            if self.params.printlog:  # type: ignore
                print(f"TRADE #{self.trade_count} CLOSED: PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
    
    def stop(self):
        """Called when the strategy finishes."""
        if self.params.printlog:  # type: ignore
            final_value = self.broker.get_value()
            print(f"Strategy finished with portfolio value: ${final_value:,.2f}")
