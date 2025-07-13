"""
Example of creating a new strategy in the BT Sandbox package.

This demonstrates how to implement a simple Bollinger Bands strategy
using the new package structure.
"""

import backtrader as bt
from typing import Any


class BollingerBandsStrategy(bt.Strategy):
    """
    Bollinger Bands trading strategy.
    
    This strategy:
    - Buys when price touches the lower Bollinger Band
    - Sells when price touches the upper Bollinger Band
    - Uses configurable position sizing
    
    Parameters:
        bb_period (int): Period for Bollinger Bands calculation (default: 20)
        bb_dev (float): Standard deviation multiplier (default: 2.0)
        position_size (float): Fraction of available cash to use (default: 0.95)
        printlog (bool): Whether to print trade logs (default: True)
    """
    
    params = (
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('position_size', 0.95),
        ('printlog', True),
    )
    
    def __init__(self):
        """Initialize the strategy with Bollinger Bands indicator."""
        # Create Bollinger Bands indicator
        self.bb = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,  # type: ignore
            devfactor=self.params.bb_dev  # type: ignore
        )
        
        # Track orders
        self.order: Any = None
        self.trade_count = 0
        
        if self.params.printlog:  # type: ignore
            print(f"Bollinger Bands Strategy initialized:")
            print(f"  - BB Period: {self.params.bb_period}")  # type: ignore
            print(f"  - BB Deviation: {self.params.bb_dev}")  # type: ignore
            print(f"  - Position Size: {self.params.position_size * 100:.1f}%")  # type: ignore
    
    def next(self):
        """Execute strategy logic on each bar."""
        if self.order:
            return
        
        current_price = self.data.close[0]
        current_datetime = bt.num2date(self.data.datetime[0])
        position_size = self.position.size if self.position else 0
        
        # Buy signal: price touches lower Bollinger Band
        if (current_price <= self.bb.lines.bot[0] and position_size == 0):
            if self.params.printlog:  # type: ignore
                print(f"BUY signal at ${current_price:.2f} (BB Lower: ${self.bb.lines.bot[0]:.2f}) on {current_datetime.strftime('%Y-%m-%d %H:%M')}")
            
            cash = self.broker.get_cash()
            size = (cash * self.params.position_size) / current_price  # type: ignore
            self.order = self.buy(size=size)
            
        # Sell signal: price touches upper Bollinger Band
        elif (current_price >= self.bb.lines.top[0] and position_size > 0):
            if self.params.printlog:  # type: ignore
                print(f"SELL signal at ${current_price:.2f} (BB Upper: ${self.bb.lines.top[0]:.2f}) on {current_datetime.strftime('%Y-%m-%d %H:%M')}")
            
            self.order = self.close()
    
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
            print(f"Bollinger Bands strategy finished with portfolio value: ${final_value:,.2f}")
