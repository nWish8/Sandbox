"""
RSI Strategy for Backtrader.

This module contains a Backtrader-compatible RSI strategy that buys when RSI
is oversold and sells when RSI is overbought.
"""

import backtrader as bt
from typing import Any


class RSIBacktraderStrategy(bt.Strategy):
    """
    RSI-based trading strategy for Backtrader.
    
    This strategy:
    - Buys when RSI falls below the oversold threshold (default: 30)
    - Sells when RSI rises above the overbought threshold (default: 70)
    - Only takes one position at a time
    - Uses configurable position sizing
    
    Parameters:
        rsi_period (int): Period for RSI calculation (default: 14)
        oversold (float): RSI level considered oversold (default: 30)
        overbought (float): RSI level considered overbought (default: 70)
        position_size (float): Fraction of available cash to use (default: 0.95)
        printlog (bool): Whether to print trade logs (default: True)
    """
    
    params = (
        ('rsi_period', 14),
        ('oversold', 30),
        ('overbought', 70),
        ('position_size', 0.95),
        ('printlog', True),
    )
    
    def __init__(self):
        """Initialize the strategy with RSI indicator and tracking variables."""
        # Create RSI indicator
        self.rsi = bt.indicators.RSI(
            self.data.close, 
            period=self.params.rsi_period  # type: ignore
        )
        
        # Track orders and trades
        self.order: Any = None
        self.trade_count = 0
        
        # Log strategy initialization
        if self.params.printlog:  # type: ignore
            print(f"RSI Strategy initialized:")
            print(f"  - RSI Period: {self.params.rsi_period}")  # type: ignore
            print(f"  - Oversold Level: {self.params.oversold}")  # type: ignore
            print(f"  - Overbought Level: {self.params.overbought}")  # type: ignore
            print(f"  - Position Size: {self.params.position_size * 100:.1f}%")  # type: ignore
    
    def next(self):
        """Execute strategy logic on each bar."""
        # Check if we have a pending order
        if self.order:
            return
        
        # Skip if RSI is not ready yet
        if len(self.rsi) < 1:
            return
        
        current_rsi = self.rsi[0]
        current_price = self.data.close[0]
        current_date = self.data.datetime.date(0)
        
        # Buy signal: RSI oversold and no current position
        if current_rsi < self.params.oversold and not self.position:  # type: ignore
            if self.params.printlog:  # type: ignore
                print(f"BUY signal at ${current_price:.2f} (RSI: {current_rsi:.1f}) on {current_date}")
            
            # Calculate position size based on available cash
            cash = self.broker.get_cash()
            size = (cash * self.params.position_size) / current_price  # type: ignore
            self.order = self.buy(size=size)
            
        # Sell signal: RSI overbought and we have a position
        elif current_rsi > self.params.overbought and self.position:  # type: ignore
            if self.params.printlog:  # type: ignore
                print(f"SELL signal at ${current_price:.2f} (RSI: {current_rsi:.1f}) on {current_date}")
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
