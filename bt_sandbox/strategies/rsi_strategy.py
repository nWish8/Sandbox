"""
Improved RSI strategy following Backtrader best practices.

This strategy demonstrates:
1. Proper indicator initialization in __init__
2. Efficient signal processing in next()
3. Clean order management
4. Realistic broker configuration
"""

import backtrader as bt
from datetime import datetime


class RSIBacktraderStrategy(bt.Strategy):
    """
    RSI Reversal Strategy with Backtrader best practices.
    
    Strategy Logic:
    - Buy when RSI crosses above oversold level (30)
    - Sell when RSI crosses below overbought level (70)
    - Uses proper indicator initialization for efficiency
    """
    
    # Strategy parameters
    params = (
        ('rsi_period', 14),        # RSI calculation period
        ('oversold', 30),          # Oversold threshold
        ('overbought', 70),        # Overbought threshold  
        ('position_size', 0.95),   # Position size as fraction of available cash
        ('printlog', True),        # Enable trade logging
    )
    
    def __init__(self):
        """Initialize indicators and signals."""
        # Keep reference to the close price
        self.dataclose = self.datas[0].close
        
        # Initialize indicators in __init__ for efficiency
        self.rsi = bt.indicators.RSI(
            self.datas[0].close,
            period=self.params.rsi_period
        )
        
        # Keep track of pending orders
        self.order = None
        
        # Track trade statistics
        self.trades_count = 0
        
        # Log strategy initialization
        if self.params.printlog:
            print("-" * 50)
            print("RSI Reversal Strategy initialized:")
            print(f"  - RSI Period: {self.params.rsi_period}")
            print(f"  - Oversold Level: {self.params.oversold}")
            print(f"  - Overbought Level: {self.params.overbought}")
            print(f"  - Position Size: {self.params.position_size * 100:.1f}%")
            print(f"  - Using efficient indicator initialization")
            print("-" * 50)
    
    def next(self):
        """Process each bar of data."""
        # Skip if we have a pending order
        if self.order:
            return
        
        # Check if we are in the market
        if not self.position:
            # We are not in the market, look for buy signal
            # Buy when RSI crosses above oversold level
            if (len(self.rsi) > 1 and 
                self.rsi[-1] <= self.params.oversold and 
                self.rsi[0] > self.params.oversold):
                
                # Calculate position size
                size = (self.broker.get_cash() * self.params.position_size) / self.dataclose[0]
                
                # Buy signal
                self.order = self.buy(size=size)
                
                if self.params.printlog:
                    print(f"BUY signal at ${self.dataclose[0]:.2f} "
                          f"(RSI cross above {self.params.oversold}: "
                          f"{self.rsi[-1]:.1f} -> {self.rsi[0]:.1f}) "
                          f"on {self.datas[0].datetime.date(0)}")
        
        else:
            # We are in the market, look for sell signal
            # Sell when RSI crosses below overbought level
            if (len(self.rsi) > 1 and 
                self.rsi[-1] >= self.params.overbought and 
                self.rsi[0] < self.params.overbought):
                
                # Sell signal
                self.order = self.sell()
                
                if self.params.printlog:
                    print(f"SELL signal at ${self.dataclose[0]:.2f} "
                          f"(RSI cross below {self.params.overbought}: "
                          f"{self.rsi[-1]:.1f} -> {self.rsi[0]:.1f}) "
                          f"on {self.datas[0].datetime.date(0)}")
    
    def notify_order(self, order):
        """Handle order status notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return
        
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.params.printlog:
                    print(f"BUY EXECUTED: Price: ${order.executed.price:.2f}, "
                          f"Size: {order.executed.size:.4f}")
            else:  # Sell
                if self.params.printlog:
                    print(f"SELL EXECUTED: Price: ${order.executed.price:.2f}, "
                          f"Size: {order.executed.size:.4f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.params.printlog:
                print(f"Order {order.status}")
        
        # Reset order reference
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return
        
        self.trades_count += 1
        
        if self.params.printlog:
            print(f"TRADE #{self.trades_count} CLOSED: "
                  f"PnL: ${trade.pnl:.2f} ({trade.pnlcomm:.2f}%)")
    
    def stop(self):
        """Called when the strategy finishes."""
        if self.params.printlog:
            final_value = self.broker.getvalue()
            print(f"Strategy finished with portfolio value: ${final_value:,.2f}")


# Additional improved strategies for variety
class MovingAverageCrossStrategy(bt.Strategy):
    """
    Moving Average Crossover strategy with best practices.
    
    Demonstrates efficient indicator usage and signal-based logic.
    """
    
    params = (
        ('fast_period', 20),       # Fast MA period
        ('slow_period', 50),       # Slow MA period
        ('position_size', 0.95),   # Position size
        ('printlog', True),        # Enable logging
    )
    
    def __init__(self):
        """Initialize indicators efficiently."""
        self.dataclose = self.datas[0].close
        
        # Create moving averages
        self.sma_fast = bt.indicators.SMA(period=self.params.fast_period)
        self.sma_slow = bt.indicators.SMA(period=self.params.slow_period)
        
        # Create crossover signal
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
        
        # Track orders and trades
        self.order = None
        self.trades_count = 0
        
        if self.params.printlog:
            print("-" * 50)
            print("Moving Average Crossover Strategy initialized:")
            print(f"  - Fast MA: {self.params.fast_period} periods")
            print(f"  - Slow MA: {self.params.slow_period} periods")
            print(f"  - Position Size: {self.params.position_size * 100:.1f}%")
            print("-" * 50)
    
    def next(self):
        """Process each bar."""
        if self.order:
            return
        
        if not self.position:
            # Look for buy signal (fast MA crosses above slow MA)
            if self.crossover > 0:
                size = (self.broker.get_cash() * self.params.position_size) / self.dataclose[0]
                self.order = self.buy(size=size)
                
                if self.params.printlog:
                    print(f"BUY signal at ${self.dataclose[0]:.2f} "
                          f"(MA crossover: {self.sma_fast[0]:.2f} > {self.sma_slow[0]:.2f}) "
                          f"on {self.datas[0].datetime.date(0)}")
        
        else:
            # Look for sell signal (fast MA crosses below slow MA)
            if self.crossover < 0:
                self.order = self.sell()
                
                if self.params.printlog:
                    print(f"SELL signal at ${self.dataclose[0]:.2f} "
                          f"(MA crossover: {self.sma_fast[0]:.2f} < {self.sma_slow[0]:.2f}) "
                          f"on {self.datas[0].datetime.date(0)}")
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.params.printlog:
                    print(f"BUY EXECUTED: Price: ${order.executed.price:.2f}, "
                          f"Size: {order.executed.size:.4f}")
            else:
                if self.params.printlog:
                    print(f"SELL EXECUTED: Price: ${order.executed.price:.2f}, "
                          f"Size: {order.executed.size:.4f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.params.printlog:
                print(f"Order {order.status}")
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return
        
        self.trades_count += 1
        
        if self.params.printlog:
            print(f"TRADE #{self.trades_count} CLOSED: "
                  f"PnL: ${trade.pnl:.2f} ({trade.pnlcomm:.2f}%)")


class SignalMAStrategy(bt.SignalStrategy):
    """
    Concise signal-based Moving Average strategy.
    
    Demonstrates bt.SignalStrategy for very short strategy implementations.
    """
    
    params = (
        ('fast_period', 20),
        ('slow_period', 50),
    )
    
    def __init__(self):
        """Initialize with signal-based approach."""
        # Create the crossover signal and add it as a LONG signal
        self.signal_add(
            bt.SIGNAL_LONG,
            bt.indicators.CrossOver(
                bt.indicators.SMA(period=self.params.fast_period),
                bt.indicators.SMA(period=self.params.slow_period)
            )
        )
        
        print(f"Signal-based MA Strategy: {self.params.fast_period}/{self.params.slow_period}")
