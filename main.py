"""
Simplified main entry point using Backtrader engine.

This script runs a basic moving average crossover strategy using the enhanced
Backtrader integration without the old GUI components.
"""

import backtrader as bt
from data_pipeline import DataManager
from backtesting.engine import SandboxEngine
import sys
from typing import Any


class RSIStrategy(bt.Strategy):
    """
    RSI-based trading strategy for Backtrader.
    Buys when RSI < oversold level, sells when RSI > overbought level.
    """
    params = (
        ('rsi_period', 14),
        ('oversold', 30),
        ('overbought', 70),
        ('position_size', 0.95),  # Use 95% of available cash
    )
    
    def __init__(self):
        # Create RSI indicator
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)  # type: ignore
        
        # Track orders and trades
        self.order: Any = None
        self.trade_count = 0
    
    def next(self):
        # Check if we have a pending order
        if self.order:
            return
        
        # Skip if RSI is not ready yet
        if len(self.rsi) < 1:
            return
        
        current_rsi = self.rsi[0]
        
        # Buy signal: RSI oversold
        if current_rsi < self.params.oversold and not self.position:  # type: ignore
            print(f"BUY signal at {self.data.close[0]:.2f} (RSI: {current_rsi:.1f}) on {self.data.datetime.date(0)}")
            self.order = self.buy(size=self.params.position_size)  # type: ignore
            
        # Sell signal: RSI overbought
        elif current_rsi > self.params.overbought and self.position:  # type: ignore
            print(f"SELL signal at {self.data.close[0]:.2f} (RSI: {current_rsi:.1f}) on {self.data.datetime.date(0)}")
            self.order = self.sell()
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED: Price: {order.executed.price:.2f}, Size: {order.executed.size:.4f}")
            else:
                print(f"SELL EXECUTED: Price: {order.executed.price:.2f}, Size: {order.executed.size:.4f}")
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"Order {order.status}")
            self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
            pnl = trade.pnl
            pnl_pct = (pnl / trade.value) * 100 if trade.value != 0 else 0
            print(f"TRADE #{self.trade_count} CLOSED: PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")


class MovingAverageCrossStrategy(bt.Strategy):
    """
    Simple moving average crossover strategy for Backtrader.
    """
    params = (
        ('fast_period', 3),
        ('slow_period', 7),
        ('position_size', 0.95),  # Use 95% of available cash
    )
    
    def __init__(self):
        # Create moving averages
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)  # type: ignore
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)  # type: ignore
        
        # Create crossover signal
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)  # type: ignore
        
        # Track orders and trades
        self.order: Any = None
        self.trade_count = 0
    
    def next(self):
        # Check if we have a pending order
        if self.order:
            return
        
        # Buy signal: fast MA crosses above slow MA
        if self.crossover > 0 and not self.position:
            print(f"BUY signal at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
            self.order = self.buy(size=self.params.position_size)  # type: ignore
            
        # Sell signal: fast MA crosses below slow MA  
        elif self.crossover < 0 and self.position:
            print(f"SELL signal at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
            self.order = self.sell()
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED: Price: {order.executed.price:.2f}, Size: {order.executed.size:.4f}")
            else:
                print(f"SELL EXECUTED: Price: {order.executed.price:.2f}, Size: {order.executed.size:.4f}")
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"Order {order.status}")
            self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
            pnl = trade.pnl
            pnl_pct = (pnl / trade.value) * 100 if trade.value != 0 else 0
            print(f"TRADE #{self.trade_count} CLOSED: PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")


def main():
    """Run the backtest with Backtrader engine."""
    print("=" * 60)
    print("Sandbox Trading Research Stack - RSI Strategy Demo")
    print("=" * 60)
    
    # Initialize data manager
    print("1. Initializing data manager...")
    config = {
        'csv': {'data_dir': 'market_data'},
        'default_provider': 'csv'
    }
    manager = DataManager(config)
    
    # Get available data
    symbols = manager.get_available_symbols()
    print(f"Available symbols: {symbols}")
    
    if not symbols.get('csv'):
        print("❌ No CSV data available. Please add CSV files to market_data/ directory.")
        return
    
    # Use the 1-minute BTCUSDT data
    symbol = 'BTCUSDT_1m_2025-06-08_to_2025-07-08'
    print(f"2. Loading 1-minute data for {symbol}...")
    
    try:
        data = manager.fetch_ohlcv(symbol, '1m')
        print(f"   ✓ Loaded {len(data)} bars of 1-minute data")
        print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
        print(f"   💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize Backtrader engine
    print("3. Setting up Backtrader engine...")
    engine = SandboxEngine(
        cash=100000.0,
        commission=0.001  # 0.1% commission
    )
    
    # Run backtest
    print("4. Running RSI strategy backtest...")
    print("-" * 40)
    
    try:
        results, portfolio = engine.run_backtest(
            RSIStrategy,
            data=data,
            rsi_period=14,
            oversold=30,
            overbought=70,
            position_size=0.95
        )
        
        print("-" * 40)
        print("5. Backtest Results:")
        print(f"   💵 Starting Value: ${results.get('starting_value', 0):,.2f}")
        print(f"   💰 Final Value:    ${results.get('final_value', 0):,.2f}")
        print(f"   📈 Total Return:   {results.get('total_return', 0):.2%}")
        
        sharpe = results.get('sharpe_ratio')
        print(f"   📊 Sharpe Ratio:   {sharpe:.2f}" if sharpe is not None else "   📊 Sharpe Ratio:   N/A")
        
        drawdown = results.get('max_drawdown')
        print(f"   📉 Max Drawdown:   {drawdown:.2%}" if drawdown is not None else "   📉 Max Drawdown:   N/A")
        
        if results.get('total_trades', 0) > 0:
            print(f"   🔄 Total Trades:   {results.get('total_trades', 0)}")
            print(f"   ✅ Winning Trades: {results.get('winning_trades', 0)}")
            print(f"   ❌ Losing Trades:  {results.get('losing_trades', 0)}")
        else:
            print("   🔄 No trades executed")
        
        print("\n📊 Portfolio tracking enabled. Check portfolio DataFrame for detailed progression.")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=" * 60)
    print("✅ Backtest completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
