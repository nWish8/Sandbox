#!/usr/bin/env python3
"""
Test script to verify the final coordination between strategy and portfolio.
This ensures no duplicate trades and proper position management.
"""

import pandas as pd
import numpy as np
from backtesting.backtest_engine import BacktestEngine
from strategies.moving_average_cross import MovingAverageCross

def create_test_data():
    """Create synthetic data with clear crossover patterns."""
    np.random.seed(42)
    
    # Create a dataset with clear uptrend and downtrend patterns
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    
    # Create price data with clear MA crossovers
    prices = []
    base_price = 100
    
    # First 100 days: downtrend (short MA below long MA)
    for i in range(100):
        noise = np.random.normal(0, 0.5)
        base_price = base_price * (1 - 0.002 + noise * 0.01)
        prices.append(base_price)
    
    # Next 100 days: uptrend (short MA crosses above long MA)
    for i in range(100):
        noise = np.random.normal(0, 0.5)
        base_price = base_price * (1 + 0.003 + noise * 0.01)
        prices.append(base_price)
    
    # Last 100 days: downtrend again (short MA crosses below long MA)
    for i in range(100):
        noise = np.random.normal(0, 0.5)
        base_price = base_price * (1 - 0.002 + noise * 0.01)
        prices.append(base_price)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000 for _ in prices]
    })
    
    data.set_index('timestamp', inplace=True)
    return data

def test_coordination():
    """Test that strategy and portfolio coordinate correctly."""
    print("Testing strategy-portfolio coordination...")
    
    # Create test data
    data = create_test_data()
    print(f"Created test data with {len(data)} bars")
    
    # Initialize strategy and backtest engine
    strategy = MovingAverageCross(short_win=20, long_win=50)
    engine = BacktestEngine(data, strategy, initial_capital=10000.0, commission=0.001)
    
    # Run backtest
    equity_curve, trade_log = engine.run()
    
    # Analyze results
    print(f"\nBacktest Results:")
    print(f"Total trades: {len(trade_log)}")
    print(f"Initial capital: ${engine.portfolio.cash + engine.portfolio.position * data.iloc[-1]['close']:.2f}")
    print(f"Final equity: ${equity_curve[-1][1]:.2f}")
    
    # Check trade log for correctness
    print(f"\nTrade Log Analysis:")
    buy_count = sum(1 for trade in trade_log if trade['action'] == 'BUY')
    sell_count = sum(1 for trade in trade_log if trade['action'] == 'SELL')
    print(f"BUY trades: {buy_count}")
    print(f"SELL trades: {sell_count}")
    
    # Verify no duplicate trades
    consecutive_buys = 0
    consecutive_sells = 0
    prev_action = None
    
    for trade in trade_log:
        if trade['action'] == 'BUY' and prev_action == 'BUY':
            consecutive_buys += 1
        elif trade['action'] == 'SELL' and prev_action == 'SELL':
            consecutive_sells += 1
        prev_action = trade['action']
    
    print(f"Consecutive BUY errors: {consecutive_buys}")
    print(f"Consecutive SELL errors: {consecutive_sells}")
    
    # Show first few trades
    print(f"\nFirst 10 trades:")
    for i, trade in enumerate(trade_log[:10]):
        print(f"  {i+1}: {trade['action']} at ${trade['price']:.2f} on {trade['time']}")
    
    # Test portfolio state at end
    final_cash = engine.portfolio.cash
    final_position = engine.portfolio.position
    final_price = data.iloc[-1]['close']
    final_equity = engine.portfolio.equity(final_price)
    
    print(f"\nFinal Portfolio State:")
    print(f"Cash: ${final_cash:.2f}")
    print(f"Position: {final_position} shares")
    print(f"Final price: ${final_price:.2f}")
    print(f"Calculated equity: ${final_equity:.2f}")
    print(f"Equity curve final: ${equity_curve[-1][1]:.2f}")
    
    # Verify equity calculation
    assert abs(final_equity - equity_curve[-1][1]) < 0.01, "Equity calculation mismatch!"
    
    print(f"\n✓ All coordination tests passed!")
    
    return equity_curve, trade_log

if __name__ == "__main__":
    equity_curve, trade_log = test_coordination()
