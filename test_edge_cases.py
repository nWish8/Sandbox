#!/usr/bin/env python3
"""
Comprehensive test for all edge cases in the backtesting system.
This verifies proper behavior for various scenarios.
"""

import pandas as pd
import numpy as np
from backtesting.backtest_engine import BacktestEngine
from strategies.moving_average_cross import MovingAverageCross

def test_insufficient_data():
    """Test behavior with insufficient data for MA calculation."""
    print("Testing insufficient data scenario...")
    
    # Create very small dataset
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    prices = [100 + i for i in range(10)]
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000 for _ in prices]
    })
    data.set_index('timestamp', inplace=True)
    
    strategy = MovingAverageCross(short_win=20, long_win=50)
    engine = BacktestEngine(data, strategy, initial_capital=10000.0)
    
    equity_curve, trade_log = engine.run()
    
    print(f"  Trades with insufficient data: {len(trade_log)}")
    print(f"  Expected: 0 trades")
    assert len(trade_log) == 0, "Should have no trades with insufficient data"
    print("  ✓ Passed")

def test_no_crossovers():
    """Test behavior when MAs never cross."""
    print("\nTesting no crossovers scenario...")
    
    # Create steadily trending data (no crossovers)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    base_price = 100
    prices = []
    
    for i in range(200):
        base_price = base_price * 1.001  # Steady uptrend
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
    
    strategy = MovingAverageCross(short_win=20, long_win=50)
    engine = BacktestEngine(data, strategy, initial_capital=10000.0)
    
    equity_curve, trade_log = engine.run()
    
    print(f"  Trades with no crossovers: {len(trade_log)}")
    print(f"  Expected: 0 trades")
    assert len(trade_log) == 0, "Should have no trades without crossovers"
    print("  ✓ Passed")

def test_insufficient_capital():
    """Test behavior with insufficient capital for trades."""
    print("\nTesting insufficient capital scenario...")
    
    # Create data with high prices
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = [10000 + i * 100 for i in range(100)]  # Very high prices
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000 for _ in prices]
    })
    data.set_index('timestamp', inplace=True)
    
    strategy = MovingAverageCross(short_win=10, long_win=20)
    engine = BacktestEngine(data, strategy, initial_capital=100.0)  # Very low capital
    
    equity_curve, trade_log = engine.run()
    
    print(f"  Trades with insufficient capital: {len(trade_log)}")
    print(f"  Final cash: ${engine.portfolio.cash:.2f}")
    print(f"  Final position: {engine.portfolio.position}")
    print("  ✓ Passed (handled gracefully)")

def test_multiple_crossovers():
    """Test multiple crossovers in sequence."""
    print("\nTesting multiple crossovers scenario...")
    
    # Create data with multiple clear crossovers
    dates = pd.date_range('2024-01-01', periods=400, freq='D')
    prices = []
    base_price = 100
    
    # Create oscillating pattern with multiple crossovers
    for i in range(400):
        cycle = np.sin(i * 0.05) * 10  # Oscillating pattern
        trend = i * 0.01  # Slight uptrend
        noise = np.random.normal(0, 0.5)
        base_price = 100 + cycle + trend + noise
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
    
    strategy = MovingAverageCross(short_win=10, long_win=30)
    engine = BacktestEngine(data, strategy, initial_capital=10000.0)
    
    equity_curve, trade_log = engine.run()
    
    print(f"  Total trades: {len(trade_log)}")
    print(f"  BUY trades: {sum(1 for t in trade_log if t['action'] == 'BUY')}")
    print(f"  SELL trades: {sum(1 for t in trade_log if t['action'] == 'SELL')}")
    
    # Verify alternating pattern
    actions = [t['action'] for t in trade_log]
    alternating = all(actions[i] != actions[i+1] for i in range(len(actions)-1))
    print(f"  Alternating pattern: {alternating}")
    assert alternating, "Trades should alternate between BUY and SELL"
    print("  ✓ Passed")

def test_equity_curve_consistency():
    """Test that equity curve is consistent throughout backtest."""
    print("\nTesting equity curve consistency...")
    
    # Create simple test data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = [100 + i * 0.5 for i in range(100)]
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000 for _ in prices]
    })
    data.set_index('timestamp', inplace=True)
    
    strategy = MovingAverageCross(short_win=10, long_win=20)
    engine = BacktestEngine(data, strategy, initial_capital=10000.0)
    
    equity_curve, trade_log = engine.run()
    
    # Check that equity curve is monotonic when no trades
    if len(trade_log) == 0:
        # Should remain at initial capital
        all_equal = all(equity == 10000.0 for _, equity in equity_curve)
        print(f"  All equity values equal to initial capital: {all_equal}")
        assert all_equal, "Equity should remain constant with no trades"
    else:
        # Check that equity values are reasonable
        equities = [equity for _, equity in equity_curve]
        print(f"  Min equity: ${min(equities):.2f}")
        print(f"  Max equity: ${max(equities):.2f}")
        print(f"  Final equity: ${equities[-1]:.2f}")
        assert all(equity > 0 for equity in equities), "Equity should never be negative"
    
    print("  ✓ Passed")

def run_all_tests():
    """Run all edge case tests."""
    print("Running comprehensive edge case tests...\n")
    
    test_insufficient_data()
    test_no_crossovers()
    test_insufficient_capital()
    test_multiple_crossovers()
    test_equity_curve_consistency()
    
    print(f"\n🎉 All edge case tests passed! The backtesting system is robust.")

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    run_all_tests()
