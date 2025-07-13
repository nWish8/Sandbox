"""
Main entry point for BT Sandbox.

This script runs trading strategies using the enhanced Backtrader integration
and the modular strategy implementations from the bt_sandbox package.

Usage:
    python -m bt_sandbox.main
"""

import backtrader as bt
from bt_sandbox.datafeeds import DataManager
from bt_sandbox.backtesting import SandboxEngine
from bt_sandbox.strategies import RSIBacktraderStrategy
import sys
from typing import Any
from datetime import datetime, timedelta


def main():
    """Run the backtest with Backtrader engine."""
    print("=" * 60)
    print("BT Sandbox - RSI Strategy Demo (4H Timeframe)")
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
    
    # Use the fresh 4-hour BTCUSDT data
    symbol = 'BTCUSDT_4h_recent'
    print(f"2. Loading 4-hour data for {symbol}...")
    
    try:
        data = manager.fetch_ohlcv(symbol, '4h')
        print(f"   ✓ Loaded {len(data)} bars of 4-hour data")
        print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
        print(f"   💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize Backtrader engine
    print("3. Setting up Backtrader engine...")
    engine = SandboxEngine(
        cash=1000.0,
        commission=0.001  # 0.1% commission
    )
    
    # Run backtest
    print("4. Running RSI strategy backtest on 4-hour timeframe...")
    print("-" * 40)
    
    try:
        results, portfolio = engine.run_backtest(
            RSIBacktraderStrategy,
            data=data,
            rsi_period=14,
            oversold=30,
            overbought=70,
            position_size=0.95,
            printlog=True
        )
        
        print("-" * 40)
        print("5. Backtest Results:")
        print(f"   💵 Starting Value: $1,000.00")
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
        
        # Plot the results
        print("6. Displaying backtest results chart...")
        try:
            # The engine is itself a cerebro instance (SandboxEngine extends bt.Cerebro)
            print("   📈 Opening Backtrader plot window...")
            engine.plot(style='candlestick', barup='green', bardown='red')
            print("   ✓ Plot window opened successfully!")
        except Exception as plot_error:
            print(f"   ⚠️  Plotting failed: {plot_error}")
            print("   📊 Results are still valid, but chart display is unavailable")
        
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
