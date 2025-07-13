"""
Enhanced main entry point demonstrating Backtrader best practices.

This script showcases:
1. Native Yahoo Finance data feeds
2. Proper broker configuration
3. Efficient strategy implementation
4. Realistic commission and slippage
"""

from datetime import datetime, timedelta
import backtrader as bt
from bt_sandbox.backtesting import SandboxEngine
from bt_sandbox.strategies import RSIBacktraderStrategy, MovingAverageCrossStrategy
from bt_sandbox.datafeeds.providers import YahooFinanceProvider, CSVProvider


def demo_csv_data():
    """Demonstrate CSV data with enhanced engine."""
    print("=" * 70)
    print("🚀 BT SANDBOX - Enhanced CSV Demo")
    print("=" * 70)
    
    # Create enhanced engine with realistic settings
    engine = SandboxEngine(
        cash=10000.0,           # $10k starting cash
        commission=0.001,       # 0.1% commission
        slippage_perc=0.0001,   # 0.01% slippage
        sizer_percent=25.0      # 25% position sizing
    )
    
    # Check for local CSV data
    csv_provider = CSVProvider({'data_dir': 'market_data'})
    symbols = csv_provider.get_available_symbols()
    
    if not symbols:
        print("⚠️  No CSV data found in market_data/ directory")
        print("💡 Run fetch_btc_data.py to get sample data")
        return None
    
    # Use first available symbol
    symbol = symbols[0]
    print(f"📊 Using local CSV data: {symbol}")
    
    # Add CSV data using the enhanced engine
    csv_file = f"market_data/{symbol}.csv"
    engine.add_csv_data(csv_file)
    
    # Run strategy
    results = engine.run_backtest(
        RSIBacktraderStrategy,
        rsi_period=14,
        oversold=30,
        overbought=70,
        printlog=True
    )
    
    print_results(results, f"RSI Strategy on {symbol} (CSV)")
    engine.plot_results()
    
    return results


def demo_yahoo_finance():
    """Demonstrate native Yahoo Finance data feed usage."""
    print("=" * 70)
    print("🚀 YAHOO FINANCE DEMO - Enhanced Backtrader Integration")
    print("=" * 70)
    
    # Create enhanced engine with realistic settings
    engine = SandboxEngine(
        cash=100000.0,          # $100k starting cash
        commission=0.0005,      # 0.05% commission (realistic for stocks)
        slippage_perc=0.0001,   # 0.01% slippage
        sizer_percent=15.0      # 15% position sizing
    )
    
    # Add Yahoo Finance data using Backtrader's native feed
    print("\n📈 Adding Apple (AAPL) data using native Backtrader Yahoo feed...")
    
    try:
        data_feed = engine.add_yahoo_data(
            symbol='AAPL',
            fromdate=datetime(2023, 1, 1),
            todate=datetime(2024, 1, 1),
            timeframe=bt.TimeFrame.Days,
            adjclose=True  # Use adjusted close prices
        )
        
        # Run enhanced RSI strategy
        print("\n🎯 Running enhanced RSI strategy...")
        results = engine.run_backtest(
            RSIBacktraderStrategy,
            rsi_period=14,
            oversold=30,
            overbought=70,
            printlog=True
        )
        
        # Display comprehensive results
        print_results(results, "Enhanced RSI Strategy on AAPL")
        
        # Plot with enhanced styling
        engine.plot_results()
        
        return results
        
    except Exception as e:
        print(f"❌ Yahoo Finance demo failed: {e}")
        print("💡 Make sure you have an internet connection")
        return None


def print_results(results: dict, title: str):
    """Print formatted backtest results."""
    print("\n" + "=" * 70)
    print(f"📊 {title.upper()} - RESULTS")
    print("=" * 70)
    
    print(f"� Starting Value: ${results['start_value']:,.2f}")
    print(f"💰 Final Value:    ${results['end_value']:,.2f}")
    print(f"📈 Total Return:   {results['total_return_pct']:+.2f}%")
    
    # Performance metrics
    if results.get('sharpe_ratio'):
        print(f"📊 Sharpe Ratio:   {results['sharpe_ratio']:.3f}")
    
    if results.get('max_drawdown'):
        print(f"📉 Max Drawdown:   {results['max_drawdown']:.2%}")
    
    # Trading statistics
    total_trades = results.get('total_trades', 0)
    winning_trades = results.get('winning_trades', 0)
    losing_trades = results.get('losing_trades', 0)
    
    print(f"🔄 Total Trades:   {total_trades}")
    
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        print(f"✅ Winning Trades: {winning_trades} ({win_rate:.1f}%)")
        print(f"❌ Losing Trades:  {losing_trades} ({100-win_rate:.1f}%)")
    
    print("=" * 70)


def main():
    """Run the enhanced backtest demos."""
    print("🎯 BT Sandbox - Enhanced Backtrader Demo")
    print("Showcasing best practices for professional backtesting")
    
    try:
        # Demo 1: CSV data (always available)
        csv_results = demo_csv_data()
        
        # Demo 2: Yahoo Finance (requires internet)
        yahoo_results = demo_yahoo_finance()
        
        print("\n🎉 Demos completed!")
        
        # Summary comparison
        if csv_results and yahoo_results:
            print("\n📊 PERFORMANCE COMPARISON:")
            print(f"CSV Data:      {csv_results['total_return_pct']:+.2f}%")
            print(f"AAPL (Yahoo):  {yahoo_results['total_return_pct']:+.2f}%")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
