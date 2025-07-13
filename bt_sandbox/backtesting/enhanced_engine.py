"""
Enhanced Backtrader engine following best practices.

Improvements include:
1. Centralized broker configuration
2. Realistic commission and slippage settings
3. Proper sizer configuration
4. Support for native Backtrader data feeds
"""

import backtrader as bt
from typing import Dict, Any, Union, Optional, Type
import pandas as pd
from datetime import datetime


class EnhancedEngine(bt.Cerebro):
    """
    Enhanced Backtrader engine with best practices.
    
    Features:
    - Centralized broker configuration
    - Realistic commission and slippage
    - Proper sizer setup
    - Native data feed support
    """
    
    def __init__(
        self, 
        cash: float = 100000.0,
        commission: float = 0.0005,  # 0.05% commission
        slippage_perc: float = 0.0001,  # 0.01% slippage
        sizer_percent: float = 10.0,  # 10% of cash per trade
        **kwargs
    ):
        """
        Initialize enhanced engine with best practices.
        
        Args:
            cash: Starting cash (single source of truth)
            commission: Commission rate as decimal (0.0005 = 0.05%)
            slippage_perc: Slippage percentage  
            sizer_percent: Percentage of cash to use per trade
            **kwargs: Additional Cerebro parameters
        """
        super().__init__(**kwargs)
        
        # Configure broker (single source of truth)
        self._setup_broker(cash, commission, slippage_perc)
        
        # Configure sizer
        self._setup_sizer(sizer_percent)
        
        # Add default analyzers
        self._add_analyzers()
        
        print(f"✅ Enhanced Engine initialized:")
        print(f"   💰 Starting Cash: ${cash:,.2f}")
        print(f"   📊 Commission: {commission:.4f} ({commission*100:.2f}%)")
        print(f"   ⚡ Slippage: {slippage_perc:.4f} ({slippage_perc*100:.2f}%)")
        print(f"   📏 Position Size: {sizer_percent:.1f}% of cash")
    
    def _setup_broker(self, cash: float, commission: float, slippage_perc: float):
        """Configure broker with realistic settings."""
        # Set cash (single source of truth)
        self.broker.setcash(cash)
        
        # Set commission
        self.broker.setcommission(commission=commission)
        
        # Add slippage for more realistic results
        if slippage_perc > 0:
            self.broker.set_slippage_perc(perc=slippage_perc)
    
    def _setup_sizer(self, percent: float):
        """Configure position sizer."""
        self.addsizer(bt.sizers.PercentSizer, percents=percent)
    
    def _add_analyzers(self):
        """Add comprehensive performance analyzers."""
        self.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.addanalyzer(bt.analyzers.DrawDown, _name='drawdown') 
        self.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        self.addanalyzer(bt.analyzers.Calmar, _name='calmar')
    
    def add_yahoo_data(
        self,
        symbol: str,
        fromdate: datetime,
        todate: datetime,
        timeframe: Any = bt.TimeFrame.Days,
        compression: int = 1,
        **kwargs
    ) -> bt.feeds.YahooFinanceData:
        """
        Add Yahoo Finance data using Backtrader's native feed.
        
        Args:
            symbol: Yahoo symbol (e.g., 'AAPL', 'BTC-USD')
            fromdate: Start date
            todate: End date
            timeframe: bt.TimeFrame constant
            compression: Compression factor
            **kwargs: Additional parameters
            
        Returns:
            The data feed that was added
        """
        data_feed = bt.feeds.YahooFinanceData(
            dataname=symbol,
            fromdate=fromdate,
            todate=todate,
            timeframe=timeframe,
            compression=compression,
            adjclose=True,  # Use adjusted close
            **kwargs
        )
        
        self.adddata(data_feed)
        print(f"📈 Added Yahoo data: {symbol} ({fromdate.date()} to {todate.date()})")
        
        return data_feed
    
    def add_csv_data(
        self,
        filepath: str,
        fromdate: Optional[datetime] = None,
        todate: Optional[datetime] = None,
        timeframe: Any = bt.TimeFrame.Days,
        compression: int = 1,
        **kwargs
    ) -> bt.feeds.GenericCSVData:
        """
        Add CSV data using Backtrader's native CSV feed.
        
        Args:
            filepath: Path to CSV file
            fromdate: Optional start date filter
            todate: Optional end date filter  
            timeframe: bt.TimeFrame constant
            compression: Compression factor
            **kwargs: Additional CSV parameters
            
        Returns:
            The data feed that was added
        """
        params = {
            'dataname': filepath,
            'timeframe': timeframe,
            'compression': compression,
            'dtformat': '%Y-%m-%d %H:%M:%S',
            'datetime': 0,
            'open': 1,
            'high': 2,
            'low': 3,
            'close': 4,
            'volume': 5,
            'openinterest': -1,
            **kwargs
        }
        
        if fromdate:
            params['fromdate'] = fromdate
        if todate:
            params['todate'] = todate
        
        data_feed = bt.feeds.GenericCSVData(**params)
        self.adddata(data_feed)
        
        print(f"📁 Added CSV data: {filepath}")
        if fromdate or todate:
            print(f"   📅 Date range: {fromdate or 'start'} to {todate or 'end'}")
        
        return data_feed
    
    def add_pandas_data(
        self,
        dataframe: pd.DataFrame,
        fromdate: Optional[datetime] = None,
        todate: Optional[datetime] = None,
        timeframe: Any = bt.TimeFrame.Days,
        compression: int = 1,
        **kwargs
    ) -> bt.feeds.PandasData:
        """
        Add pandas DataFrame as data feed.
        
        Args:
            dataframe: Pandas DataFrame with OHLCV data
            fromdate: Optional start date filter
            todate: Optional end date filter
            timeframe: bt.TimeFrame constant
            compression: Compression factor
            **kwargs: Additional parameters
            
        Returns:
            The data feed that was added
        """
        params = {
            'dataname': dataframe,
            'timeframe': timeframe,
            'compression': compression,
            **kwargs
        }
        
        if fromdate:
            params['fromdate'] = fromdate
        if todate:
            params['todate'] = todate
        
        data_feed = bt.feeds.PandasData(**params)
        self.adddata(data_feed)
        
        print(f"🐼 Added Pandas data: {len(dataframe)} rows")
        if fromdate or todate:
            print(f"   📅 Date range: {fromdate or 'start'} to {todate or 'end'}")
        
        return data_feed
    
    def run_backtest(
        self,
        strategy_class: Type[bt.Strategy],
        **strategy_params
    ) -> Dict[str, Any]:
        """
        Run backtest with the given strategy.
        
        Args:
            strategy_class: Strategy class to run
            **strategy_params: Parameters to pass to strategy
            
        Returns:
            Dictionary with backtest results
        """
        # Add strategy
        self.addstrategy(strategy_class, **strategy_params)
        
        # Record starting value
        start_value = self.broker.getvalue()
        
        # Run backtest
        print(f"🚀 Running backtest with {strategy_class.__name__}...")
        results = self.run()
        
        # Record ending value
        end_value = self.broker.getvalue()
        
        # Extract results from analyzers
        strategy_results = results[0]
        
        # Calculate performance metrics
        performance = {
            'start_value': start_value,
            'end_value': end_value,
            'total_return': (end_value - start_value) / start_value,
            'total_return_pct': ((end_value - start_value) / start_value) * 100
        }
        
        # Add analyzer results
        analyzers = {}
        for name, analyzer in strategy_results.analyzers._items.items():
            try:
                analyzers[name] = analyzer.get_analysis()
            except:
                analyzers[name] = None
        
        performance['analyzers'] = analyzers
        
        # Extract key metrics
        if 'sharpe' in analyzers and analyzers['sharpe']:
            performance['sharpe_ratio'] = analyzers['sharpe'].get('sharperatio')
        
        if 'drawdown' in analyzers and analyzers['drawdown']:
            performance['max_drawdown'] = analyzers['drawdown'].get('max', {}).get('drawdown', 0) / 100
        
        if 'trades' in analyzers and analyzers['trades']:
            trades_data = analyzers['trades']
            performance['total_trades'] = trades_data.get('total', {}).get('total', 0)
            performance['winning_trades'] = trades_data.get('won', {}).get('total', 0)
            performance['losing_trades'] = trades_data.get('lost', {}).get('total', 0)
        
        return performance
    
    def plot_results(self, **plot_kwargs):
        """
        Plot backtest results with sensible defaults.
        
        Args:
            **plot_kwargs: Arguments passed to bt.Cerebro.plot()
        """
        default_args = {
            'style': 'candlestick',
            'barup': 'green',
            'bardown': 'red',
            'volume': True,
        }
        
        # Override defaults with user arguments
        plot_args = {**default_args, **plot_kwargs}
        
        try:
            print("📊 Opening interactive plot...")
            self.plot(**plot_args)
            print("✅ Plot displayed successfully!")
        except Exception as e:
            print(f"❌ Plot failed: {e}")
            print("💡 Try installing matplotlib: pip install matplotlib")


# Example usage function
def run_enhanced_example():
    """Example of using the enhanced engine with best practices."""
    from datetime import datetime, timedelta
    from bt_sandbox.strategies.improved_strategies import ImprovedRSIStrategy
    
    # Create enhanced engine with realistic settings
    engine = EnhancedEngine(
        cash=100000.0,          # $100k starting cash
        commission=0.0005,      # 0.05% commission
        slippage_perc=0.0001,   # 0.01% slippage
        sizer_percent=10.0      # 10% position sizing
    )
    
    # Add Yahoo Finance data (recommended approach)
    engine.add_yahoo_data(
        symbol='AAPL',
        fromdate=datetime(2023, 1, 1),
        todate=datetime(2024, 1, 1),
        timeframe=bt.TimeFrame.Days
    )
    
    # Run backtest
    results = engine.run_backtest(
        ImprovedRSIStrategy,
        rsi_period=14,
        oversold=30,
        overbought=70,
        printlog=True
    )
    
    # Display results
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    print(f"💰 Final Value: ${results['end_value']:,.2f}")
    print(f"📈 Total Return: {results['total_return_pct']:.2f}%")
    
    if results.get('sharpe_ratio'):
        print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    if results.get('max_drawdown'):
        print(f"📉 Max Drawdown: {results['max_drawdown']:.2%}")
    
    print(f"🔄 Total Trades: {results.get('total_trades', 0)}")
    print("="*60)
    
    # Plot results
    engine.plot_results()


if __name__ == "__main__":
    run_enhanced_example()
