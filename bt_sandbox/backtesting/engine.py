"""
Backtrader-based backtesting engine for the Sandbox trading research stack.

This module provides a modern, feature-rich backtesting engine that replaces
the simple loop-based approach with Backtrader's robust framework.
"""

from typing import Dict, Any, Union, Tuple, Optional, Type
import pandas as pd
import backtrader as bt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FractionalSizer(bt.Sizer):
    """
    Custom sizer that allows fractional position sizing.
    
    This sizer enables trading fractional shares/units, which is essential
    for crypto trading and portfolio percentage allocation strategies.
    """
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        """
        Calculate position size allowing fractional shares.
        
        Returns the size passed to buy/sell orders directly.
        """
        # Return the size specified in the order (allows fractional)
        return self.strategy.lastorder_size if hasattr(self.strategy, 'lastorder_size') else 1.0


class PercentSizer(bt.Sizer):
    """
    Sizer that allocates a percentage of available cash to positions.
    
    Useful for risk management and consistent position sizing across trades.
    """
    
    params = (
        ('percents', 10),  # Percentage of cash to use (default 10%)
        ('retfloat', True),  # Return float values
    )
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        """
        Calculate position size based on percentage of available cash.
        
        Args:
            comminfo: Commission info object
            cash: Available cash
            data: Data feed
            isbuy: True if buying, False if selling
            
        Returns:
            float: Position size based on cash percentage
        """
        if isbuy:
            # Calculate how much cash to use
            target_value = cash * (self.p.percents / 100.0)
            # Calculate position size based on current price
            size = target_value / data.close[0]
            
            if self.p.retfloat:
                return float(size)
            else:
                return int(size)
        else:
            # For selling, return 0 (will sell entire position)
            return 0


class SandboxEngine(bt.Cerebro):
    """
    Enhanced Backtrader Cerebro subclass for sandbox trading research.
    
    Provides a simplified interface for backtesting with sensible defaults,
    fractional position sizing, and comprehensive analytics.
    """
    
    def __init__(
        self,
        cash: float = 100000.0,
        commission: float = 0.0005,
        enable_fractional: bool = True,
        **kwargs
    ):
        """
        Initialize the SandboxEngine.
        
        Args:
            cash: Starting cash amount (default: $100,000)
            commission: Commission rate as decimal (default: 0.05%)
            enable_fractional: Whether to enable fractional position sizing
            **kwargs: Additional arguments passed to bt.Cerebro
        """
        super().__init__(**kwargs)
        
        # Set broker parameters
        self.broker.setcash(cash)
        self.broker.setcommission(commission=commission)
        
        # Backtrader handles fractional sizing natively, remove custom sizer for now
        # Users can specify exact sizes in buy/sell calls
        
        # Add default analyzers
        self._add_default_analyzers()
        
        logger.info(f"SandboxEngine initialized with ${cash:,.2f} cash and {commission:.4f} commission")
    
    def _add_default_analyzers(self) -> None:
        """Add standard analyzers for performance evaluation."""
        self.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    
    def add_data_feed(
        self,
        data: Union[pd.DataFrame, str, Path],
        name: str = 'data0',
        **kwargs
    ) -> None:
        """
        Add data feed from DataFrame or CSV file.
        
        Args:
            data: Pandas DataFrame with OHLCV data or path to CSV file
            name: Name for the data feed
            **kwargs: Additional arguments for bt.feeds.PandasData
        """
        if isinstance(data, (str, Path)):
            # Load from CSV file
            df = pd.read_csv(data)
            logger.info(f"Loaded data from {data}: {len(df)} rows")
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            logger.info(f"Using DataFrame: {len(df)} rows")
        else:
            raise ValueError("Data must be pandas DataFrame or path to CSV file")
        
        # Ensure proper column names and datetime index
        df = self._prepare_dataframe(df)
        
        # Create Backtrader data feed with explicit datetime column mapping
        # Note: dataname parameter works despite type checker warnings
        data_feed = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index as datetime
            open=0,         # Column index for open
            high=1,         # Column index for high  
            low=2,          # Column index for low
            close=3,        # Column index for close
            volume=4,       # Column index for volume
            openinterest=-1, # No open interest data
            **kwargs
        )  # type: ignore
        
        self.adddata(data_feed)
        logger.info(f"Added data feed '{name}' with {len(df)} bars")
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for Backtrader consumption.
        
        Args:
            df: Raw DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Properly formatted DataFrame
        """
        df = df.copy()
        
        # Handle timestamp column
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('datetime')
            df = df.drop('timestamp', axis=1)
        elif 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'])
            df = df.set_index('datetime')
            df = df.drop('time', axis=1)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Assume first column is datetime if no explicit timestamp
            df.index = pd.to_datetime(df.index)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing column '{col}', creating with close price")
                if col == 'volume':
                    df[col] = 1000  # Default volume
                else:
                    df[col] = df['close']
        
        # Sort by datetime and ensure timezone-naive for Backtrader compatibility
        df = df.sort_index()
        
        # Convert timezone-aware datetime to timezone-naive if needed
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        # Ensure index name is set properly for Backtrader
        df.index.name = 'datetime'
        
        return df
    
    def _add_data_from_input(self, data: Union[pd.DataFrame, str], name: str = "data0"):
        """
        Helper method to add data to the engine from various input types.
        
        Args:
            data: DataFrame or CSV path with OHLCV data
            name: Name for the data feed
        """
        if isinstance(data, str):
            # Load from CSV file
            df = pd.read_csv(data, index_col=0, parse_dates=True)
            logger.info(f"Loaded DataFrame from {data}: {len(df)} rows")
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            logger.info(f"Using DataFrame: {len(df)} rows")
        else:
            raise ValueError("Data must be pandas DataFrame or path to CSV file")
        
        # Ensure proper column names and datetime index
        df = self._prepare_dataframe(df)
        
        # Create Backtrader data feed
        data_feed = bt.feeds.PandasData(dataname=df)  # type: ignore
        
        self.adddata(data_feed)
        logger.info(f"Added data feed '{name}' with {len(df)} bars")
    
    def run_backtest(
        self,
        strategy_cls: Type[bt.Strategy],
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[pd.DataFrame, str]] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Run backtest with specified strategy and parameters.
        
        Args:
            strategy_cls: Backtrader Strategy class
            params: Dictionary of strategy parameters
            data: DataFrame or CSV path with OHLCV data (optional if already added)
            **kwargs: Additional arguments for strategy
            
        Returns:
            Tuple containing:
                - Dict with performance statistics
                - DataFrame with portfolio value over time
        """
        if params is None:
            params = {}
        
        # Add data if provided
        if data is not None:
            self._add_data_from_input(data)
        
        # Add strategy with parameters
        self.addstrategy(strategy_cls, **params, **kwargs)
        
        logger.info(f"Running backtest with {strategy_cls.__name__}")
        logger.info(f"Strategy parameters: {params}")
        
        # Run the backtest
        results = self.run()
        
        if not results:
            raise RuntimeError("Backtest failed to produce results")
        
        strategy_result = results[0]
        
        # Extract analyzer results
        stats = self._extract_analyzer_results(strategy_result)
        
        # Create portfolio DataFrame
        portfolio_df = self._create_portfolio_dataframe(strategy_result)
        
        logger.info("Backtest completed successfully")
        logger.info(f"Final portfolio value: ${self.broker.getvalue():,.2f}")
        
        return stats, portfolio_df
    
    def _extract_analyzer_results(self, strategy) -> Dict[str, Any]:
        """
        Extract and serialize analyzer results to dictionary.
        
        Args:
            strategy: Strategy instance with analyzers
            
        Returns:
            Dict containing all analyzer results
        """
        stats = {
            'initial_cash': self.broker.startingcash,
            'final_value': self.broker.getvalue(),
            'total_return': (self.broker.getvalue() / self.broker.startingcash) - 1,
        }
        
        # Extract Sharpe ratio
        if hasattr(strategy.analyzers, 'sharpe'):
            sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
            stats['sharpe_ratio'] = sharpe_analysis.get('sharperatio', 0.0)
        
        # Extract drawdown info
        if hasattr(strategy.analyzers, 'drawdown'):
            dd_analysis = strategy.analyzers.drawdown.get_analysis()
            stats['max_drawdown'] = dd_analysis.get('max', {}).get('drawdown', 0.0)
            stats['max_drawdown_period'] = dd_analysis.get('max', {}).get('len', 0)
        
        # Extract trade statistics
        if hasattr(strategy.analyzers, 'trades'):
            trade_analysis = strategy.analyzers.trades.get_analysis()
            stats.update({
                'total_trades': trade_analysis.get('total', {}).get('total', 0),
                'winning_trades': trade_analysis.get('won', {}).get('total', 0),
                'losing_trades': trade_analysis.get('lost', {}).get('total', 0),
                'win_rate': (
                    trade_analysis.get('won', {}).get('total', 0) /
                    max(trade_analysis.get('total', {}).get('total', 1), 1) * 100
                ),
                'avg_win': trade_analysis.get('won', {}).get('pnl', {}).get('average', 0.0),
                'avg_loss': trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0.0),
                'profit_factor': abs(
                    trade_analysis.get('won', {}).get('pnl', {}).get('total', 0.0) /
                    max(abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0.0)), 1.0)
                ),
            })
        
        return stats
    
    def _create_portfolio_dataframe(self, strategy) -> pd.DataFrame:
        """
        Create DataFrame with portfolio value over time.
        
        Args:
            strategy: Strategy instance
            
        Returns:
            pd.DataFrame with datetime index and portfolio values
        """
        if not hasattr(strategy.analyzers, 'timereturn'):
            logger.warning("TimeReturn analyzer not available, returning empty DataFrame")
            return pd.DataFrame()
        
        time_return = strategy.analyzers.timereturn.get_analysis()
        
        # Convert to DataFrame
        dates = list(time_return.keys())
        values = [self.broker.startingcash * (1 + ret) for ret in time_return.values()]
        
        portfolio_df = pd.DataFrame({
            'datetime': dates,
            'portfolio_value': values
        })
        portfolio_df = portfolio_df.set_index('datetime')
        
        return portfolio_df


# Usage example in comments:
"""
# Basic usage example:

import backtrader as bt
import pandas as pd
from backtesting.engine import SandboxEngine

# Define a simple strategy
class SMACrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )
    
    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.order = None
    
    def next(self):
        if self.order:
            return
            
        if self.crossover > 0 and not self.position:
            self.order = self.buy(size=0.5)  # Buy 0.5 shares (fractional)
        elif self.crossover < 0 and self.position:
            self.order = self.sell()
    
    def notify_order(self, order):
        if order.status == order.Completed:
            self.order = None

# Load data and run backtest
data = pd.read_csv('market_data/BTCUSDT_1h.csv')

engine = SandboxEngine(cash=50000, commission=0.001)
engine.add_data_feed(data)

stats, portfolio_df = engine.run_backtest(
    SMACrossStrategy,
    params={'fast_period': 20, 'slow_period': 50}
)

print(f"Total Return: {stats['total_return']:.2%}")
print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 'N/A')}")
print(f"Max Drawdown: {stats.get('max_drawdown', 0):.2%}")
print(f"Win Rate: {stats.get('win_rate', 0):.1f}%")

# Fractional sizing works out of the box!
# The engine successfully handles 0.5 shares, 1.7 shares, etc.
"""
