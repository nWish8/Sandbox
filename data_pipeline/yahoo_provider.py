"""
Yahoo Finance data provider for the Sandbox trading stack.

This provider fetches stock market data from Yahoo Finance using the
yfinance library, supporting stocks, ETFs, indices, and other securities.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import logging

try:
    import yfinance as yf
except ImportError:
    yf = None

from .base_provider import BaseDataProvider

logger = logging.getLogger(__name__)


class YahooProvider(BaseDataProvider):
    """
    Data provider for Yahoo Finance stock market data.
    
    Fetches OHLCV data for stocks, ETFs, indices, and other securities
    using the yfinance library. Supports various timeframes and handles
    timezone conversion automatically.
    
    Configuration options:
    - auto_adjust: Apply stock splits and dividends (default: True)
    - prepost: Include pre/post market data (default: False)
    - threads: Number of threads for multi-symbol requests (default: True)
    - proxy: Proxy server URL (optional)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Yahoo Finance data provider.
        
        Args:
            config: Configuration dictionary with yfinance-specific options
        """
        super().__init__("Yahoo", config)
        
        if yf is None:
            raise ImportError("yfinance library is required for YahooProvider. Install with: pip install yfinance")
        
        # Configuration
        self.auto_adjust = self.config.get('auto_adjust', True)
        self.prepost = self.config.get('prepost', False)
        self.threads = self.config.get('threads', True)
        self.proxy = self.config.get('proxy', None)
        
        # Yahoo Finance timeframe mappings
        self._yf_timeframes = {
            '1m': '1m',
            '2m': '2m', 
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '60m',
            '90m': '90m',
            '1h': '1h',
            '1d': '1d',
            '5d': '5d',
            '1wk': '1wk',
            '1mo': '1mo',
            '3mo': '3mo'
        }
        
        logger.info("Initialized Yahoo Finance data provider")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'SPY', '^GSPC')
            timeframe: Data timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of periods (applied after date filtering)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Validate timeframe
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Timeframe {timeframe} not supported by Yahoo Finance")
        
        # Check cache first
        cache_key = self.get_cache_key(symbol, timeframe, start_date, end_date)
        if cache_key in self._cache:
            logger.debug(f"Returning cached data for {cache_key}")
            return self._cache[cache_key]
        
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Determine period and interval
            yf_interval = self._yf_timeframes[timeframe]
            
            # Set default date range if not provided
            if start_date is None and end_date is None:
                # Default lookback based on timeframe
                if timeframe in ['1m', '2m', '5m']:
                    period = '7d'  # Yahoo limits for minute data
                elif timeframe in ['15m', '30m', '60m', '90m', '1h']:
                    period = '60d'
                else:
                    period = '1y'
                
                # Fetch with period
                df = ticker.history(
                    period=period,
                    interval=yf_interval,
                    auto_adjust=self.auto_adjust,
                    prepost=self.prepost,
                    threads=self.threads,
                    proxy=self.proxy
                )
            else:
                # Fetch with date range
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=yf_interval,
                    auto_adjust=self.auto_adjust,
                    prepost=self.prepost,
                    threads=self.threads,
                    proxy=self.proxy
                )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Process the data
            df = self._process_yahoo_data(df)
            
            # Apply limit if specified
            if limit:
                df = df.tail(limit)
            
            # Normalize data
            df = self._normalize_dataframe(df)
            
            # Cache result
            self._cache[cache_key] = df
            
            logger.info(f"Fetched {len(df)} periods for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            raise
    
    def _process_yahoo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw Yahoo Finance data to standard format.
        
        Args:
            df: Raw DataFrame from yfinance
            
        Returns:
            DataFrame with standardized columns and index
        """
        # Yahoo Finance returns data with proper column names already
        # Just need to standardize case and ensure proper index
        
        # Rename columns to lowercase
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure index is named properly
        df.index.name = 'time'
        
        # Select only OHLCV columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if len(available_columns) != len(required_columns):
            missing = set(required_columns) - set(available_columns)
            logger.warning(f"Missing columns: {missing}")
        
        df = df[available_columns]
        
        # Handle any timezone issues by converting to UTC
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC')
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of common symbols (Yahoo doesn't provide a complete list).
        
        This returns a sample of common symbols. For actual symbol validation,
        use validate_symbol() which attempts to fetch data.
        
        Returns:
            List of common stock symbols
        """
        # Return a list of popular symbols as examples
        common_symbols = [
            # Major stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            # Major ETFs
            'SPY', 'QQQ', 'VTI', 'VEA', 'VWO', 'BND', 'VNQ',
            # Major indices
            '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX',
            # Commodities
            'GLD', 'SLV', 'USO', 'UNG',
            # Currencies
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X'
        ]
        
        return common_symbols
    
    def get_available_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes for Yahoo Finance.
        
        Returns:
            List of supported timeframes
        """
        return list(self._yf_timeframes.keys())
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists by attempting to fetch basic info.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid info
            return 'symbol' in info or 'shortName' in info or len(info) > 1
            
        except Exception:
            return False
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a symbol.
        
        Args:
            symbol: Symbol to get info for
            
        Returns:
            Dictionary with symbol information
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {}
    
    def fetch_multiple_symbols(
        self, 
        symbols: List[str], 
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols efficiently.
        
        Args:
            symbols: List of symbols to fetch
            timeframe: Data timeframe
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        try:
            # Use yfinance's bulk download feature
            yf_interval = self._yf_timeframes[timeframe]
            
            if start_date is None and end_date is None:
                # Use period
                period = '1y' if timeframe in ['1d', '5d', '1wk', '1mo'] else '60d'
                data = yf.download(
                    symbols,
                    period=period,
                    interval=yf_interval,
                    auto_adjust=self.auto_adjust,
                    prepost=self.prepost,
                    threads=self.threads,
                    group_by='ticker'
                )
            else:
                data = yf.download(
                    symbols,
                    start=start_date,
                    end=end_date,
                    interval=yf_interval,
                    auto_adjust=self.auto_adjust,
                    prepost=self.prepost,
                    threads=self.threads,
                    group_by='ticker'
                )
            
            # Process results
            results = {}
            for symbol in symbols:
                if len(symbols) == 1:
                    # Single symbol returns different structure
                    symbol_data = data
                else:
                    symbol_data = data[symbol]
                
                if not symbol_data.empty:
                    symbol_data = self._process_yahoo_data(symbol_data)
                    results[symbol] = self._normalize_dataframe(symbol_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching multiple symbols: {e}")
            return {}
    
    def __repr__(self) -> str:
        return f"YahooProvider(auto_adjust={self.auto_adjust})"
