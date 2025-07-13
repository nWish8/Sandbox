"""
Binance exchange data provider for the Sandbox trading stack.

This provider fetches real-time and historical market data from Binance
via the ccxt library, replacing the existing crypto data fetching logic.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

try:
    import ccxt
except ImportError:
    ccxt = None

from .base_provider import BaseDataProvider

logger = logging.getLogger(__name__)


class BinanceProvider(BaseDataProvider):
    """
    Data provider for Binance cryptocurrency exchange.
    
    Fetches OHLCV data using the ccxt library with proper rate limiting
    and error handling. Supports both spot and futures markets.
    
    Configuration options:
    - market_type: 'spot' or 'futures' (default: 'spot')
    - rate_limit: Milliseconds between requests (default: 1200)
    - sandbox: Use sandbox/testnet mode (default: False)
    - api_key: API key for authenticated endpoints (optional)
    - api_secret: API secret for authenticated endpoints (optional)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Binance data provider.
        
        Args:
            config: Configuration dictionary with Binance-specific options
        """
        super().__init__("Binance", config)
        
        if ccxt is None:
            raise ImportError("ccxt library is required for BinanceProvider. Install with: pip install ccxt")
        
        # Configuration
        self.market_type = self.config.get('market_type', 'spot')
        self.rate_limit = self.config.get('rate_limit', 1200)
        self.sandbox = self.config.get('sandbox', False)
        
        # Initialize exchange
        self._init_exchange()
        
        # Cache for exchange info
        self._markets = None
        self._timeframes = None
    
    def _init_exchange(self):
        """Initialize the ccxt exchange instance."""
        try:
            exchange_config = {
                'rateLimit': self.rate_limit,
                'enableRateLimit': True,
                'timeout': 30000,
            }
            
            # Add API credentials if provided
            if 'api_key' in self.config:
                exchange_config['apiKey'] = self.config['api_key']
            if 'api_secret' in self.config:
                exchange_config['secret'] = self.config['api_secret']
            
            # Initialize exchange based on market type
            if self.market_type == 'futures':
                self.exchange = ccxt.binance(exchange_config)
                self.exchange.set_sandbox_mode(self.sandbox)
                # Set to futures mode
                if hasattr(self.exchange, 'options'):
                    self.exchange.options['defaultType'] = 'future'
            else:
                self.exchange = ccxt.binance(exchange_config)
                self.exchange.set_sandbox_mode(self.sandbox)
            
            logger.info(f"Initialized Binance {self.market_type} exchange")
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance exchange: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of candles (default: 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Validate inputs
        if not self.validate_symbol(symbol):
            raise ValueError(f"Symbol {symbol} not available on Binance {self.market_type}")
        
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Timeframe {timeframe} not supported by Binance")
        
        # Check cache first
        cache_key = self.get_cache_key(symbol, timeframe, start_date, end_date)
        if cache_key in self._cache:
            logger.debug(f"Returning cached data for {cache_key}")
            return self._cache[cache_key]
        
        try:
            # If we have both start and end dates, fetch in chunks
            if start_date and end_date:
                data = self._fetch_date_range(symbol, timeframe, start_date, end_date)
            else:
                # Single request with limit
                if limit is None:
                    limit = 1000
                
                since = None
                if start_date:
                    since = int(start_date.timestamp() * 1000)
                
                data = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )
            
            if not data:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = self._process_ohlcv_data(data)
            
            # Apply end date filter if specified
            if end_date:
                df = df[df.index <= end_date]
            
            # Normalize data
            df = self._normalize_dataframe(df)
            
            # Cache result
            self._cache[cache_key] = df
            
            logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Binance: {e}")
            raise
    
    def _fetch_date_range(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List:
        """
        Fetch data across a date range by making multiple requests.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            List of OHLCV data
        """
        all_data = []
        current_start = start_date
        
        # Calculate timeframe in milliseconds
        timeframe_ms = self._timeframe_to_ms(timeframe)
        max_candles = 1000  # Binance limit
        
        while current_start < end_date:
            since = int(current_start.timestamp() * 1000)
            
            # Calculate how many candles we can fetch
            time_diff_ms = int(end_date.timestamp() * 1000) - since
            candles_needed = min(max_candles, time_diff_ms // timeframe_ms)
            
            if candles_needed <= 0:
                break
            
            logger.debug(f"Fetching {candles_needed} candles from {current_start}")
            
            try:
                data = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=candles_needed
                )
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # Update start time for next batch
                last_timestamp = data[-1][0]
                current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=timeframe_ms)
                
                # Rate limiting
                time.sleep(self.rate_limit / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in batch fetch: {e}")
                break
        
        return all_data
    
    def _process_ohlcv_data(self, data: List) -> pd.DataFrame:
        """
        Convert raw OHLCV data to DataFrame.
        
        Args:
            data: List of OHLCV arrays from ccxt
            
        Returns:
            DataFrame with proper datetime index
        """
        df = pd.DataFrame(
            data, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df.index.name = 'time'
        
        # Convert to proper data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """
        Convert timeframe string to milliseconds.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h')
            
        Returns:
            Timeframe in milliseconds
        """
        timeframe_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000,
        }
        
        return timeframe_map.get(timeframe, 60 * 1000)  # Default to 1m
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading pairs from Binance.
        
        Returns:
            List of available symbols
        """
        try:
            if self._markets is None:
                self._markets = self.exchange.load_markets()
            
            return list(self._markets.keys())
            
        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            return []
    
    def get_available_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes for Binance.
        
        Returns:
            List of supported timeframes
        """
        if self._timeframes is None:
            try:
                self._timeframes = list(self.exchange.timeframes.keys())
            except Exception as e:
                logger.error(f"Error fetching timeframes: {e}")
                # Default Binance timeframes
                self._timeframes = [
                    '1m', '3m', '5m', '15m', '30m', 
                    '1h', '2h', '4h', '6h', '8h', '12h',
                    '1d', '3d', '1w', '1M'
                ]
        
        return self._timeframes
    
    def get_market_type(self) -> str:
        """Get the current market type (spot/futures)."""
        return self.market_type
    
    def __repr__(self) -> str:
        return f"BinanceProvider(market_type='{self.market_type}')"
