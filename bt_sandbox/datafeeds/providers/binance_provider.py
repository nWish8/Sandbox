"""
Binance data provider for cryptocurrency data.

This provider fetches data from Binance API using the ccxt library.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
import backtrader as bt

from .base_provider import BaseProvider

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


class BinanceProvider(BaseProvider):
    """Binance cryptocurrency data provider."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Binance provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt library required for Binance provider. Install with: pip install ccxt")
        
        self.exchange = ccxt.binance({
            'apiKey': config.get('api_key'),
            'secret': config.get('secret'),
            'sandbox': config.get('sandbox', False),
            'rateLimit': 1200,  # Be respectful to the API
        })
    
    def get_available_symbols(self) -> List[str]:
        """
        Get available trading symbols from Binance.
        
        Returns:
            List of available symbols
        """
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            print(f"❌ Error fetching Binance symbols: {e}")
            return []
    
    def fetch_data(self, symbol: str, timeframe: str = '1d', **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe ('1m', '5m', '1h', '1d', etc.)
            **kwargs: Additional parameters (limit, since)
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=kwargs.get('limit', 500),
                since=kwargs.get('since')
            )
            
            if not ohlcv:
                print(f"⚠️  No data received for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching {symbol} from Binance: {e}")
            return None
    
    def get_bt_data_feed(self, symbol: str, **kwargs) -> Optional[bt.feeds.PandasData]:
        """
        Get Backtrader data feed from Binance data.
        
        Note: Binance doesn't have a native Backtrader feed, so we use PandasData.
        
        Args:
            symbol: Trading symbol
            **kwargs: Additional parameters
            
        Returns:
            Backtrader PandasData feed or None if error
        """
        # Fetch data first
        data = self.fetch_data(symbol, **kwargs)
        
        if data is None:
            return None
        
        try:
            # Create PandasData feed
            params = {
                'dataname': data,
                'timeframe': kwargs.get('timeframe', bt.TimeFrame.Days),
                'compression': kwargs.get('compression', 1),
            }
            
            # Add date filtering if provided
            if 'fromdate' in kwargs:
                params['fromdate'] = kwargs['fromdate']
            if 'todate' in kwargs:
                params['todate'] = kwargs['todate']
            
            data_feed = bt.feeds.PandasData(**params)
            
            return data_feed
            
        except Exception as e:
            print(f"❌ Error creating Binance data feed for {symbol}: {e}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is available from Binance.
        
        Args:
            symbol: Trading symbol to validate (e.g., 'BTC/USDT')
            
        Returns:
            True if symbol is valid for Binance
        """
        if not symbol or len(symbol.strip()) == 0:
            return False
        
        try:
            # Check if symbol exists in Binance markets
            if hasattr(self.exchange, 'markets') and self.exchange.markets:
                return symbol in self.exchange.markets
            else:
                # Fallback: basic format validation for crypto pairs
                return '/' in symbol and len(symbol.split('/')) == 2
        except Exception:
            # If exchange is not available, do basic validation
            return '/' in symbol and len(symbol.split('/')) == 2
