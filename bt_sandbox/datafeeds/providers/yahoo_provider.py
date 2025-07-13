"""
Yahoo Finance data provider using Backtrader's built-in feeds.

This provider leverages bt.feeds.YahooFinanceData for better integration
with Backtrader's date-range filtering, timezone handling, and automatic
adjustment for stock splits.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
import backtrader as bt
import yfinance as yf

from .base_provider import BaseProvider


class YahooFinanceProvider(BaseProvider):
    """Yahoo Finance data provider using Backtrader's native feeds."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Yahoo Finance provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.default_timeframe = config.get('default_timeframe', 'Days')
        self.auto_adjust = config.get('auto_adjust', True)
    
    def get_available_symbols(self) -> List[str]:
        """
        Get available symbols (this is informational for Yahoo Finance).
        
        Returns:
            List of common symbols (in practice, Yahoo supports thousands)
        """
        # Yahoo Finance supports a vast number of symbols
        # Return some common examples for demonstration
        return [
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN',  # Tech stocks
            '^GSPC', '^DJI', '^IXIC',  # Indices  
            'BTC-USD', 'ETH-USD',  # Crypto
            'GLD', 'SPY', 'QQQ'  # ETFs
        ]
    
    def fetch_data(self, symbol: str, timeframe: str = '1d', **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch data as DataFrame (legacy method for compatibility).
        
        Note: Using get_bt_data_feed() is recommended for better Backtrader integration.
        
        Args:
            symbol: Yahoo Finance symbol (e.g., 'AAPL', 'BTC-USD')
            timeframe: Data timeframe ('1d', '1h', etc.)
            **kwargs: Additional parameters (start_date, end_date, period)
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            # Extract parameters
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            period = kwargs.get('period', '1y')  # Default to 1 year
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=timeframe,
                    auto_adjust=self.auto_adjust
                )
            else:
                data = ticker.history(
                    period=period,
                    interval=timeframe,
                    auto_adjust=self.auto_adjust
                )
            
            if data.empty:
                print(f"⚠️  No data received for {symbol}")
                return None
            
            # Standardize column names for consistency
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching {symbol} from Yahoo Finance: {e}")
            return None
    
    def get_bt_data_feed(self, symbol: str, **kwargs) -> Optional[bt.feeds.YahooFinanceData]:
        """
        Get Backtrader Yahoo Finance data feed (recommended approach).
        
        This method uses Backtrader's built-in YahooFinanceData feed which handles:
        - Date range filtering
        - Timezone conversion  
        - Automatic adjustment for stock splits
        - Proper resampling
        
        Args:
            symbol: Yahoo Finance symbol (e.g., 'AAPL', 'BTC-USD')
            **kwargs: Data feed parameters:
                - fromdate: Start date (datetime object)
                - todate: End date (datetime object)  
                - timeframe: bt.TimeFrame constant (Days, Hours, etc.)
                - compression: Compression factor (1, 5, 15, etc.)
                - adjclose: Use adjusted close prices (default: True)
                
        Returns:
            Backtrader YahooFinanceData feed or None if error
        """
        try:
            # Set default parameters
            default_params = {
                'dataname': symbol,
                'fromdate': kwargs.get('fromdate', datetime.now() - timedelta(days=365)),
                'todate': kwargs.get('todate', datetime.now()),
                'timeframe': kwargs.get('timeframe', bt.TimeFrame.Days),
                'compression': kwargs.get('compression', 1),
                'adjclose': kwargs.get('adjclose', self.auto_adjust),
            }
            
            # Override with any provided parameters
            params = {**default_params, **kwargs}
            
            # Create and return the data feed
            data_feed = bt.feeds.YahooFinanceData(**params)
            
            return data_feed
            
        except Exception as e:
            print(f"❌ Error creating Backtrader data feed for {symbol}: {e}")
            return None
    
    def get_intraday_feed(self, symbol: str, timeframe_minutes: int = 60, **kwargs) -> Optional[bt.feeds.YahooFinanceData]:
        """
        Get intraday data feed for higher frequency trading.
        
        Args:
            symbol: Yahoo Finance symbol
            timeframe_minutes: Minutes per bar (1, 5, 15, 30, 60)
            **kwargs: Additional parameters
            
        Returns:
            Backtrader data feed configured for intraday data
        """
        # Map minutes to timeframe and compression
        if timeframe_minutes == 1:
            timeframe = bt.TimeFrame.Minutes
            compression = 1
        elif timeframe_minutes == 5:
            timeframe = bt.TimeFrame.Minutes
            compression = 5
        elif timeframe_minutes == 15:
            timeframe = bt.TimeFrame.Minutes
            compression = 15
        elif timeframe_minutes == 30:
            timeframe = bt.TimeFrame.Minutes
            compression = 30
        elif timeframe_minutes == 60:
            timeframe = bt.TimeFrame.Minutes
            compression = 60
        else:
            print(f"⚠️  Unsupported timeframe: {timeframe_minutes} minutes")
            return None
        
        # Set shorter default date range for intraday data
        default_from = datetime.now() - timedelta(days=30)  # Last 30 days
        
        params = {
            'timeframe': timeframe,
            'compression': compression,
            'fromdate': kwargs.get('fromdate', default_from),
            **kwargs
        }
        
        return self.get_bt_data_feed(symbol, **params)
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is available from Yahoo Finance.
        
        Args:
            symbol: Yahoo Finance symbol to validate
            
        Returns:
            True if symbol is valid (basic check)
        """
        # For Yahoo Finance, we can do a basic format check
        # More sophisticated validation would require an API call
        if not symbol or len(symbol.strip()) == 0:
            return False
        
        # Basic symbol format validation
        return len(symbol) >= 1 and symbol.replace('-', '').replace('^', '').replace('.', '').isalnum()
