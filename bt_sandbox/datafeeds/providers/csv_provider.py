"""
CSV data provider for local market data files.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import backtrader as bt
from backtrader.feeds import GenericCSVData

from .base_provider import BaseProvider


class CSVProvider(BaseProvider):
    """CSV data provider for local market data files."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CSV provider.
        
        Args:
            config: Configuration dictionary with 'data_dir' key
        """
        super().__init__(config)
        self.data_dir = config.get('data_dir', 'market_data')
    
    def get_available_symbols(self) -> List[str]:
        """
        Get available symbols from CSV files in data directory.
        
        Returns:
            List of symbol names (without .csv extension)
        """
        symbols = []
        
        if not os.path.exists(self.data_dir):
            return symbols
        
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                symbol = file.replace('.csv', '')
                symbols.append(symbol)
        
        return symbols
    
    def fetch_data(self, symbol: str, timeframe: str = '1d', **kwargs) -> Optional[pd.DataFrame]:
        """
        Load CSV data as DataFrame.
        
        Args:
            symbol: Symbol name (CSV filename without extension)
            timeframe: Timeframe (ignored for CSV files)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        csv_file = os.path.join(self.data_dir, f"{symbol}.csv")
        
        if not os.path.exists(csv_file):
            print(f"❌ CSV file not found: {csv_file}")
            return None
        
        try:
            # Load CSV data
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns in {csv_file}: {missing_cols}")
                return None
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading CSV file {csv_file}: {e}")
            return None

    def get_bt_data_feed(self, symbol: str, **kwargs) -> Optional[GenericCSVData]:
        """
        Get Backtrader CSV data feed.
        
        Args:
            symbol: Symbol name (CSV filename without extension)
            **kwargs: Data feed parameters:
                - dtformat: Date format string (default: '%Y-%m-%d %H:%M:%S')
                - timeframe: bt.TimeFrame constant
                - compression: Compression factor
                - fromdate: Start date filter
                - todate: End date filter
                
        Returns:
            Backtrader GenericCSVData feed or None if error
        """
        csv_file = os.path.join(self.data_dir, f"{symbol}.csv")
        
        if not os.path.exists(csv_file):
            print(f"❌ CSV file not found: {csv_file}")
            return None
        
        try:
            # Default CSV parameters
            default_params = {
                'dataname': csv_file,
                'dtformat': kwargs.get('dtformat', '%Y-%m-%d %H:%M:%S'),
                'datetime': 0,  # Column index for datetime
                'open': 1,      # Column index for open
                'high': 2,      # Column index for high  
                'low': 3,       # Column index for low
                'close': 4,     # Column index for close
                'volume': 5,    # Column index for volume
                'openinterest': -1,  # No open interest column
                'timeframe': kwargs.get('timeframe', bt.TimeFrame.Days),
                'compression': kwargs.get('compression', 1),
            }
            
            # Add date filtering if provided
            if 'fromdate' in kwargs:
                default_params['fromdate'] = kwargs['fromdate']
            if 'todate' in kwargs:
                default_params['todate'] = kwargs['todate']
            
            # Override with any provided parameters
            params = {**default_params, **kwargs}
            
            # Create and return the data feed
            data_feed = bt.feeds.GenericCSVData(**params)
            
            return data_feed
            
        except Exception as e:
            print(f"❌ Error creating CSV data feed for {symbol}: {e}")
            return None
    
    def get_pandas_feed(self, symbol: str, **kwargs) -> Optional[bt.feeds.PandasData]:
        """
        Get Backtrader PandasData feed from CSV.
        
        This method loads CSV as DataFrame first, then creates PandasData feed.
        Useful when you need to preprocess the data before feeding to Backtrader.
        
        Args:
            symbol: Symbol name (CSV filename without extension)
            **kwargs: Data feed parameters
            
        Returns:
            Backtrader PandasData feed or None if error
        """
        # Load data as DataFrame first
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
            print(f"❌ Error creating PandasData feed for {symbol}: {e}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is available as CSV file.
        
        Args:
            symbol: Symbol name to validate
            
        Returns:
            True if CSV file exists for symbol
        """
        csv_file = os.path.join(self.data_dir, f"{symbol}.csv")
        return os.path.exists(csv_file)
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1d', start_date=None, end_date=None, limit=None, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data (alias for fetch_data to match manager interface).
        
        Args:
            symbol: Symbol name (CSV filename without extension)
            timeframe: Timeframe (ignored for CSV files)
            start_date: Start date (optional)
            end_date: End date (optional)
            limit: Limit number of rows (optional)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        return self.fetch_data(symbol, timeframe, **kwargs)
