"""
CSV file data provider for the Sandbox trading stack.

This provider handles loading market data from CSV files, supporting the
existing market_data directory structure and various CSV formats.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

from .base_provider import BaseDataProvider

logger = logging.getLogger(__name__)


class CSVProvider(BaseDataProvider):
    """
    Data provider for CSV files containing OHLCV market data.
    
    Supports the existing market_data directory structure and handles
    various CSV formats including those with timestamp/time columns.
    
    Expected CSV format:
    - Columns: timestamp/time, open, high, low, close, volume
    - Or: datetime index with open, high, low, close, volume columns
    
    Configuration options:
    - data_dir: Path to directory containing CSV files (default: 'market_data')
    - date_column: Name of the date/time column (default: auto-detect)
    - filename_pattern: Pattern for CSV filenames (default: '{symbol}_{timeframe}.csv')
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CSV data provider.
        
        Args:
            config: Configuration dictionary with options:
                - data_dir: Path to CSV data directory
                - date_column: Name of date/time column
                - filename_pattern: CSV filename pattern
        """
        super().__init__("CSV", config)
        
        self.data_dir = Path(self.config.get('data_dir', 'market_data'))
        self.date_column = self.config.get('date_column', None)  # Auto-detect if None
        self.filename_pattern = self.config.get('filename_pattern', '{symbol}_{timeframe}.csv')
        
        # Ensure data directory exists
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from a CSV file.
        
        Args:
            symbol: Trading symbol (used to construct filename)
            timeframe: Data timeframe (used to construct filename)
            start_date: Filter data from this date (optional)
            end_date: Filter data to this date (optional)
            limit: Limit number of rows returned (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Handle special case for minute data file
        if symbol == "BTCUSDT_1m_2025-06-08_to_2025-07-08":
            filename = f"{symbol}.csv"
        else:
            filename = self.filename_pattern.format(symbol=symbol, timeframe=timeframe)
        
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Check cache first
        cache_key = self.get_cache_key(symbol, timeframe, start_date, end_date)
        if cache_key in self._cache:
            logger.debug(f"Returning cached data for {cache_key}")
            return self._cache[cache_key]
        
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            # Handle datetime index
            df = self._process_datetime_index(df)
            
            # Ensure required columns exist and rename if necessary
            df = self._standardize_columns(df)
            
            # Apply date filtering
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Apply limit
            if limit:
                df = df.tail(limit)
            
            # Normalize and validate
            df = self._normalize_dataframe(df)
            
            # Cache result
            self._cache[cache_key] = df
            
            logger.info(f"Successfully loaded {len(df)} rows for {symbol}_{timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            raise
    
    def _process_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and set datetime index from various column formats.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            DataFrame with datetime index
        """
        # If already has datetime index, return as-is
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Try to find datetime column
        datetime_column = None
        
        if self.date_column and self.date_column in df.columns:
            datetime_column = self.date_column
        else:
            # Auto-detect datetime column
            for col in ['timestamp', 'time', 'datetime', 'date']:
                if col in df.columns:
                    datetime_column = col
                    break
        
        if datetime_column:
            # Set datetime index
            df = df.set_index(pd.to_datetime(df[datetime_column]))
            df.index.name = 'time'
            df = df.drop(datetime_column, axis=1)
        else:
            # Try to parse first column as datetime
            try:
                df = df.set_index(pd.to_datetime(df.iloc[:, 0]))
                df.index.name = 'time'
                df = df.drop(df.columns[0], axis=1)
            except:
                raise ValueError("Could not find or parse datetime column")
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to expected format.
        
        Args:
            df: DataFrame with raw column names
            
        Returns:
            DataFrame with standardized column names
        """
        # Column name mappings
        column_mappings = {
            'o': 'open', 'Open': 'open',
            'h': 'high', 'High': 'high', 
            'l': 'low', 'Low': 'low',
            'c': 'close', 'Close': 'close',
            'v': 'volume', 'Volume': 'volume', 'vol': 'volume'
        }
        
        # Apply mappings
        df = df.rename(columns=column_mappings)
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from CSV files in data directory.
        
        Returns:
            List of symbols found in CSV files
        """
        if not self.data_dir.exists():
            return []
        
        symbols = set()
        for csv_file in self.data_dir.glob("*.csv"):
            # Extract symbol from filename
            filename = csv_file.stem
            if '_' in filename:
                # Standard format: SYMBOL_TIMEFRAME
                symbol = filename.split('_')[0]
                symbols.add(symbol)
            else:
                # Special cases or single symbol files
                symbols.add(filename)
        
        return sorted(list(symbols))
    
    def get_available_timeframes(self) -> List[str]:
        """
        Get list of available timeframes from CSV files.
        
        Returns:
            List of timeframes found in CSV files
        """
        if not self.data_dir.exists():
            return []
        
        timeframes = set()
        for csv_file in self.data_dir.glob("*.csv"):
            filename = csv_file.stem
            if '_' in filename:
                # Extract timeframe from filename
                parts = filename.split('_')
                if len(parts) >= 2:
                    timeframe = parts[1]
                    timeframes.add(timeframe)
        
        return sorted(list(timeframes))
    
    def get_available_files(self) -> List[str]:
        """
        Get list of all CSV files in the data directory.
        
        Returns:
            List of CSV filenames
        """
        if not self.data_dir.exists():
            return []
        
        return [f.name for f in self.data_dir.glob("*.csv")]
    
    def __repr__(self) -> str:
        return f"CSVProvider(data_dir='{self.data_dir}')"
