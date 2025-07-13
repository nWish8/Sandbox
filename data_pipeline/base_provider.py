"""
Base data provider interface for the Sandbox trading stack.

This module defines the common interface that all data providers must implement,
ensuring consistent behavior across different data sources.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from datetime import datetime, timedelta


class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers.
    
    Data providers are responsible for fetching, caching, and formatting
    market data from various sources (exchanges, APIs, files, etc.).
    
    All data providers must return data in a standardized format:
    - DataFrame with datetime index
    - Columns: ['open', 'high', 'low', 'close', 'volume']
    - Proper timezone handling
    - Consistent data types
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data provider.
        
        Args:
            name: Human-readable name for this provider
            config: Provider-specific configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._cache = {}
        
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')
            timeframe: Data timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval  
            limit: Maximum number of candles to retrieve
            
        Returns:
            DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            and datetime index
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from this provider.
        
        Returns:
            List of available trading symbols
        """
        pass
    
    @abstractmethod
    def get_available_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes for this provider.
        
        Returns:
            List of supported timeframe strings
        """
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is available from this provider.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            True if symbol is available, False otherwise
        """
        available = self.get_available_symbols()
        return symbol in available
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Validate if a timeframe is supported by this provider.
        
        Args:
            timeframe: Timeframe to validate
            
        Returns:
            True if timeframe is supported, False otherwise
        """
        available = self.get_available_timeframes()
        return timeframe in available
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a DataFrame to the standard Sandbox format.
        
        Ensures consistent column names, data types, and index format.
        
        Args:
            df: Raw DataFrame from data source
            
        Returns:
            Normalized DataFrame
        """
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index")
        
        # Ensure proper data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by index
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        return df[required_columns]
    
    def get_cache_key(self, symbol: str, timeframe: str, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> str:
        """
        Generate a cache key for data storage.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            Cache key string
        """
        key_parts = [self.name, symbol, timeframe]
        if start_date:
            key_parts.append(start_date.strftime('%Y%m%d'))
        if end_date:
            key_parts.append(end_date.strftime('%Y%m%d'))
        return "_".join(key_parts)
    
    def clear_cache(self):
        """Clear the internal data cache."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
