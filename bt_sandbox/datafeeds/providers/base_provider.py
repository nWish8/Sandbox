"""
Base provider interface for data sources.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import backtrader as bt


class BaseProvider(ABC):
    """Abstract base class for data providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
    
    @abstractmethod
    def get_available_symbols(self) -> list:
        """Get list of available symbols from this provider."""
        pass
    
    @abstractmethod
    def fetch_data(self, symbol: str, timeframe: str = '1d', **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data as DataFrame.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        pass
    
    @abstractmethod
    def get_bt_data_feed(self, symbol: str, **kwargs) -> Optional[bt.feeds.DataBase]:
        """
        Get Backtrader data feed directly (recommended approach).
        
        Args:
            symbol: Trading symbol
            **kwargs: Additional parameters for the data feed
            
        Returns:
            Backtrader data feed instance or None if error
        """
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is available from this provider.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            True if symbol is available
        """
        pass
