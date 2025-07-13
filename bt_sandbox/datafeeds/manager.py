"""
Data manager for the Sandbox trading stack.

This module provides a unified interface for accessing market data from
multiple providers, with caching, fallback mechanisms, and data validation.
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class DataManager:
    """
    Unified data manager for the Sandbox trading stack.
    
    Manages multiple data providers and provides a single interface for
    fetching market data with automatic provider selection, caching,
    and fallback mechanisms.
    
    Features:
    - Multiple provider support (CSV, Binance, Yahoo Finance)
    - Automatic provider selection based on symbol/market
    - Data caching and validation
    - Fallback mechanisms
    - Unified data format across providers
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data manager.
        
        Args:
            config: Configuration dictionary with provider settings:
                - csv: Configuration for CSV provider
                - binance: Configuration for Binance provider  
                - yahoo: Configuration for Yahoo provider
                - default_provider: Default provider name
                - cache_enabled: Enable/disable caching (default: True)
        """
        self.config = config or {}
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.default_provider = self.config.get('default_provider', 'csv')
        
        # Initialize providers
        self.providers: Dict[str, BaseDataProvider] = {}
        self._init_providers()
        
        # Global cache
        self._global_cache = {}
        
        logger.info(f"DataManager initialized with providers: {list(self.providers.keys())}")
    
    def _init_providers(self):
        """Initialize all configured data providers."""
        # Always initialize CSV provider (for local data)
        try:
            csv_config = self.config.get('csv', {})
            self.providers['csv'] = CSVProvider(csv_config)
            logger.info("CSV provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize CSV provider: {e}")
        
        # Initialize Binance provider if configured
        if 'binance' in self.config:
            try:
                binance_config = self.config['binance']
                self.providers['binance'] = BinanceProvider(binance_config)
                logger.info("Binance provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Binance provider: {e}")
        
        # Initialize Yahoo provider if configured
        if 'yahoo' in self.config:
            try:
                yahoo_config = self.config['yahoo']
                self.providers['yahoo'] = YahooProvider(yahoo_config)
                logger.info("Yahoo provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Yahoo provider: {e}")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        provider: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data using the best available provider.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of candles
            provider: Specific provider to use (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Determine provider to use
        if provider:
            if provider not in self.providers:
                raise ValueError(f"Provider '{provider}' not available")
            selected_provider = provider
        else:
            selected_provider = self._select_provider(symbol, timeframe)
        
        # Check global cache first
        if self.cache_enabled:
            cache_key = f"{selected_provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{limit}"
            if cache_key in self._global_cache:
                logger.debug(f"Returning cached data for {cache_key}")
                return self._global_cache[cache_key]
        
        # Fetch data from provider
        try:
            provider_instance = self.providers[selected_provider]
            data = provider_instance.fetch_ohlcv(
                symbol, timeframe, start_date, end_date, limit
            )
            
            # Cache result
            if self.cache_enabled and not data.empty:
                self._global_cache[cache_key] = data
            
            logger.info(f"Fetched {len(data)} rows from {selected_provider} for {symbol}_{timeframe}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from {selected_provider}: {e}")
            
            # Try fallback providers
            if not provider:  # Only use fallback if provider wasn't explicitly specified
                return self._fetch_with_fallback(symbol, timeframe, start_date, end_date, limit, selected_provider)
            else:
                raise
    
    def _select_provider(self, symbol: str, timeframe: str) -> str:
        """
        Select the best provider for a given symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Provider name to use
        """
        # Provider selection logic
        
        # 1. Check for crypto symbols (contain '/' or 'USDT')
        if '/' in symbol or 'USDT' in symbol.upper():
            if 'binance' in self.providers:
                return 'binance'
        
        # 2. Check for stock symbols (traditional format)
        if symbol.replace('^', '').replace('.', '').isalpha() or '^' in symbol:
            if 'yahoo' in self.providers:
                return 'yahoo'
        
        # 3. Check if CSV provider has the data
        if 'csv' in self.providers:
            csv_provider = self.providers['csv']
            if csv_provider.validate_symbol(symbol):
                return 'csv'
        
        # 4. Fall back to default provider
        if self.default_provider in self.providers:
            return self.default_provider
        
        # 5. Use first available provider
        if self.providers:
            return list(self.providers.keys())[0]
        
        raise ValueError("No data providers available")
    
    def _fetch_with_fallback(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: Optional[int],
        failed_provider: str
    ) -> pd.DataFrame:
        """
        Try fallback providers when primary provider fails.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            limit: Data limit
            failed_provider: Provider that failed
            
        Returns:
            DataFrame with OHLCV data or empty DataFrame if all fail
        """
        # Try remaining providers
        remaining_providers = [p for p in self.providers.keys() if p != failed_provider]
        
        for provider_name in remaining_providers:
            try:
                logger.info(f"Trying fallback provider: {provider_name}")
                provider = self.providers[provider_name]
                
                # Check if provider supports this symbol
                if not provider.validate_symbol(symbol):
                    continue
                
                data = provider.fetch_ohlcv(symbol, timeframe, start_date, end_date, limit)
                if not data.empty:
                    logger.info(f"Successfully fetched data from fallback provider: {provider_name}")
                    return data
                    
            except Exception as e:
                logger.warning(f"Fallback provider {provider_name} also failed: {e}")
                continue
        
        logger.error(f"All providers failed for {symbol}_{timeframe}")
        return pd.DataFrame()
    
    def get_available_symbols(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get available symbols from all or specific provider(s).
        
        Args:
            provider: Specific provider name (optional)
            
        Returns:
            Dictionary mapping provider names to symbol lists
        """
        results = {}
        
        if provider:
            if provider in self.providers:
                results[provider] = self.providers[provider].get_available_symbols()
        else:
            for name, provider_instance in self.providers.items():
                try:
                    results[name] = provider_instance.get_available_symbols()
                except Exception as e:
                    logger.error(f"Error getting symbols from {name}: {e}")
                    results[name] = []
        
        return results
    
    def get_available_timeframes(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get available timeframes from all or specific provider(s).
        
        Args:
            provider: Specific provider name (optional)
            
        Returns:
            Dictionary mapping provider names to timeframe lists
        """
        results = {}
        
        if provider:
            if provider in self.providers:
                results[provider] = self.providers[provider].get_available_timeframes()
        else:
            for name, provider_instance in self.providers.items():
                try:
                    results[name] = provider_instance.get_available_timeframes()
                except Exception as e:
                    logger.error(f"Error getting timeframes from {name}: {e}")
                    results[name] = []
        
        return results
    
    def validate_symbol(self, symbol: str, provider: Optional[str] = None) -> bool:
        """
        Validate if a symbol is available from any or specific provider.
        
        Args:
            symbol: Symbol to validate
            provider: Specific provider to check (optional)
            
        Returns:
            True if symbol is available, False otherwise
        """
        if provider:
            if provider in self.providers:
                return self.providers[provider].validate_symbol(symbol)
            return False
        else:
            # Check any provider
            for provider_instance in self.providers.values():
                if provider_instance.validate_symbol(symbol):
                    return True
            return False
    
    def add_provider(self, name: str, provider: BaseDataProvider):
        """
        Add a new data provider.
        
        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider
        logger.info(f"Added provider: {name}")
    
    def remove_provider(self, name: str):
        """
        Remove a data provider.
        
        Args:
            name: Provider name to remove
        """
        if name in self.providers:
            del self.providers[name]
            logger.info(f"Removed provider: {name}")
    
    def clear_cache(self, provider: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            provider: Specific provider to clear cache for (optional)
        """
        if provider:
            if provider in self.providers:
                self.providers[provider].clear_cache()
        else:
            # Clear all caches
            self._global_cache.clear()
            for provider_instance in self.providers.values():
                provider_instance.clear_cache()
        
        logger.info(f"Cleared cache for {provider or 'all providers'}")
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all providers.
        
        Returns:
            Dictionary with provider status information
        """
        status = {}
        
        for name, provider in self.providers.items():
            try:
                status[name] = {
                    'available': True,
                    'type': provider.__class__.__name__,
                    'symbols_count': len(provider.get_available_symbols()),
                    'timeframes': provider.get_available_timeframes(),
                    'cache_size': len(provider._cache)
                }
            except Exception as e:
                status[name] = {
                    'available': False,
                    'error': str(e)
                }
        
        return status
    
    def __repr__(self) -> str:
        return f"DataManager(providers={list(self.providers.keys())})"
