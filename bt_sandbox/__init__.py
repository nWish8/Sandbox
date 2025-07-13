"""
BT Sandbox - A comprehensive backtesting framework for trading strategies.

This package provides a modular framework for developing, testing, and analyzing
trading strategies using Backtrader as the core backtesting engine.

Modules:
    strategies: Trading strategy implementations
    datafeeds: Data acquisition and management
    backtesting: Backtesting engine and utilities
    utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "BT Sandbox Team"

# Import main components for easy access
from .strategies import RSIBacktraderStrategy, MovingAverageCrossStrategy, BollingerBandsStrategy
from .backtesting import SandboxEngine
from .datafeeds import DataManager
from .datafeeds.providers import YahooFinanceProvider, CSVProvider

__all__ = [
    # Core strategies (now using best practices)
    "RSIBacktraderStrategy",
    "MovingAverageCrossStrategy",
    "BollingerBandsStrategy", 
    # Core engine (now enhanced)
    "SandboxEngine", 
    "DataManager",
    # Providers
    "YahooFinanceProvider",
    "CSVProvider",
]
