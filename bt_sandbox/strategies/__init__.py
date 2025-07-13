"""
Strategies module for BT Sandbox.

This module contains various trading strategy implementations
for use with the Backtrader backtesting framework.
"""

from .rsi_strategy import RSIBacktraderStrategy
from .ma_crossover_strategy import MovingAverageCrossStrategy
from .bollinger_bands_strategy import BollingerBandsStrategy
from .improved_strategies import ImprovedRSIStrategy, ImprovedMAStrategy, SignalMAStrategy

__all__ = [
    # Legacy strategies
    "RSIBacktraderStrategy",
    "MovingAverageCrossStrategy", 
    "BollingerBandsStrategy",
    # Enhanced strategies (recommended)
    "ImprovedRSIStrategy",
    "ImprovedMAStrategy",
    "SignalMAStrategy",
]
