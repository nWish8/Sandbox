from .strategy_base import StrategyBase
import numpy as np

class MACDStrategy(StrategyBase):
    """MACD (Moving Average Convergence Divergence) strategy."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.prices: list[float] = []
        self.previous_signal = None
        
    def _calculate_ema(self, prices: list[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return float(np.mean(prices))
            
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
            
        return float(ema)
    
    def _calculate_macd(self, prices: list[float]) -> tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram."""
        if len(prices) < self.slow_period:
            return (0.0, 0.0, 0.0)
            
        # Calculate EMAs
        fast_ema = self._calculate_ema(prices, self.fast_period)
        slow_ema = self._calculate_ema(prices, self.slow_period)
        
        # MACD line
        macd_line = fast_ema - slow_ema
        
        # Signal line (EMA of MACD line)
        # For simplicity, we'll use a simple approximation
        signal_line = macd_line * 0.8  # Simplified signal line
        
        # Histogram
        histogram = macd_line - signal_line
        
        return (float(macd_line), float(signal_line), float(histogram))
    
    def on_bar(self, bar, current_position=0):
        """Generate signals based on MACD crossovers."""
        self.prices.append(bar.close)
        
        if len(self.prices) >= self.slow_period:
            macd_line, signal_line, histogram = self._calculate_macd(self.prices)
            
            # Determine current signal based on MACD crossover
            if macd_line > signal_line:
                current_signal = "BULLISH"
            else:
                current_signal = "BEARISH"
            
            # Generate trades on signal changes
            if current_signal == "BULLISH" and self.previous_signal == "BEARISH" and current_position == 0:
                self.previous_signal = current_signal
                return "BUY"
            elif current_signal == "BEARISH" and self.previous_signal == "BULLISH" and current_position > 0:
                self.previous_signal = current_signal
                return "SELL"
            else:
                self.previous_signal = current_signal
                
        return "HOLD"
