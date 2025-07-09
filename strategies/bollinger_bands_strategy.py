from .strategy_base import StrategyBase
import numpy as np

class BollingerBandsStrategy(StrategyBase):
    """Bollinger Bands mean reversion strategy."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.prices: list[float] = []
        self.previous_signal = None
        
    def _calculate_bollinger_bands(self, prices: list[float]) -> tuple[float, float, float]:
        """Calculate Bollinger Bands: (lower, middle, upper)."""
        if len(prices) < self.period:
            return (0.0, 0.0, 0.0)
            
        recent_prices = prices[-self.period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)
        
        return (float(lower), float(middle), float(upper))
    
    def on_bar(self, bar, current_position=0):
        """Generate signals based on Bollinger Bands."""
        self.prices.append(bar.close)
        
        if len(self.prices) >= self.period:
            lower, middle, upper = self._calculate_bollinger_bands(self.prices)
            current_price = bar.close
            
            # Determine current signal
            if current_price < lower:
                current_signal = "BULLISH"  # Oversold
            elif current_price > upper:
                current_signal = "BEARISH"  # Overbought
            else:
                current_signal = "NEUTRAL"
            
            # Generate trades on signal changes
            if current_signal == "BULLISH" and self.previous_signal != "BULLISH" and current_position == 0:
                self.previous_signal = current_signal
                return "BUY"
            elif current_signal == "BEARISH" and self.previous_signal != "BEARISH" and current_position > 0:
                self.previous_signal = current_signal
                return "SELL"
            else:
                self.previous_signal = current_signal
                
        return "HOLD"
