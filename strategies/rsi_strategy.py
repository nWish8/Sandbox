from .strategy_base import StrategyBase
import numpy as np

class RSIStrategy(StrategyBase):
    """RSI-based trading strategy."""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.prices: list[float] = []
        self.previous_signal = None
        
    def _calculate_rsi(self, prices: list[float]) -> float:
        """Calculate RSI for the given prices."""
        if len(prices) < self.period + 1:
            return 50.0  # Neutral RSI
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas]
        losses = [max(0, -delta) for delta in deltas]
        
        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def on_bar(self, bar, current_position=0):
        """Generate signals based on RSI levels."""
        self.prices.append(bar.close)
        
        if len(self.prices) >= self.period + 1:
            rsi = self._calculate_rsi(self.prices)
            
            # Determine current signal
            if rsi < self.oversold:
                current_signal = "BULLISH"
            elif rsi > self.overbought:
                current_signal = "BEARISH"
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
