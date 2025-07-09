from .strategy_base import StrategyBase
import numpy as np

class MovingAverageCross(StrategyBase):
    def __init__(self, short_win: int = 50, long_win: int = 200):
        self.short_win = short_win
        self.long_win = long_win
        self.prices: list[float] = []
        self.previous_signal = None  # Track the previous MA signal, not position state

    def on_bar(self, bar, current_position=0):
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            bar: Current price bar
            current_position: Current portfolio position (0 = no position, >0 = long position)
        """
        self.prices.append(bar.close)
        
        if len(self.prices) >= self.long_win:
            short_ma = np.mean(self.prices[-self.short_win:])
            long_ma = np.mean(self.prices[-self.long_win:])
            
            # Determine current signal based on MA relationship
            current_signal = "BULLISH" if short_ma > long_ma else "BEARISH"
            
            # Generate trading actions based on signal changes and current position
            if current_signal == "BULLISH" and self.previous_signal == "BEARISH" and current_position == 0:
                # Signal changed to bullish and we have no position -> BUY
                self.previous_signal = current_signal
                return "BUY"
            elif current_signal == "BEARISH" and self.previous_signal == "BULLISH" and current_position > 0:
                # Signal changed to bearish and we have a position -> SELL
                self.previous_signal = current_signal
                return "SELL"
            else:
                # Update the signal state but don't trade
                self.previous_signal = current_signal
                
        return "HOLD"
