from data_pipeline.data_fetcher import load_csv
from strategies.moving_average_cross import MovingAverageCross
import numpy as np

# Test with very short MA periods
data = load_csv('BTCUSDT_1m_2025-06-08_to_2025-07-08', '')
print(f'Data shape: {data.shape}')

# Create a simple debug strategy
class DebugMovingAverageCross:
    def __init__(self, short_win: int = 5, long_win: int = 20):
        self.short_win = short_win
        self.long_win = long_win
        self.prices: list[float] = []
        self.last_signal = "HOLD"
        self.signals = []

    def on_bar(self, bar):
        self.prices.append(float(bar.close))
        if len(self.prices) >= self.long_win:
            short_ma = np.mean(self.prices[-self.short_win:])
            long_ma = np.mean(self.prices[-self.long_win:])
            
            # Store for debugging
            self.signals.append((len(self.prices), short_ma, long_ma, short_ma > long_ma))
            
            # Generate signals only when crossing occurs
            if short_ma > long_ma and self.last_signal != "BUY":
                self.last_signal = "BUY"
                print(f"BUY signal at bar {len(self.prices)}: short_ma={short_ma:.2f}, long_ma={long_ma:.2f}")
                return "BUY"
            elif short_ma < long_ma and self.last_signal != "SELL":
                self.last_signal = "SELL"
                print(f"SELL signal at bar {len(self.prices)}: short_ma={short_ma:.2f}, long_ma={long_ma:.2f}")
                return "SELL"
                
        return "HOLD"

# Test the debug strategy
strategy = DebugMovingAverageCross(short_win=5, long_win=20)

# Run through first 1000 bars
for i, bar in enumerate(data.itertuples()):
    if i >= 1000:
        break
    result = strategy.on_bar(bar)
    if result != "HOLD":
        print(f"Signal: {result}")

print(f"\nChecked {min(1000, len(data))} bars")
print(f"Generated {len([s for s in strategy.signals if len(s) > 0])} MA calculations")

# Show some MA values
if strategy.signals:
    print("Sample MA values:")
    for i in range(0, min(50, len(strategy.signals)), 10):
        bar_num, short_ma, long_ma, is_above = strategy.signals[i]
        print(f"Bar {bar_num}: Short MA={short_ma:.2f}, Long MA={long_ma:.2f}, Above={is_above}")
