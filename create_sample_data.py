import pandas as pd
import numpy as np

# Generate sample OHLCV data for testing
np.random.seed(42)
dates = pd.date_range('2025-01-01', periods=1000, freq='1H')
initial_price = 50000

# Generate realistic OHLCV data
prices = []
current_price = initial_price

for i in range(1000):
    # Random walk with slight upward trend
    change = np.random.normal(0.001, 0.02)  # Small trend, moderate volatility
    current_price *= (1 + change)
    
    # Generate OHLC from current price
    volatility = 0.005
    high = current_price * (1 + np.random.uniform(0, volatility))
    low = current_price * (1 - np.random.uniform(0, volatility))
    open_price = current_price * (1 + np.random.uniform(-volatility/2, volatility/2))
    close_price = current_price
    volume = np.random.uniform(100, 1000)
    
    prices.append({
        'time': dates[i],
        'open': open_price,
        'high': high,
        'low': low,
        'close': close_price,
        'volume': volume
    })

df = pd.DataFrame(prices)
df.to_csv('data_pipeline/market_data/BTCUSDT_1h.csv', index=False)
print(f"Created sample data with {len(df)} rows")
print(df.head())
