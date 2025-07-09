import pandas as pd


def load_csv(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV under market_data/."""
    if symbol == "BTCUSDT_1m_2025-06-08_to_2025-07-08":
        path = f"market_data/{symbol}.csv"
    else:
        path = f"market_data/{symbol}_{timeframe}.csv"
    
    df = pd.read_csv(path)
    
    # Handle different timestamp column names
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"]))
        df.index.name = 'time'
        df = df.drop('timestamp', axis=1)
    elif "time" in df.columns:
        df = df.set_index(pd.to_datetime(df["time"]))
        df.index.name = 'time'
        df = df.drop('time', axis=1)
    
    return df
