import pandas as pd


def load_csv(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV under market_data/."""
    path = f"market_data/{symbol}_{timeframe}.csv"
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return df
