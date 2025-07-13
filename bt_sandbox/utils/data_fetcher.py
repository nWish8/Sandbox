"""
Simple data fetcher for downloading BTC 4-hour data.

This script fetches BTC data in 4-hour intervals and converts it to the appropriate format
for backtesting with the trading system.
"""

import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Optional


def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "4h", limit: int = 500) -> Optional[pd.DataFrame]:
    """
    Fetch kline data from Binance public API.
    
    Args:
        symbol: Trading pair symbol (default: BTCUSDT)
        interval: Kline interval (default: 4h)
        limit: Number of klines to fetch (max 1000)
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    url = "https://api.binance.com/api/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Ensure the index is timezone-naive for Backtrader compatibility
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
        
    except Exception as e:
        print(f"Error fetching data from Binance: {e}")
        return None


def save_to_csv(df: pd.DataFrame, filename: str = "BTCUSDT_4h_recent.csv") -> bool:
    """
    Save DataFrame to CSV file in the market_data directory.
    
    Args:
        df: DataFrame to save
        filename: Output filename
    
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = f"market_data/{filename}"
        # Save with explicit datetime format for better Backtrader compatibility
        df.to_csv(filepath, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Data saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False


if __name__ == "__main__":
    print("Fetching BTC 4-hour data from Binance...")
    
    # Fetch recent 500 4-hour candles (about 83 days)
    df = fetch_binance_klines(symbol="BTCUSDT", interval="4h", limit=500)
    
    if df is not None:
        print(f"Fetched {len(df)} candles")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Save to CSV
        save_to_csv(df, "BTCUSDT_4h_recent.csv")
    else:
        print("Failed to fetch data")
