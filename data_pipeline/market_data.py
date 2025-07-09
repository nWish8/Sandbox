# market_data.py
# Fetching historical data

import ccxt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# --- Configurable Section ---
ASSET_TYPE = None   # Set from main.py
SYMBOL = None       # Set from main.py
TIMEFRAME = None    # Set from main.py
# ---------------------------

# Example usage:
# symbols = ['BTC/USDT', 'ETH/USDT']
# timeframes = ['1m', '5m', '1h']
# for symbol in symbols:
#     for tf in timeframes:
#         main('crypto', symbol, tf, None)

# stocks = ['AAPL', 'SPY']
# for symbol in stocks:
#     for tf in ['1m', '5m', '1d']:
#         main('stock', symbol, tf, '7d' if 'm' in tf else '1y')


def fetch_crypto_ohlcv(symbol, timeframe, limit=1000):
    ex = ccxt.binance()
    data = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
    return df

def fetch_stock_ohlcv(symbol, interval, start_date, end_date):
    data = yf.download(symbol, interval=interval, start=start_date, end=end_date)
    data = data.rename(columns=str.lower).reset_index()
    if 'datetime' in data.columns:
        data = data.rename(columns={'datetime':'timestamp'})
    else:
        data = data.rename(columns={'date':'timestamp'})
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return data

def fetch_all_crypto_ohlcv(symbol, timeframe, start_date, end_date, limit=1000):
    ex = ccxt.binance()
    all_data = []
    since = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    print(f"Starting download for {symbol} {timeframe} from {start_date} to {end_date}...")
    last_print = datetime.now()
    while since < end_ts:
        data = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not data:
            break
        # Only keep candles within the range
        data = [row for row in data if row[0] < end_ts]
        all_data += data
        now = datetime.now()
        if (now - last_print).total_seconds() > 2 or len(data) < limit:
            dt = pd.to_datetime(data[-1][0], unit='ms') if data else ''
            sys.stdout.write(f"\rDownloaded: {len(all_data)} rows, last candle: {dt}")
            sys.stdout.flush()
            last_print = now
        if len(data) < limit:
            break
        since = data[-1][0] + 1
    print(f"\nDownload complete. Total rows: {len(all_data)}")
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
    return df

def save_ohlcv_to_csv(df, asset_type, symbol, timeframe, start_date=None, end_date=None):
    os.makedirs('market_data', exist_ok=True)
    name = symbol.replace('/', '') if asset_type == 'crypto' else symbol
    if start_date and end_date:
        file = f"{name}_{timeframe}_{start_date}_to_{end_date}.csv"
    else:
        file = f"{name}_{timeframe}.csv"
    file_path = os.path.join('market_data', file)
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path} ({len(df)} rows)")

def main(asset_type, symbol, timeframe, start_date=None, end_date=None):
    if asset_type == 'crypto':
        print(f"Fetching {symbol} {timeframe} data from Binance (paginated, date range)...")
        df = fetch_all_crypto_ohlcv(symbol, timeframe, start_date, end_date)
        save_ohlcv_to_csv(df, asset_type, symbol, timeframe, start_date, end_date)
    elif asset_type == 'stock':
        print(f"Fetching {symbol} {timeframe} data from Yahoo Finance (date range)...")
        df = fetch_stock_ohlcv(symbol, timeframe, start_date, end_date)
        save_ohlcv_to_csv(df, asset_type, symbol, timeframe, start_date, end_date)
    else:
        print("Asset type must be 'crypto' or 'stock'")

if __name__ == '__main__':
    # Parameters should be set from main.py, not here
    pass
