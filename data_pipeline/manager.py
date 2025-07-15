import ccxt
import pandas as pd
import yaml
from datetime import datetime, timedelta
import os
import json

# Load config
with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})
exchange.headers = {'Accept-Encoding': 'gzip, deflate'}

def download_ohlcv(symbol, timeframe, since, until):
    data = []
    while since < until:
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not bars:
            break
        data.extend(bars)
        since = bars[-1][0] + 1  # Next timestamp
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

# Load training windows
windows = pd.read_csv(config['training_windows_csv'], parse_dates=['start_ts', 'end_ts'])

os.makedirs(config['raw_data_dir'], exist_ok=True)

for idx, row in windows.iterrows():
    start = row['start_ts']
    end = row['end_ts']
    # Add lookback
    lookback_start = start - timedelta(days=config['lookback_days'])
    since = int(lookback_start.timestamp() * 1000)
    until = int(end.timestamp() * 1000)
    
    df = download_ohlcv(config['symbol'], config['timeframe'], since, until)
    
    # Save per window
    filename = f"{config['symbol'].replace('/', '')}_{config['timeframe']}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    filepath = os.path.join(config['raw_data_dir'], filename)
    df.to_csv(filepath, index=False, float_format='%.2f')
    
    # Log provenance (corrected attribute)
    last_timestamp = exchange.lastRestRequestTimestamp if hasattr(exchange, 'lastRestRequestTimestamp') else 0
    log = {'ccxt_version': ccxt.__version__, 'download_time': datetime.now().isoformat(), 'api_latency': last_timestamp}
    with open(filepath.replace('.csv', '.json'), 'w') as f:
        json.dump(log, f)
    
    print(f"Downloaded and saved: {filename}")
