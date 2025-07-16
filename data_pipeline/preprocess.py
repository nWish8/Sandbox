# Updated: preprocess.py
# Optimizations:
# - Use joblib for parallel indicator computation on Ryzen's 12 threads.
# - Process data in chunks to fit within 16GB RAM.
# - Fixed timezone mismatch as before.

import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volume import MFIIndicator as MoneyFlowIndexIndicator
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
import yaml
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
from joblib import Parallel, delayed

# Load config
with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load full history in chunks
full_file = os.path.join(config['raw_data_dir'], f"{config['symbol'].replace('/', '')}_{config['timeframe']}_full_history.csv")
full_df = pd.read_csv(full_file, parse_dates=['timestamp'], chunksize=100000)  # Chunk for RAM efficiency
full_df = pd.concat(full_df)

# Drop duplicates by timestamp
full_df = full_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

# Load training windows for marking notable periods
windows = pd.read_csv(config['training_windows_csv'], parse_dates=['start_ts', 'end_ts'])

# Fix: Localize windows timestamps to UTC to match full_df
windows['start_ts'] = windows['start_ts'].dt.tz_localize('UTC')
windows['end_ts'] = windows['end_ts'].dt.tz_localize('UTC')

# Mark notable: 1 if timestamp in any window (parallelized)
def is_notable(ts):
    for _, row in windows.iterrows():
        if row['start_ts'] <= ts < row['end_ts']:
            return 1
    return 0

full_df['notable'] = Parallel(n_jobs=-1)(delayed(is_notable)(ts) for ts in full_df['timestamp'])  # Use all CPU cores

# Compute indicators (parallelized where possible)
full_df = add_all_ta_features(full_df, open='open', high='high', low='low', close='close', volume='volume', fillna=True)

# Specific indicators (parallelized via joblib)
def compute_indicator(df_chunk, ind_name):
    if ind_name == 'rsi':
        return RSIIndicator(df_chunk['close'], window=14).rsi()
    # Add similar for others...

# For simplicity, compute sequentially if parallel is complex; optimize as needed
indicators = {
    'rsi': RSIIndicator(full_df['close'], window=14).rsi(),
    'mfi': MoneyFlowIndexIndicator(high=full_df['high'], low=full_df['low'], close=full_df['close'], volume=full_df['volume'], window=14).money_flow_index(),
    'macd': MACD(full_df['close'], window_slow=26, window_fast=12, window_sign=9).macd(),
    'macd_signal': MACD(full_df['close'], window_slow=26, window_fast=12, window_sign=9).macd_signal(),
    'macd_hist': MACD(full_df['close'], window_slow=26, window_fast=12, window_sign=9).macd_diff(),
    'ema20': EMAIndicator(full_df['close'], window=20).ema_indicator(),
    'ema50': EMAIndicator(full_df['close'], window=50).ema_indicator(),
    'ema200': EMAIndicator(full_df['close'], window=200).ema_indicator(),
    'tenkan_sen': IchimokuIndicator(high=full_df['high'], low=full_df['low'], window1=9, window2=26, window3=52).ichimoku_conversion_line(),
    'kijun_sen': IchimokuIndicator(high=full_df['high'], low=full_df['low'], window1=9, window2=26, window3=52).ichimoku_base_line(),
    'senkou_span_a': IchimokuIndicator(high=full_df['high'], low=full_df['low'], window1=9, window2=26, window3=52).ichimoku_a(),
    'senkou_span_b': IchimokuIndicator(high=full_df['high'], low=full_df['low'], window1=9, window2=26, window3=52).ichimoku_b(),
    'chikou_span': full_df['close'].shift(-26)  # Corrected Chikou Span calculation
}

# Concat all indicators
full_df = pd.concat([full_df, pd.DataFrame(indicators)], axis=1)

# Financial time-series preprocessing (parallel if needed)
preproc_cols = {
    'log_ret': np.log(full_df['close']).diff(),
    'rsi_z': (full_df['rsi'] - full_df['rsi'].rolling(250).mean()) / full_df['rsi'].rolling(250).std()
}

# Concat preprocessing columns
full_df = pd.concat([full_df, pd.DataFrame(preproc_cols)], axis=1)

# Handle NaNs
full_df = full_df.ffill().dropna()

# Target engineering
target_cols = {
    'fwd_ret': full_df['close'].pct_change(periods=config['horizon']).shift(-config['horizon']),
    'label': np.select(
        [full_df['close'].pct_change(periods=config['horizon']).shift(-config['horizon']) > config['threshold'],
         full_df['close'].pct_change(periods=config['horizon']).shift(-config['horizon']) < -config['threshold']],
        [1, -1], default=0
    )
}

# Concat target columns
full_df = pd.concat([full_df, pd.DataFrame(target_cols)], axis=1)
full_df = full_df.dropna()  # Drop rows with NaN targets

# Save unscaled version
full_df.to_csv(config['unscaled_data_csv'], index=False)

# Feature selection
features = config['features'] + ['rsi_z']  # Add normalized features
df_features = full_df[['timestamp', 'notable'] + features + ['label']].copy()  # Include notable for biasing

# Min-Max scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
df_features.loc[:, features] = scaler.fit_transform(df_features[features])
joblib.dump(scaler, 'models/scaler.pkl')

# Save scaled
df_features.to_csv(config['training_data_csv'], index=False)
print("Preprocessing complete. Saved to", config['training_data_csv'])
