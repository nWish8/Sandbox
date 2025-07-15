import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volume import ChaikinMoneyFlowIndicator as MoneyFlowIndexIndicator
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
import yaml
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load config
with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load all raw CSVs and concatenate with window_id
dfs = []
for file in os.listdir(config['raw_data_dir']):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(config['raw_data_dir'], file), parse_dates=['timestamp'])
        window_id = file.split('_')[2] + '_' + file.split('_')[3].split('.')[0]  # Extract start_end
        df['window_id'] = window_id
        dfs.append(df)
full_df = pd.concat(dfs).sort_values('timestamp').reset_index(drop=True)

# Compute indicators (using ta library)
full_df = add_all_ta_features(full_df, open='open', high='high', low='low', close='close', volume='volume', fillna=True)

# Specific indicators as per request (compute and collect in a dict to avoid fragmentation)
indicators = {
    'rsi': RSIIndicator(full_df['close'], window=14).rsi(),
    'mfi': MoneyFlowIndexIndicator(high=full_df['high'], low=full_df['low'], close=full_df['close'], volume=full_df['volume'], window=14).chaikin_money_flow(),
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
    'chikou_span': full_df['close'].shift(-26)  # Corrected Chikou Span calculation (typically shifted close)
}

# Concat all indicators at once
full_df = pd.concat([full_df, pd.DataFrame(indicators)], axis=1)

# Financial time-series preprocessing (collect in dict)
preproc_cols = {
    'log_ret': np.log(full_df['close']).diff(),
    'rsi_z': (full_df['rsi'] - full_df['rsi'].rolling(250).mean()) / full_df['rsi'].rolling(250).std()
}

# Concat preprocessing columns
full_df = pd.concat([full_df, pd.DataFrame(preproc_cols)], axis=1)

# Handle NaNs (using ffill instead of deprecated method)
full_df = full_df.ffill().dropna()

# Target engineering (collect in dict)
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

# Feature selection
features = config['features'] + ['rsi_z']  # Add normalized features
df_features = full_df[['timestamp', 'window_id'] + features + ['label']].copy()  # Explicit copy to avoid SettingWithCopy

# Min-Max scaling (use .loc for assignment)
scaler = MinMaxScaler(feature_range=(-1, 1))
df_features.loc[:, features] = scaler.fit_transform(df_features[features])
joblib.dump(scaler, 'models/scaler.pkl')

# Save
df_features.to_csv(config['training_data_csv'], index=False)
print("Preprocessing complete. Saved to", config['training_data_csv'])
