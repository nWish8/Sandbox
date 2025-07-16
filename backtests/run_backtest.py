# Updated: run_backtest.py
# Optimizations:
# - Move model and data to GPU for inference.
# - Use AMP for mixed precision inference.
# - Parallel data loading with num_workers=12.

import torch
import pandas as pd
import yaml
import sys
import os
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Add project root to path for imports (adjust if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.train import BTCLSTM, BTCWindowDataset  # Relative import from project root
from backtesting import Backtest, Strategy

# Load config
with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use RTX 4050

# Generate signals (inference)
def generate_signals():
    # Load full DF to get timestamps (using unscaled for original timestamps)
    full_df = pd.read_csv(config['unscaled_data_csv'], parse_dates=['timestamp'])

    # Date-based split for test
    test_df = full_df[(full_df['timestamp'] >= config['test_start']) & (full_df['timestamp'] < config['test_end'])].reset_index(drop=True)

    # Create dataset for test split (uses scaled training_data_csv internally)
    ds = BTCWindowDataset(config['training_data_csv'], config['seq_len'], 'models/scaler.pkl', 'test')
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=12)  # Parallel loading

    model = BTCLSTM(n_features=len(ds.features)).to(device)
    model.load_state_dict(torch.load('models/btc_lstm.pth'))
    model.eval()

    signals = []
    with torch.no_grad():
        for X, _ in loader:  # Ignore y for inference
            X = X.to(device)
            with autocast():  # AMP
                out = model(X)
            preds = torch.argmax(out, dim=1).cpu().tolist()
            # Reverse remap: 0 -> -1 (sell), 1 -> 0 (hold), 2 -> 1 (buy)
            batch_signals = [-1 if p == 0 else (0 if p == 1 else 1) for p in preds]
            signals.extend(batch_signals)

    # Align with test_df, skipping seq_len rows
    aligned_df = test_df.iloc[config['seq_len']:].copy()

    # Debug: Check lengths
    print(f"Generated signals length: {len(signals)}")
    print(f"Aligned DataFrame length: {len(aligned_df)}")
    if len(signals) != len(aligned_df):
        raise ValueError("Signal length mismatch after alignment—check seq_len or date ranges")

    aligned_df['signal'] = signals
    aligned_df[['timestamp', 'signal']].to_csv('backtests/inference_signals.csv', index=False)
    print("Signals generated and saved to backtests/inference_signals.csv")

class LSTMStrategy(Strategy):
    def init(self):
        sig_df = pd.read_csv('backtests/inference_signals.csv', parse_dates=['timestamp']).set_index('timestamp')
        self.signal = self.I(lambda: sig_df['signal'].reindex(self.data.df.index, method='ffill').values)

    def next(self):
        sig = self.signal[-1]
        if sig == 1 and not self.position.is_long:
            self.position.close()
            self.buy(size=1)
        elif sig == -1 and not self.position.is_short:
            self.position.close()
            self.sell(size=1)

# Run backtest
generate_signals()  # First generate

# Use unscaled data for backtest, filtered to test period
unscaled_df = pd.read_csv(config['unscaled_data_csv'], parse_dates=['timestamp'])
test_df = unscaled_df[(unscaled_df['timestamp'] >= config['test_start']) & (unscaled_df['timestamp'] < config['test_end'])]
test_df = test_df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]
test_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # backtesting.py format

bt = Backtest(test_df, LSTMStrategy, commission=0.0004, trade_on_close=True)
stats = bt.run()
print(stats)
bt.plot()  # Uses matplotlib under the hood
