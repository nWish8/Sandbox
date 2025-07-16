import torch
import pandas as pd
import yaml
import sys
import os

# Add project root to path for imports (adjust if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.train import BTCLSTM, BTCWindowDataset  # Relative import from project root

from backtesting import Backtest, Strategy

# Load config
with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Generate signals (inference)
def generate_signals():
    # Load full DF to get timestamps
    full_df = pd.read_csv(config['training_data_csv'], parse_dates=['timestamp'])
    
    # Replicate split logic from dataset
    split_idx = int(len(full_df) * 0.7)
    val_df = full_df.iloc[split_idx:].reset_index(drop=True)
    
    # Create dataset for val split
    ds = BTCWindowDataset(config['training_data_csv'], config['seq_len'], 'models/scaler.pkl', 'val')
    
    model = BTCLSTM(n_features=len(ds.features))
    model.load_state_dict(torch.load('models/btc_lstm.pth'))
    model.eval()
    
    signals = []
    with torch.no_grad():
        for i in range(len(ds)):
            X, _ = ds[i]  # Ignore y for inference
            X = X.unsqueeze(0)  # Batch dim
            out = model(X)
            pred = torch.argmax(out, dim=1).item()
            # Reverse remap: 0 -> -1 (sell), 1 -> 0 (hold), 2 -> 1 (buy)
            signal = -1 if pred == 0 else (0 if pred == 1 else 1)
            signals.append(signal)
    
    # Align with val_df, skipping seq_len rows
    aligned_df = val_df.iloc[config['seq_len']:].copy()
    
    # Debug: Check lengths
    print(f"Generated signals length: {len(signals)}")
    print(f"Aligned DataFrame length: {len(aligned_df)}")
    
    if len(signals) != len(aligned_df):
        raise ValueError("Signal length mismatch after alignment—check split_idx or seq_len")
    
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
test_df = pd.read_csv(config['training_data_csv'], parse_dates=['timestamp'])  # Use full for backtest
test_df = test_df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]
test_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # backtesting.py format

bt = Backtest(test_df, LSTMStrategy, commission=0.0004, trade_on_close=True)
stats = bt.run()
print(stats)
bt.plot()  # Uses matplotlib under the hood
