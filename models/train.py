import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
import joblib
from sklearn.metrics import f1_score
import numpy as np

EPOCHS = 5  # Adjust as needed

# Load config
with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

class BTCWindowDataset(Dataset):
    def __init__(self, csv_path, seq_len, scaler_path, split='train'):
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        # Temporal split: first 70% train, last 30% val
        split_idx = int(len(df) * 0.7)
        if split == 'train':
            df = df.iloc[:split_idx]
        else:
            df = df.iloc[split_idx:]
        
        self.features = config['features'] + ['rsi_z']  # Match preprocess
        self.X = df[self.features].values.astype('float32')
        self.y = df['label'].values.astype('int64')
        self.seq_len = seq_len
        self.scaler = joblib.load(scaler_path)
        
        # Debug: Print label distribution
        unique, counts = np.unique(self.y, return_counts=True)
        print(f"Label distribution in {split} split: {dict(zip(unique, counts))}")
    
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def remap_label(self, label):
        """Remap labels: -1 (sell) -> 0, 0 (hold) -> 1, 1 (buy) -> 2"""
        if label == -1:
            return 0
        elif label == 0:
            return 1
        elif label == 1:
            return 2
        else:
            raise ValueError(f"Unexpected label: {label}")
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_len]
        y_tgt = self.y[idx + self.seq_len - 1]
        y_remapped = self.remap_label(y_tgt)  # Apply remapping here
        return torch.tensor(X_seq), torch.tensor(y_remapped)

class BTCLSTM(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 3)  # 3 classes: sell(0), hold(1), buy(2)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

# Training function
def train_model():
    train_ds = BTCWindowDataset(config['training_data_csv'], config['seq_len'], 'models/scaler.pkl', 'train')
    val_ds = BTCWindowDataset(config['training_data_csv'], config['seq_len'], 'models/scaler.pkl', 'val')
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=True)
    
    model = BTCLSTM(n_features=len(train_ds.features))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(EPOCHS):  # Adjust epochs
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                out = model(X)
                preds = torch.argmax(out, dim=1).numpy()
                val_preds.extend(preds)
                val_true.extend(y.numpy())
        f1 = f1_score(val_true, val_preds, average='macro')
        print(f"Epoch {epoch}: Val F1 {f1:.4f}")
    
    torch.save(model.state_dict(), 'models/btc_lstm.pth')
    print("Training complete.")

if __name__ == "__main__":
    train_model()
