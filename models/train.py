# Updated: train.py
# Optimizations:
# - Use GPU (RTX 4050) with torch.device('cuda').
# - Enable AMP for mixed precision (faster training, lower VRAM use).
# - Increase batch_size to 128 and num_workers=12 for Ryzen's threads.
# - Added early stopping based on F1 score stagnation.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import yaml
import joblib
from sklearn.metrics import f1_score
import numpy as np
from torch.cuda.amp import autocast, GradScaler  # For AMP

EPOCHS = 100
PATIENCE = 10  # Early stopping patience

# Load config
with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use RTX 4050

class BTCWindowDataset(Dataset):
    def __init__(self, csv_path, seq_len, scaler_path, split='train'):
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])

        # Date-based split
        if split == 'train':
            df = df[df['timestamp'] < config['train_end']].reset_index(drop=True)
        elif split == 'val':
            df = df[(df['timestamp'] >= config['val_start']) & (df['timestamp'] < config['val_end'])].reset_index(drop=True)
        else:
            raise ValueError("Invalid split")

        self.features = config['features'] + ['rsi_z']  # Match preprocess
        self.X = df[self.features].values.astype('float32')
        self.y = df['label'].values.astype('int64')
        self.notable = df['notable'].values.astype('float32')  # For weighting
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
        notable_weight = self.notable[idx + self.seq_len - 1]  # For sampler
        return torch.tensor(X_seq), torch.tensor(y_remapped), notable_weight

# Training function
def train_model():
    train_ds = BTCWindowDataset(config['training_data_csv'], config['seq_len'], 'models/scaler.pkl', 'train')
    val_ds = BTCWindowDataset(config['training_data_csv'], config['seq_len'], 'models/scaler.pkl', 'val')

    # Biased sampling for train: higher weight for notable=1
    weights = np.where(train_ds.notable == 1, 5.0, 1.0)  # Bias factor 5 for notable
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler, drop_last=True, num_workers=12)  # Use 12 CPU threads
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, drop_last=True, num_workers=12)

    model = BTCLSTM(n_features=len(train_ds.features)).to(device)  # Move to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler()  # For AMP

    best_f1 = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)  # Move to GPU
            optimizer.zero_grad()
            with autocast():  # AMP
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device)
                with autocast():
                    out = model(X)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y.cpu().numpy())

        f1 = f1_score(val_true, val_preds, average='macro')
        print(f"Epoch {epoch}: Val F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), 'models/btc_lstm.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    print("Training complete.")

class BTCLSTM(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 3)  # 3 classes: sell(0), hold(1), buy(2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

if __name__ == "__main__":
    train_model()
