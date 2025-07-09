import torch
import torch.nn as nn
import numpy as np

class Agent:
    def __init__(self, input_dim, initial_balance=1000, cuda=True):
        self.input_dim = input_dim
        self.balance = initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.holdings = 0
        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 0: hold, 1: buy, 2: sell
        ).to(self.device)
        self.history = []

    def act(self, obs):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            action = torch.argmax(logits, dim=1).item()
        return action

    def update_portfolio(self, action, price):
        # Simple buy/sell/hold logic
        if action == 1 and self.balance > price:  # Buy
            self.position = 1
            self.holdings += 1
            self.balance -= price
        elif action == 2 and self.holdings > 0:  # Sell
            self.position = 0
            self.balance += price
            self.holdings -= 1
        # else: hold
        self.history.append((self.balance, self.holdings, price, action))

    # Placeholder for RL update logic
    def observe_and_train(self, *args, **kwargs):
        pass
