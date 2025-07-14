# data_pipeline/manager.py

import pandas as pd
from backtesting.test import GOOG, BTCUSD  # Example: built-in test data

class DataManager:
    def __init__(self):
        # Placeholder for managing multiple datasets
        self.datasets = {
            "GOOG": GOOG,
            "BTCUSD": BTCUSD
            # Add more datasets here as needed
        }

    def get_dataset(self, name="GOOG"):
        """
        Retrieve a dataset by name.
        Args:
            name (str): The name of the dataset to load.
        Returns:
            pd.DataFrame: The requested dataset.
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")
        return self.datasets[name]

    def add_dataset(self, name, df):
        """
        Add a new dataset to the manager.
        Args:
            name (str): Name to assign to the dataset.
            df (pd.DataFrame): The dataset.
        """
        self.datasets[name] = df

# Example usage (for testing only; remove or comment out in production)
if __name__ == "__main__":
    manager = DataManager()
    data = manager.get_dataset("GOOG")
    print(data.head())
