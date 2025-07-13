# BT Sandbox Package Structure

The project has been reorganized into a proper Python package structure for better maintainability and navigation.

## Package Structure

```
bt_sandbox/
├── __init__.py              # Main package initialization
├── __main__.py             # Module entry point (python -m bt_sandbox)
├── main.py                 # Main application logic
├── strategies/             # Trading strategy implementations
│   ├── __init__.py
│   ├── rsi_strategy.py     # RSI reversal cross strategy
│   └── ma_crossover_strategy.py  # Moving average crossover strategy
├── datafeeds/              # Data acquisition and management
│   ├── __init__.py
│   ├── manager.py          # Unified data manager
│   └── providers/          # Data provider implementations
│       ├── __init__.py
│       ├── base_provider.py
│       ├── csv_provider.py
│       ├── binance_provider.py
│       └── yahoo_provider.py
├── backtesting/            # Backtesting engine and utilities
│   ├── __init__.py
│   └── engine.py           # Enhanced Backtrader engine
└── utils/                  # Helper functions and utilities
    ├── __init__.py
    └── data_fetcher.py     # Data fetching utilities
```

## Usage

### Run the main application:
```bash
# Either of these commands will work:
python -m bt_sandbox
python -m bt_sandbox.main
```

### Import components in your code:
```python
from bt_sandbox import RSIBacktraderStrategy, SandboxEngine, DataManager
from bt_sandbox.strategies import MovingAverageCrossStrategy
from bt_sandbox.utils import fetch_binance_klines
```

## Benefits

1. **Better Organization**: Strategies, data providers, and utilities are now in separate modules
2. **Easier Navigation**: No more flat file structure - everything has its place
3. **Modular Design**: Easy to add new strategies, indicators, or data providers
4. **Professional Structure**: Follows Python package best practices
5. **Extensible**: Ready for adding optimizers, performance analyzers, etc.

## Migration

The old flat structure files are still available for reference, but the new package structure is the recommended way to run and extend the system.
