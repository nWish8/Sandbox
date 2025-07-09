# Advanced Backtesting Engine

A modular Python backtesting framework with an interactive PyQt6 GUI for trading strategy research and analysis.

## Features

- **Modular Architecture**: Cleanly separated components for backtesting, strategies, data pipeline, and UI
- **Interactive GUI**: PyQt6 interface with finplot candlestick charts and matplotlib equity curves
- **Multiple Strategies**: Built-in implementations of popular trading strategies
- **Real-time Visualization**: Interactive charts with trade markers and performance metrics
- **Comprehensive Testing**: Edge case handling and robust error management
- **Extensible Design**: Easy to add new strategies and data sources
- **Live Trading:** Connect to broker APIs, execute trades, and monitor live performance.
- **Configurable:** Centralized YAML config for assets, timeframes, and API keys.
- **Extensible:** Add new indicators, models, or strategies easily.

## Project Structure
```
├── data_pipeline/          # Data ingestion and processing
│   ├── market_data.py      # Fetching historical data
│   ├── live_feed.py        # Real-time data feed handling
│   └── preprocess.py       # Cleaning, normalization, feature engineering
│   └── market_data/        # Saved OHLCV CSVs
├── indicators/             # Technical and proprietary indicators
│   ├── messiah.py
│   ├── oracle.py
│   └── ...
├── models/                 # ML and RL models
│   ├── rl_agent.py
│   ├── rl_env.py
│   ├── supervised.py
│   └── unsupervised.py
├── strategies/             # Trading strategies
│   ├── base_strategy.py
│   ├── rl_strategy.py
│   ├── ml_strategy.py
│   └── ...
├── backtesting/            # Backtesting engine and evaluation
│   ├── backtest_engine.py
│   ├── backtrader_strategies/
│   └── evaluation.py
├── live_trading/           # Live trading components
│   ├── execution.py
│   ├── risk_management.py
│   └── monitor.py
├── tests/                  # Unit tests
├── main.py                 # Main entry point
└── config.yaml             # System configuration
```

## Quick Start
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Configure your system:**
   - Edit `config.yaml` with your symbols, API keys, and settings.
3. **Prepare data:**
   - Place your CSV files under the `market_data/` directory (e.g. `BTCUSDT_1h.csv`).
4. **Run a backtest:**
   ```sh
   python main.py
   ```

## Requirements
- Python 3.9+
- See `requirements.txt` for all dependencies

## Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes.

## License
MIT License
