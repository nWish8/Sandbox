# Sandbox Trading Research Stack

A comprehensive Python-based trading system featuring modular strategies, data pipeline integration, and backtesting capabilities using Backtrader.

## 🚀 Features

- **Modular Strategy Architecture**: Clean separation of strategy logic from execution engine
- **Multi-Provider Data Pipeline**: Support for CSV, Yahoo Finance, and Binance data sources
- **Backtrader Integration**: Professional-grade backtesting engine with extensive analytics
- **Extensible Design**: Easy to add new strategies, data providers, and evaluation metrics
- **Real-time Logging**: Comprehensive trade and performance logging
- **Portfolio Analytics**: Detailed performance metrics including Sharpe ratio, drawdown analysis

## 📁 Project Structure

```csv
sandbox/
├── main.py                     # Main entry point and demo script
├── config.yaml                 # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data_pipeline/              # Data management and providers
│   ├── __init__.py
│   ├── manager.py              # Main data manager
│   ├── base_provider.py        # Base data provider interface
│   ├── csv_provider.py         # CSV data source
│   ├── yahoo_provider.py       # Yahoo Finance integration
│   ├── binance_provider.py     # Binance API integration
│   ├── market_data.py          # Market data structures
│   ├── preprocess.py           # Data preprocessing utilities
│   ├── data_fetcher.py         # Data fetching utilities
│   └── live_feed.py            # Live data feed support
│
├── backtesting/                # Backtesting engine and utilities
│   ├── engine.py               # Core backtesting engine
│   ├── backtest_engine.py      # Backtrader integration
│   ├── portfolio.py            # Portfolio management
│   └── evaluation.py          # Performance evaluation metrics
│
├── strategies/                 # Trading strategy implementations
│   ├── strategy_base.py        # Base strategy class
│   ├── rsi_strategy.py         # RSI strategy (legacy, StrategyBase)
│   ├── rsi_backtrader.py       # RSI strategy (Backtrader compatible)
│   ├── ma_crossover_backtrader.py  # Moving Average crossover (Backtrader)
│   ├── bollinger_bands_strategy.py # Bollinger Bands strategy
│   ├── macd_strategy.py        # MACD strategy
│   └── moving_average_cross.py # Moving Average strategy (legacy)
│
└── market_data/                # Sample market data
    ├── BTCUSDT_1h.csv          # Bitcoin hourly data
    └── BTCUSDT_1m_2025-06-08_to_2025-07-08.csv  # Bitcoin minute data
```

## 🛠️ Installation

1. **Clone the repository** (if applicable) or ensure you have all files in your working directory.

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data availability**:
   Ensure sample data files are present in the `market_data/` directory.

## 🎯 Quick Start

Run the main demo script to see the RSI strategy in action:

```bash
python main.py
```

This will:

- Initialize the data pipeline
- Load Bitcoin 1-minute data
- Run the RSI strategy backtest
- Display comprehensive performance metrics

## 📊 Available Strategies

### 1. RSI Strategy (`rsi_backtrader.py`)

**Algorithm**: Relative Strength Index momentum strategy

**Logic**:

- **Buy Signal**: RSI falls below oversold threshold (default: 30)
- **Sell Signal**: RSI rises above overbought threshold (default: 70)
- **Position Sizing**: Configurable percentage of available cash (default: 95%)

**Parameters**:

- `rsi_period` (int): RSI calculation period (default: 14)
- `oversold` (float): Oversold threshold (default: 30)
- `overbought` (float): Overbought threshold (default: 70)
- `position_size` (float): Position size as fraction of cash (default: 0.95)
- `printlog` (bool): Enable trade logging (default: True)

**Use Cases**:

- Momentum trading in trending markets
- Counter-trend strategies in ranging markets
- Short-term swing trading

### 2. Moving Average Crossover (`ma_crossover_backtrader.py`)

**Algorithm**: Dual moving average crossover system

**Logic**:

- **Buy Signal**: Fast MA crosses above Slow MA
- **Sell Signal**: Fast MA crosses below Slow MA
- **Position Management**: Single position, full exit on opposite signal

**Parameters**:

- `fast_period` (int): Fast moving average period (default: 3)
- `slow_period` (int): Slow moving average period (default: 7)
- `position_size` (float): Position size as fraction of cash (default: 0.95)
- `printlog` (bool): Enable trade logging (default: True)

**Use Cases**:

- Trend-following strategies
- Long-term investment approaches
- Breakout trading

### 3. Legacy Strategies

The `strategies/` folder also contains legacy strategy implementations:

- `bollinger_bands_strategy.py`: Bollinger Bands mean reversion
- `macd_strategy.py`: MACD momentum strategy
- `moving_average_cross.py`: Simple MA crossover
- `rsi_strategy.py`: Original RSI implementation

These use the `StrategyBase` class and are not Backtrader-compatible. They can be converted following the pattern established in the Backtrader versions.

## 🔧 Data Pipeline

### Supported Data Sources

1. **CSV Provider** (`csv_provider.py`)
   - Local CSV file reading
   - Standard OHLCV format support
   - Custom date parsing

2. **Yahoo Finance** (`yahoo_provider.py`)
   - Historical data fetching
   - Multiple timeframes
   - Automatic symbol validation

3. **Binance API** (`binance_provider.py`)
   - Cryptocurrency data
   - Real-time and historical data
   - Multiple trading pairs

### Data Management

The `DataManager` class (`manager.py`) provides:

- Unified interface across providers
- Automatic provider selection
- Data caching and preprocessing
- Symbol discovery and validation

## 🧪 Backtesting Engine

### Core Features

- **Backtrader Integration**: Professional backtesting framework
- **Portfolio Tracking**: Real-time portfolio value monitoring
- **Commission Support**: Configurable transaction costs
- **Slippage Modeling**: Realistic execution simulation
- **Performance Analytics**: Comprehensive metrics calculation

### Key Metrics

- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to losses

### Usage Example

```python
from backtesting.engine import SandboxEngine
from strategies.rsi_backtrader import RSIBacktraderStrategy
from data_pipeline import DataManager

# Initialize components
engine = SandboxEngine(cash=100000.0, commission=0.001)
manager = DataManager(config)
data = manager.fetch_ohlcv('BTCUSDT_1m_2025-06-08_to_2025-07-08', '1m')

# Run backtest
results, portfolio = engine.run_backtest(
    RSIBacktraderStrategy,
    data=data,
    rsi_period=14,
    oversold=30,
    overbought=70
)
```

## 🎛️ Configuration

### Main Configuration (`config.yaml`)

```yaml
data_sources:
  csv:
    data_dir: "market_data"
  yahoo:
    cache_dir: "cache"
  binance:
    api_key: "your_api_key"
    api_secret: "your_api_secret"

default_provider: "csv"

backtesting:
  default_cash: 100000.0
  default_commission: 0.001
```

### Strategy Parameters

All strategies support parameter customization through the Backtrader parameter system:

```python
# Example: Custom RSI parameters
results, portfolio = engine.run_backtest(
    RSIBacktraderStrategy,
    data=data,
    rsi_period=21,        # Longer RSI period
    oversold=25,          # More sensitive oversold
    overbought=75,        # More sensitive overbought
    position_size=0.8,    # 80% position sizing
    printlog=False        # Disable trade logging
)
```

## 🚀 Creating New Strategies

### Backtrader Strategy Template

```python
import backtrader as bt
from typing import Any

class MyStrategy(bt.Strategy):
    """
    Custom strategy description.
    """
    
    params = (
        ('param1', default_value),
        ('param2', default_value),
        ('printlog', True),
    )
    
    def __init__(self):
        """Initialize indicators and variables."""
        # Create indicators
        self.indicator = bt.indicators.SMA(self.data.close, period=20)
        
        # Track orders and trades
        self.order: Any = None
        self.trade_count = 0
        
        # Log initialization
        if self.params.printlog:  # type: ignore
            print(f"Strategy initialized with param1={self.params.param1}")  # type: ignore
    
    def next(self):
        """Execute strategy logic on each bar."""
        # Check for pending orders
        if self.order:
            return
        
        # Strategy logic here
        if not self.position:
            # Entry condition
            if self.data.close[0] > self.indicator[0]:
                self.order = self.buy()
        else:
            # Exit condition
            if self.data.close[0] < self.indicator[0]:
                self.order = self.sell()
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Completed]:
            if order.isbuy() and self.params.printlog:  # type: ignore
                print(f"BUY EXECUTED: ${order.executed.price:.2f}")
            elif self.params.printlog:  # type: ignore
                print(f"SELL EXECUTED: ${order.executed.price:.2f}")
            self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if trade.isclosed and self.params.printlog:  # type: ignore
            self.trade_count += 1
            pnl = trade.pnl
            print(f"TRADE #{self.trade_count} CLOSED: PnL: ${pnl:.2f}")
```

### Integration Steps

1. **Create Strategy File**: Add to `strategies/` directory
2. **Import in Main**: Add import to `main.py`
3. **Configure Parameters**: Define strategy parameters
4. **Test Strategy**: Run backtests with different parameter sets

## 📈 Performance Monitoring

### Real-time Logging

All strategies support comprehensive logging:

```csv
RSI Strategy initialized:
  - RSI Period: 14
  - Oversold Level: 30
  - Overbought Level: 70
  - Position Size: 95.0%

BUY signal at $95432.10 (RSI: 28.5) on 2025-06-15
BUY EXECUTED: Price: $95432.10, Size: 1.0478
SELL signal at $97850.45 (RSI: 72.1) on 2025-06-18
SELL EXECUTED: Price: $97850.45, Size: 1.0478
TRADE #1 CLOSED: PnL: $2418.35 (2.53%)
```

### Performance Metrics

The system automatically calculates:

- **Return Metrics**: Total return, annualized return
- **Risk Metrics**: Volatility, Sharpe ratio, Sortino ratio
- **Drawdown Metrics**: Maximum drawdown, average drawdown
- **Trade Metrics**: Total trades, win rate, profit factor
- **Timing Metrics**: Average holding period, trade frequency

## 🔄 Data Requirements

### CSV Format

CSV files should follow this format:

```csv
timestamp,open,high,low,close,volume
2025-06-08 00:00:00,95432.10,95987.45,95201.30,95678.20,1234.56
2025-06-08 00:01:00,95678.20,95890.15,95543.80,95789.45,987.32
...
```

### Timeframe Support

- **Minute Data**: 1m, 5m, 15m, 30m
- **Hourly Data**: 1h, 2h, 4h, 6h, 12h
- **Daily Data**: 1d, 3d, 1w
- **Custom Intervals**: Configurable through providers

## 🛡️ Risk Management

### Built-in Features

- **Position Sizing**: Configurable position limits
- **Commission Modeling**: Realistic transaction costs
- **Slippage Simulation**: Market impact modeling
- **Drawdown Monitoring**: Automatic performance tracking

### Best Practices

1. **Start Small**: Begin with small position sizes
2. **Diversify**: Test multiple strategies and timeframes
3. **Monitor Drawdowns**: Set maximum acceptable losses
4. **Parameter Testing**: Optimize strategy parameters carefully
5. **Out-of-Sample Testing**: Reserve data for validation

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Loading**: Verify CSV format and file paths
3. **Strategy Errors**: Check parameter types and ranges
4. **Performance Issues**: Monitor memory usage with large datasets

### Debug Mode

Enable detailed logging by setting `printlog=True` in strategy parameters:

```python
results, portfolio = engine.run_backtest(
    RSIBacktraderStrategy,
    data=data,
    printlog=True  # Enable detailed logging
)
```

## 🤝 Contributing

### Adding New Strategies

1. Follow the Backtrader strategy template
2. Include comprehensive documentation
3. Add parameter validation
4. Include example usage
5. Test with multiple datasets

### Extending Data Pipeline

1. Inherit from `BaseProvider`
2. Implement required methods
3. Add configuration support
4. Include error handling
5. Document API requirements

## 📚 Dependencies

### Core Requirements

- `backtrader`: Backtesting engine
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `yfinance`: Yahoo Finance data
- `python-binance`: Binance API client
- `pyyaml`: Configuration file parsing

### Optional Dependencies

- `matplotlib`: Plotting and visualization
- `jupyter`: Interactive development
- `scikit-learn`: Machine learning features

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with data provider terms of service and applicable financial regulations when using this system for live trading.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss and is not suitable for all investors. Always conduct thorough testing and risk assessment before deploying any trading strategy with real funds.
