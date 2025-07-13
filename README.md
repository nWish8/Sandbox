# BT Sandbox - Backtesting Framework

A comprehensive Python package for developing, testing, and analyzing trading strategies using Backtrader as the core backtesting engine.

## 🚀 Features

- **Modular Package Architecture**: Clean, organized Python package structure
- **Multi-Provider Data Pipeline**: Support for CSV, Yahoo Finance, and Binance data sources
- **Backtrader Integration**: Professional-grade backtesting engine with extensive analytics
- **Extensible Design**: Easy to add new strategies, data providers, and evaluation metrics
- **Real-time Logging**: Comprehensive trade and performance logging
- **Portfolio Analytics**: Detailed performance metrics including Sharpe ratio, drawdown analysis

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Sandbox

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Your First Backtest

```bash
# Run the default RSI strategy demo
python -m bt_sandbox
```

### 3. Expected Output

When you run the command above, you should see output similar to this:

```text
============================================================
BT Sandbox - RSI Strategy Demo (4H Timeframe)
============================================================
1. Initializing data manager...
Available symbols: {'csv': ['BTCUSDT']}
2. Loading 4-hour data for BTCUSDT_4h_recent...
   ✓ Loaded 500 bars of 4-hour data
   📅 Date range: 2025-04-21 12:00:00 to 2025-07-13 16:00:00
   💰 Price range: $87415.10 - $118949.10
3. Setting up Backtrader engine...
4. Running RSI strategy backtest on 4-hour timeframe...
----------------------------------------
RSI Reversal Cross Strategy initialized:
  - RSI Period: 14
  - Oversold Level: 30
  - Overbought Level: 70
  - Position Size: 95.0%
  - Trading on reversal crosses only
BUY signal at $102493.30 (RSI cross above 30: 28.8 -> 37.0) on 2025-06-06 00:00
BUY EXECUTED: Price: $102493.30, Size: 0.0093
SELL signal at $109205.14 (RSI cross below 70: 70.8 -> 67.6) on 2025-06-10 04:00
SELL EXECUTED: Price: $109205.14, Size: -0.0093
TRADE #1 CLOSED: PnL: $62.21 (0.00%)
----------------------------------------
5. Backtest Results:
   💵 Starting Value: $1,000.00
   💰 Final Value:    $1,081.69
   📈 Total Return:   8.17%
   📊 Sharpe Ratio:   N/A
   📉 Max Drawdown:   856.43%
   🔄 Total Trades:   2
   ✅ Winning Trades: 2
   ❌ Losing Trades:  0
6. Displaying backtest results chart...
   📈 Opening Backtrader plot window...
   ✓ Plot window opened successfully!
============================================================
✅ Backtest completed successfully!
============================================================
```

The system will also open an interactive chart showing:

- 📊 Candlestick price data
- 📈 RSI indicator with overbought/oversold levels
- 🔵 Buy signals (blue arrows)
- 🔴 Sell signals (red arrows)
- 💰 Portfolio value progression

### 4. Next Steps

- **Try different strategies**: `python -m bt_sandbox.strategies.ma_crossover_strategy`
- **Add your own data**: Place CSV files in `market_data/` directory
- **Create custom strategies**: Extend the `bt_sandbox.strategies` module
- **Fetch fresh data**: Use `python fetch_btc_data.py` to get latest market data

### 5. Enhanced Features (Recommended)

For more advanced usage with Backtrader best practices:

```bash
# Run enhanced demo with Yahoo Finance data
python -m bt_sandbox.enhanced_main

# Use improved strategies in your code
from bt_sandbox import EnhancedEngine, ImprovedRSIStrategy
```

**Enhanced Features:**

- 🎯 **Native Yahoo Finance Integration**: Direct Backtrader data feeds (no manual downloads)
- ⚙️ **Realistic Broker Settings**: Proper commission, slippage, and position sizing
- 📊 **Efficient Strategies**: Indicator initialization in `__init__` for better performance
- 🔧 **Signal-Based Strategies**: Concise strategy implementations with `bt.SignalStrategy`
- 📈 **Professional Analytics**: Comprehensive performance metrics and plotting

## 📁 Project Structure

```text
sandbox/
├── bt_sandbox/                 # Main package
│   ├── __init__.py            # Package initialization
│   ├── __main__.py            # Module entry point (python -m bt_sandbox)
│   ├── main.py                # Main application logic
│   ├── README.md              # Package documentation
│   │
│   ├── strategies/            # Trading strategy implementations
│   │   ├── __init__.py
│   │   ├── rsi_strategy.py    # RSI reversal cross strategy
│   │   ├── ma_crossover_strategy.py  # Moving average crossover
│   │   └── bollinger_bands_strategy.py  # Bollinger Bands strategy
│   │
│   ├── datafeeds/             # Data acquisition and management
│   │   ├── __init__.py
│   │   ├── manager.py         # Unified data manager
│   │   └── providers/         # Data provider implementations
│   │       ├── __init__.py
│   │       ├── base_provider.py
│   │       ├── csv_provider.py
│   │       ├── binance_provider.py
│   │       └── yahoo_provider.py
│   │
│   ├── backtesting/           # Backtesting engine and utilities
│   │   ├── __init__.py
│   │   └── engine.py          # Enhanced Backtrader engine
│   │
│   └── utils/                 # Helper functions and utilities
│       ├── __init__.py
│       └── data_fetcher.py    # Data fetching utilities
│
├── market_data/               # Sample market data
├── fetch_btc_data.py         # Data fetching script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
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

Run the main package to see the RSI strategy in action:

```bash
# Run the main package
python -m bt_sandbox

# Or alternatively:
python -m bt_sandbox.main
```

This will:

- Initialize the data pipeline
- Load Bitcoin 4-hour data
- Run the RSI strategy backtest
- Display comprehensive performance metrics
- Show interactive Backtrader plot

## 📊 Available Strategies

### 1. RSI Strategy (`bt_sandbox/strategies/rsi_strategy.py`)

**Algorithm**: Relative Strength Index reversal cross strategy

**Logic**:

- **Buy Signal**: RSI crosses above oversold threshold after being below (default: 30)
- **Sell Signal**: RSI crosses below overbought threshold after being above (default: 70)
- **Position Sizing**: Configurable percentage of available cash (default: 95%)

**Parameters**:

- `rsi_period` (int): RSI calculation period (default: 14)
- `oversold` (float): Oversold threshold (default: 30)
- `overbought` (float): Overbought threshold (default: 70)
- `position_size` (float): Position size as fraction of cash (default: 0.95)
- `printlog` (bool): Enable trade logging (default: True)

**Use Cases**:

- Counter-trend strategies in ranging markets
- Mean reversion trading
- Short-term swing trading

### 2. Moving Average Crossover (`bt_sandbox/strategies/ma_crossover_strategy.py`)

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
from bt_sandbox.backtesting import SandboxEngine
from bt_sandbox.strategies import RSIBacktraderStrategy
from bt_sandbox.datafeeds import DataManager

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

### Data Manager Configuration

Configure data sources programmatically:

```python
from bt_sandbox.datafeeds import DataManager

config = {
    'csv': {'data_dir': 'market_data'},
    'yahoo': {'cache_dir': 'cache'},
    'binance': {
        'api_key': 'your_api_key',
        'api_secret': 'your_api_secret'
    },
    'default_provider': 'csv'
}

manager = DataManager(config)
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

1. **Create Strategy File**: Add to `bt_sandbox/strategies/` directory
2. **Import in Package**: Add import to strategy `__init__.py`
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
