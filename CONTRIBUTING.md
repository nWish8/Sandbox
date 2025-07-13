# Contributing to Sandbox: ML-Trading Research Stack

Welcome to the Sandbox trading research platform! This document outlines the migration roadmap, development guidelines, and how to contribute to our modular ML-trading research stack.

## 🎯 Project Vision

Sandbox is transforming from a simple backtesting tool into a comprehensive, modular ML-trading research platform that supports:

- **Modern backtesting** with Backtrader integration
- **Multiple data sources** via pluggable providers
- **ML/RL strategy development** with standardized interfaces
- **Interactive dashboards** for parameter tuning and analysis
- **Production-ready trading** with risk management

## 📋 Migration Roadmap

### ✅ Phase 1: Foundation (COMPLETED)
- [x] **Backtrader Integration**: Replaced custom backtest loop with robust Backtrader engine
  - Created `SandboxEngine` class extending `bt.Cerebro`
  - Implemented fractional sizing support (`FractionalSizer`, `PercentSizer`)
  - Added comprehensive analyzer integration (Sharpe, Drawdown, Returns, etc.)
  - Validated with test cases and demo scripts

- [x] **Data Layer Refactoring**: Modular, provider-based architecture
  - Created `sandbox.data` package with unified interfaces
  - Implemented `BaseDataProvider` abstract class
  - Added `CSVProvider` for existing local data
  - Added `BinanceProvider` for crypto data via ccxt
  - Added `YahooProvider` for stock data via yfinance  
  - Created `DataManager` for provider orchestration and caching

### 🚧 Phase 2: Strategy Modernization (IN PROGRESS)
- [ ] **Strategy Interface Standardization**
  - [ ] Create `BaseStrategy` with common interface for Backtrader and ML/RL
  - [ ] Migrate existing strategies (`moving_average_cross.py`, `rsi_strategy.py`, etc.)
  - [ ] Add strategy parameter validation and documentation
  - [ ] Implement strategy composition and chaining

- [ ] **ML/RL Strategy Support**
  - [ ] Create `MLStrategy` base class for supervised learning models
  - [ ] Create `RLStrategy` base class for reinforcement learning agents
  - [ ] Implement feature engineering pipeline
  - [ ] Add model serialization and versioning

### 📊 Phase 3: Interactive Dashboard (PLANNED)
- [ ] **Streamlit Dashboard Implementation**
  - [ ] Create parameter tuning interface
  - [ ] Add real-time backtest execution
  - [ ] Implement equity curve plotting with Plotly
  - [ ] Add strategy comparison tools
  - [ ] Create data source management UI

- [ ] **Visualization Enhancements**
  - [ ] Portfolio analytics dashboard
  - [ ] Risk metrics visualization
  - [ ] Trade analysis plots
  - [ ] Performance attribution charts

### 🧪 Phase 4: Testing & Documentation (PLANNED)
- [ ] **Comprehensive Test Suite**
  - [ ] Unit tests for all providers and strategies
  - [ ] Integration tests for engine and data flows
  - [ ] Performance benchmarking suite
  - [ ] Mock data generators for testing

- [ ] **Documentation & Examples**
  - [ ] API documentation with Sphinx
  - [ ] Strategy development tutorials
  - [ ] Data provider integration guides
  - [ ] ML/RL workflow examples

### 🚀 Phase 5: Production Features (PLANNED)
- [ ] **Live Trading Integration**
  - [ ] Real-time data feeds
  - [ ] Order execution interfaces
  - [ ] Risk management framework
  - [ ] Position monitoring

- [ ] **Advanced Features**
  - [ ] Multi-asset portfolio optimization
  - [ ] Walk-forward analysis
  - [ ] Monte Carlo simulation
  - [ ] Advanced order types

## 🛠️ Development Setup

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Key Dependencies
- **backtrader**: Modern backtesting framework
- **pandas**: Data manipulation and analysis
- **ccxt**: Cryptocurrency exchange integration
- **yfinance**: Stock market data
- **streamlit**: Interactive dashboard framework
- **scikit-learn**: Machine learning tools
- **tensorflow/pytorch**: Deep learning (optional)

### Project Structure
```
sandbox/
├── data/                   # Data management layer
│   ├── providers/         # Data provider implementations
│   │   ├── base_provider.py      # Abstract base class
│   │   ├── csv_provider.py       # Local CSV files
│   │   ├── binance_provider.py   # Crypto data via Binance
│   │   └── yahoo_provider.py     # Stock data via Yahoo
│   └── manager.py         # Unified data manager
├── strategies/            # Trading strategies (to be refactored)
├── models/               # ML/RL models (to be enhanced)
└── backtesting/          # Backtesting engine
    └── engine.py         # SandboxEngine implementation

backtesting/              # Core backtesting (legacy, being migrated)
strategies/               # Legacy strategies (being migrated)
data_pipeline/           # Legacy data code (being migrated)
ui/                      # Legacy UI (being replaced with Streamlit)
```

## 📖 Usage Examples

### New Data Layer
```python
from sandbox.data import DataManager

# Initialize with multiple providers
config = {
    'csv': {'data_dir': 'market_data'},
    'binance': {'market_type': 'spot'},
    'yahoo': {'auto_adjust': True}
}
manager = DataManager(config)

# Fetch data (automatic provider selection)
btc_data = manager.fetch_ohlcv('BTCUSDT', '1h')
aapl_data = manager.fetch_ohlcv('AAPL', '1d')
```

### Modern Backtesting
```python
from backtesting.engine import SandboxEngine
import backtrader as bt

class MyStrategy(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy(size=0.1)  # Fractional sizing supported

# Run backtest
engine = SandboxEngine(cash=100000, commission=0.001)
results, portfolio = engine.run_backtest(MyStrategy, data=btc_data)
```

## 🤝 Contributing Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for all public methods
- Include comprehensive docstrings
- Add logging for important operations

### Testing Requirements
- Write tests for new features
- Ensure backward compatibility
- Test with multiple data sources
- Validate performance impacts

### Pull Request Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run test suite (`python -m pytest`)
5. Submit pull request with clear description

### Documentation Standards
- Update API documentation
- Include usage examples
- Document configuration options
- Add migration notes for breaking changes

## 🐛 Known Issues & Technical Debt

### Current Limitations
- Limited test data (only 10 rows in BTCUSDT_1h.csv)
- Legacy strategy interfaces need modernization
- UI components need Streamlit migration
- Limited ML/RL integration

### Migration Priorities
1. **Data Volume**: Add more comprehensive test data
2. **Strategy Modernization**: Migrate legacy strategies to new interfaces  
3. **Testing**: Expand test coverage significantly
4. **Documentation**: Complete API documentation

## 📊 Performance Benchmarks

### Current State
- **Data Loading**: ~50ms for 1000 OHLCV bars
- **Backtest Execution**: ~200ms for simple strategy over 1000 bars
- **Memory Usage**: ~50MB for typical backtest with 10K bars

### Target Performance
- **Data Loading**: <20ms for 1000 bars
- **Backtest Execution**: <100ms for simple strategy
- **Concurrent Backtests**: Support 10+ parallel executions

## 🔧 Migration Utilities

### Data Migration Scripts
```bash
# Convert legacy data to new format
python scripts/migrate_data.py --source data_pipeline --target sandbox/data

# Validate data provider configurations
python scripts/validate_providers.py --config config.yaml
```

### Strategy Migration Tools
```bash
# Migrate legacy strategy to new interface
python scripts/migrate_strategy.py --strategy strategies/old_strategy.py

# Validate strategy compatibility
python scripts/validate_strategy.py --strategy MyStrategy
```

## 📞 Support & Communication

### Getting Help
- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check the docs/ directory for detailed guides

### Roadmap Updates
This roadmap is actively maintained and updated based on:
- Community feedback and contributions
- Performance benchmarking results
- Integration complexity discoveries
- Production deployment requirements

---

## 🎉 Recent Achievements

### December 2024 - January 2025
- ✅ Successfully migrated from custom backtest loop to Backtrader
- ✅ Implemented modular data provider architecture
- ✅ Created unified data manager with caching and fallback
- ✅ Added fractional sizing support for crypto trading
- ✅ Established comprehensive analyzer integration
- ✅ Validated integration with existing data sources

### Next Milestones
- 🎯 **February 2025**: Strategy interface standardization
- 🎯 **March 2025**: Streamlit dashboard MVP
- 🎯 **April 2025**: ML/RL strategy framework
- 🎯 **May 2025**: Production readiness assessment

---

**Ready to contribute?** Start by picking an issue labeled `good-first-issue` or `help-wanted`!

**Questions?** Open a discussion or check our documentation in the `docs/` directory.

**Found a bug?** Please report it with reproduction steps and system information.

Thank you for helping make Sandbox the premier ML-trading research platform! 🚀
