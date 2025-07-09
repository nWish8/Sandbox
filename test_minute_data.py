from data_pipeline.data_fetcher import load_csv
from strategies.moving_average_cross import MovingAverageCross
from backtesting.backtest_engine import BacktestEngine

# Load data
data = load_csv('BTCUSDT_1m_2025-06-08_to_2025-07-08', '')
print(f'Data shape: {data.shape}')

# Create strategy and engine
strategy = MovingAverageCross()
engine = BacktestEngine(data, strategy)

# Run backtest
equity, trades = engine.run()
print(f'Equity curve length: {len(equity)}')
print(f'Number of trades: {len(trades)}')
print(f'First few trades: {trades[:5] if trades else "No trades"}')
