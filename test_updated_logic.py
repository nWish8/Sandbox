from data_pipeline.data_fetcher import load_csv
from strategies.moving_average_cross import MovingAverageCross
from backtesting.backtest_engine import BacktestEngine

# Test with shorter MA periods to get more trades
data = load_csv('BTCUSDT_1m_2025-06-08_to_2025-07-08', '')
print(f'Data shape: {data.shape}')

# Test with much shorter MA periods
strategy = MovingAverageCross(short_win=10, long_win=30)
engine = BacktestEngine(data, strategy, initial_capital=10000, commission=0.001)

equity, trades = engine.run()
print(f'Equity curve length: {len(equity)}')
print(f'Number of trades: {len(trades)}')
print(f'Starting equity: ${equity[0][1]:.2f}')
print(f'Ending equity: ${equity[-1][1]:.2f}')
print(f'Return: {((equity[-1][1] / equity[0][1]) - 1) * 100:.2f}%')

if trades:
    print(f'First few trades: {trades[:3]}')
    print(f'Last few trades: {trades[-3:]}')
