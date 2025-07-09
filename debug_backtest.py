from data_pipeline.data_fetcher import load_csv
from strategies.moving_average_cross import MovingAverageCross
from backtesting.backtest_engine import BacktestEngine

# Test with shorter MA periods
data = load_csv('BTCUSDT_1m_2025-06-08_to_2025-07-08', '')
print(f'Data shape: {data.shape}')

# Create backtest engine with debug
strategy = MovingAverageCross(short_win=5, long_win=20)
engine = BacktestEngine(data, strategy, initial_capital=10000, commission=0.001)

# Test first 1000 bars
test_data = data.iloc[:1000]
engine.data = test_data

equity, trades = engine.run()
print(f'Equity curve length: {len(equity)}')
print(f'Number of trades: {len(trades)}')
print(f'Starting equity: ${equity[0][1]:.2f}')
print(f'Ending equity: ${equity[-1][1]:.2f}')
print(f'Return: {((equity[-1][1] / equity[0][1]) - 1) * 100:.2f}%')

if trades:
    print(f'\nFirst 5 trades:')
    for i, trade in enumerate(trades[:5]):
        print(f"Trade {i+1}: {trade['action']} at ${trade['price']:.2f} on {trade['time']}")
    
    print(f'\nLast 5 trades:')
    for i, trade in enumerate(trades[-5:]):
        print(f"Trade {len(trades)-4+i}: {trade['action']} at ${trade['price']:.2f} on {trade['time']}")
        
    # Show portfolio progression
    print(f'\nPortfolio progression:')
    print(f'Starting cash: ${engine.portfolio.cash:.2f}')
    print(f'Final cash: ${engine.portfolio.cash:.2f}')
    print(f'Final position: {engine.portfolio.position}')
    print(f'Final equity: ${engine.portfolio.equity(equity[-1][1]):.2f}')
else:
    print('No trades generated!')
