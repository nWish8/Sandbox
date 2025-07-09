"""Simple test script to verify backtesting functionality."""

from backtesting.backtest_engine import BacktestEngine
from strategies.moving_average_cross import MovingAverageCross
from data_pipeline.data_fetcher import load_csv
from backtesting.evaluation import compute_metrics


def test_backtest():
    print("Loading data...")
    data = load_csv("BTCUSDT", "1h")
    print(f"Data loaded: {len(data)} rows")
    print(f"Data columns: {list(data.columns)}")
    print(f"Data sample:\n{data.head()}")
    
    print("\nInitializing strategy...")
    strategy = MovingAverageCross()
    
    print("\nRunning backtest...")
    engine = BacktestEngine(data, strategy)
    equity_curve, trades = engine.run()
    
    print(f"\nBacktest completed!")
    print(f"Number of trades: {len(trades)}")
    print(f"Equity curve length: {len(equity_curve)}")
    
    if trades:
        print("\nSample trades:")
        for i, trade in enumerate(trades[:5]):
            print(f"Trade {i+1}: {trade}")
    
    print("\nComputing metrics...")
    metrics = compute_metrics(equity_curve, trades)
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    test_backtest()
