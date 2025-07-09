from data_pipeline.data_fetcher import load_csv
from strategies.moving_average_cross import MovingAverageCross
from backtesting.portfolio import Portfolio

# Load small dataset
data = load_csv('BTCUSDT_1m_2025-06-08_to_2025-07-08', '')
test_data = data.iloc[:100]  # Just first 100 bars

# Create strategy and portfolio
strategy = MovingAverageCross(short_win=5, long_win=20)
portfolio = Portfolio(10000)

print("Testing strategy and portfolio interaction:")
print(f"Initial portfolio: cash=${portfolio.cash:.2f}, position={portfolio.position}")

trade_count = 0
for i, bar in enumerate(test_data.itertuples()):
    action = strategy.on_bar(bar)
    price = float(bar.close)
    
    print(f"Bar {i}: action={action}, price=${price:.2f}, portfolio_position={portfolio.position}")
    
    if action == "BUY" and portfolio.position == 0:
        cost_per_share = price * (1 + 0.001)  # Include commission
        qty = int(portfolio.cash / cost_per_share)
        if qty > 0:
            portfolio.buy(qty, price, fee=0.001)
            trade_count += 1
            print(f"  --> EXECUTED BUY: {qty} shares at ${price:.2f}")
            print(f"      Cash: ${portfolio.cash:.2f}, Position: {portfolio.position}")
    
    elif action == "SELL" and portfolio.position > 0:
        qty = portfolio.position
        if qty > 0:
            portfolio.sell(qty, price, fee=0.001)
            trade_count += 1
            print(f"  --> EXECUTED SELL: {qty} shares at ${price:.2f}")
            print(f"      Cash: ${portfolio.cash:.2f}, Position: {portfolio.position}")
    
    equity = portfolio.equity(price)
    
    if i >= 50:  # Stop after 50 bars to avoid too much output
        break

print(f"\nTotal trades executed: {trade_count}")
print(f"Final portfolio: cash=${portfolio.cash:.2f}, position={portfolio.position}")
print(f"Final equity: ${portfolio.equity(price):.2f}")
