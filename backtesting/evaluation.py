import numpy as np


def compute_metrics(equity_curve, trade_log):
    """Compute basic performance metrics."""
    equity = np.array([e[1] for e in equity_curve])
    returns = np.diff(equity) / equity[:-1]
    metrics = {
        "Total Return": equity[-1] / equity[0] - 1 if len(equity) > 1 else 0.0,
        "Sharpe Ratio": (np.mean(returns) / np.std(returns) * np.sqrt(252)) if returns.size > 1 and np.std(returns) != 0 else 0.0,
        "Max Drawdown": np.max(np.maximum.accumulate(equity) - equity) if equity.size > 0 else 0.0,
        "Win Rate": _win_rate(trade_log),
    }
    return metrics


def _win_rate(trade_log):
    sells = [t for t in trade_log if t["action"] == "SELL"]
    wins = 0
    total = 0
    for trade in sells:
        # naive win if sell price higher than previous buy
        idx = trade_log.index(trade)
        for prev in reversed(trade_log[:idx]):
            if prev["action"] == "BUY":
                total += 1
                if trade["price"] > prev["price"]:
                    wins += 1
                break
    return wins / total if total else 0.0
