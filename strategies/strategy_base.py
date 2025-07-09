class StrategyBase:
    """Base class for strategies used by the backtesting engine."""
    def on_bar(self, bar):
        raise NotImplementedError("Strategy must implement on_bar")
