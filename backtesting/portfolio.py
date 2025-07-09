class Portfolio:
    """Simple portfolio to track cash and positions."""
    def __init__(self, capital: float):
        self.cash = float(capital)
        self.position = 0

    def buy(self, qty: int, price: float, fee: float = 0.0) -> None:
        total_cost = qty * price * (1 + fee)
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.position += qty

    def sell(self, qty: int, price: float, fee: float = 0.0) -> None:
        if qty > self.position:
            qty = self.position
        total_proceeds = qty * price * (1 - fee)
        if qty > 0:
            self.cash += total_proceeds
            self.position -= qty

    def equity(self, price: float) -> float:
        return self.cash + self.position * price
