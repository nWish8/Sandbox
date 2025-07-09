import unittest
from backtesting.portfolio import Portfolio


class TestPortfolio(unittest.TestCase):
    def test_buy_sell_equity(self):
        p = Portfolio(100)
        p.buy(1, 10)
        self.assertAlmostEqual(p.cash, 90)
        self.assertEqual(p.position, 1)
        p.sell(1, 12)
        self.assertAlmostEqual(p.cash, 102)
        self.assertEqual(p.position, 0)
        self.assertAlmostEqual(p.equity(12), 102)


if __name__ == "__main__":
    unittest.main()
