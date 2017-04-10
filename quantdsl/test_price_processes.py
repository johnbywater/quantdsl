import datetime
import unittest

import scipy

from quantdsl.priceprocess.blackscholes import BlackScholesPriceProcess


class TestBlackScholesPriceProcess(unittest.TestCase):

    def setUp(self):
        self.p = BlackScholesPriceProcess()

    def test_simulate_future_prices_no_requirements(self):
        prices = list(self.p.simulate_future_prices(
            observation_date=datetime.date(2011, 1, 1),
            requirements=[],
            path_count=2,
            calibration_params={'#1-LAST-PRICE': 10, '#1-ACTUAL-HISTORICAL-VOLATILITY': 50},
        ))
        prices = [(p[0], p[1], p[2], p[3].all()) for p in prices]  # For scipy.
        self.assertEqual(prices, [])

    def test_simulate_future_prices_one_market_zero_volatility(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[
                ('#1', datetime.date(2011, 1, 1), datetime.date(2011, 1, 1)),
                ('#1', datetime.date(2011, 1, 2), datetime.date(2011, 1, 2)),
            ],
            observation_date=datetime.date(2011, 1, 1),
            path_count=2,
            calibration_params={
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 0,
            },
        ))
        prices = [(p[0], p[1], p[2], p[3].mean()) for p in prices]  # For scipy.
        self.assertEqual(prices, [
            ('#1', datetime.date(2011, 1, 1), datetime.date(2011, 1, 1), scipy.array([ 10.,  10.]).mean()),
            ('#1', datetime.date(2011, 1, 2), datetime.date(2011, 1, 2), scipy.array([ 10.,  10.]).mean()),
        ])

    def test_simulate_future_prices_one_market_high_volatility(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[],
            observation_date=datetime.date(2011, 1, 1),
            path_count=1000,
            calibration_params={
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 50,
            },
        ))
        prices = [p[2].mean() for p in prices[1:]]  # For scipy.
        expected_prices = [10]
        for price, expected_price in zip(prices, expected_prices):
            self.assertNotEqual(price, expected_price)
            self.assertAlmostEqual(price, expected_price, places=0)

    def test_simulate_future_prices_two_markets_zero_volatility(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[
                ('#1', datetime.date(2011, 1, 1), datetime.date(2011, 1, 1), scipy.array([ 10.,  10.]).mean()),
                ('#1', datetime.date(2011, 1, 2), datetime.date(2011, 1, 2), scipy.array([ 10.,  10.]).mean()),
                ('#2', datetime.date(2011, 1, 1), datetime.date(2011, 1, 1), scipy.array([ 20.,  20.]).mean()),
                ('#2', datetime.date(2011, 1, 2), datetime.date(2011, 1, 2), scipy.array([ 20.,  20.]).mean()),
            ],
            observation_date=datetime.date(2011, 1, 1),
            path_count=200000, calibration_params={
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 0,
                '#2-LAST-PRICE': 20,
                '#2-ACTUAL-HISTORICAL-VOLATILITY': 0,
                '#1-#2-CORRELATION': 0,
            },
        ))
        prices = [(p[0], p[1], p[2], p[3].mean()) for p in prices]  # For scipy.
        self.assertEqual(prices, [
            ('#1', datetime.date(2011, 1, 1), datetime.date(2011, 1, 1), scipy.array([ 10.,  10.]).mean()),
            ('#1', datetime.date(2011, 1, 2), datetime.date(2011, 1, 2), scipy.array([ 10.,  10.]).mean()),
            ('#2', datetime.date(2011, 1, 1), datetime.date(2011, 1, 1), scipy.array([ 20.,  20.]).mean()),
            ('#2', datetime.date(2011, 1, 2), datetime.date(2011, 1, 2), scipy.array([ 20.,  20.]).mean()),
        ])

    def test_simulate_future_prices_two_markets_high_volatility_zero_correlation(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[],
            observation_date=datetime.date(2011, 1, 1),
            path_count=1000, calibration_params={
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 50,
                '#2-LAST-PRICE': 20,
                '#2-ACTUAL-HISTORICAL-VOLATILITY': 50,
                '#1-#2-CORRELATION': 0,
            },
        ))
        prices = [p[2].mean() for p in prices]  # For scipy.
        expected_prices = [10, 10, 20, 20]
        for price, expected_price in zip(prices, expected_prices):
            self.assertAlmostEqual(price, expected_price, places=0)

    def test_simulate_future_prices_two_markets_high_volatility_positive_correlation(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[],
            observation_date=datetime.date(2011, 1, 1),
            path_count=1000, calibration_params={
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 50,
                '#2-LAST-PRICE': 20,
                '#2-ACTUAL-HISTORICAL-VOLATILITY': 50,
                '#1-#2-CORRELATION': 0.5,
            },
        ))
        prices = [p[2].mean() for p in prices]  # For scipy.
        expected_prices = [10, 10, 20, 20]
        for price, expected_price in zip(prices, expected_prices):
            self.assertAlmostEqual(price, expected_price, places=0)
