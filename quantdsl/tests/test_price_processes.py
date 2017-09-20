import datetime
import unittest

import scipy

from quantdsl.priceprocess.blackscholes import BlackScholesPriceProcess, calc_sigma


class TestBlackScholesPriceProcess(unittest.TestCase):

    def setUp(self):
        self.p = BlackScholesPriceProcess()

    def test_simulate_future_prices_no_requirements(self):
        prices = list(self.p.simulate_future_prices(
            observation_date=datetime.datetime(2011, 1, 1),
            requirements=[],
            path_count=2,
            calibration_params={
                'market': ['#1'],
                'sigma': [0.5],
                'curve': {
                    '#1': (
                        ('2011-1-1', 10),
                    )
                },
            },
        ))
        self.assertEqual(list(prices), [])

    def test_simulate_future_prices_one_market_zero_volatility(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[
                ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
                ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)),
            ],
            observation_date=datetime.datetime(2011, 1, 1),
            path_count=2,
            calibration_params={
                'market': ['#1'],
                'sigma': [0.0],
                'curve': {
                    '#1': (
                        ('2011-1-1', 10),
                    )
                },
            },
        ))
        prices = [(p[0], p[1], p[2], p[3].mean()) for p in prices]  # For scipy.
        self.assertEqual(prices, [
            ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1), scipy.array([ 10.,  10.]).mean()),
            ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2), scipy.array([ 10.,  10.]).mean()),
        ])

    def test_simulate_future_prices_one_market_high_volatility(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[
                ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
                ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)),
            ],
            observation_date=datetime.datetime(2011, 1, 1),
            path_count=1000,
            calibration_params={
                'market': ['#1'],
                'sigma': [0.5],
                'curve': {
                    '#1': (
                        ('2011-1-1', 10),
                    )
                },
            },
        ))
        prices = [p[3].mean() for p in prices[1:]]  # For scipy.
        expected_prices = [10]
        for price, expected_price in zip(prices, expected_prices):
            self.assertNotEqual(price, expected_price)
            self.assertAlmostEqual(price, expected_price, places=0)

    def test_simulate_future_prices_two_markets_zero_volatility(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[
                ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
                ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)),
                ('#2', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
                ('#2', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)),
            ],
            observation_date=datetime.datetime(2011, 1, 1),
            path_count=200000, calibration_params={
                'market': ['#1', '#2'],
                'sigma': [0.0, 0.0],
                'curve': {
                    '#1': (
                        ('2011-1-1', 10),
                    ),
                    '#2': (
                        ('2011-1-1', 20),
                    )
                },
                'rho': [[1, 0], [0, 1]]
            }
        ))
        prices = [(p[0], p[1], p[2], p[3].mean()) for p in prices]  # For scipy.
        self.assertEqual(prices, [
            ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1), scipy.array([ 10.,  10.]).mean()),
            ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2), scipy.array([ 10.,  10.]).mean()),
            ('#2', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1), scipy.array([ 20.,  20.]).mean()),
            ('#2', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2), scipy.array([ 20.,  20.]).mean()),
        ])

    def test_simulate_future_prices_two_markets_high_volatility_zero_correlation(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[
                ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
                ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)),
                ('#2', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
                ('#2', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)),
            ],
            observation_date=datetime.datetime(2011, 1, 1),
            path_count=1000, calibration_params={
                'market': ['#1', '#2'],
                'sigma': [0.5, 0.5],
                'curve': {
                    '#1': (
                        ('2011-1-1', 10),
                    ),
                    '#2': (
                        ('2011-1-1', 20),
                    )
                },
                'rho': [[1, 0], [0, 1]]
            },
        ))
        prices = [p[3].mean() for p in prices]  # For scipy.
        expected_prices = [10, 10, 20, 20]
        for price, expected_price in zip(prices, expected_prices):
            self.assertAlmostEqual(price, expected_price, places=0)

    def test_simulate_future_prices_two_markets_high_volatility_positive_correlation(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[
                ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
                ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)),
                ('#2', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
                ('#2', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)),
            ],
            observation_date=datetime.datetime(2011, 1, 1),
            path_count=1000, calibration_params={
                'market': ['#1', '#2'],
                'sigma': [0.5, 0.5],
                'curve': {
                    '#1': (
                        ('2011-1-1', 10),
                    ),
                    '#2': (
                        ('2011-1-1', 20),
                    )
                },
                'rho': [[1, 0.5], [0.5, 1]]
            },
        ))
        assert len(prices)
        prices = [p[3].mean() for p in prices]  # For scipy.
        expected_prices = [10, 10, 20, 20]
        for price, expected_price in zip(prices, expected_prices):
            self.assertAlmostEqual(price, expected_price, places=0)

    def test_simulate_future_prices_from_longer_curve(self):
        prices = list(self.p.simulate_future_prices(
            requirements=[
                ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
                ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)),
                ('#2', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
                ('#2', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)),
            ],
            observation_date=datetime.datetime(2011, 1, 1),
            path_count=1000, calibration_params={
                'market': ['#1', '#2'],
                'sigma': [0.5, 0.5],
                'curve': {
                    '#1': (
                        ('2011-1-1', 10),
                        ('2011-1-2', 15)
                    ),
                    '#2': (
                        ('2011-1-1', 20),
                        ('2011-1-2', 25)
                    )
                },
                'rho': [[1, 0.5], [0.5, 1]]
            },
        ))
        expected_prices = [10, 15, 20, 25]
        prices = [p[3].mean() for p in prices]  # For scipy.

        for price, expected_price in zip(prices, expected_prices):
            self.assertAlmostEqual(price, expected_price, places=0)


class TestCalcSigma(unittest.TestCase):
    def test(self):
        sigma = calc_sigma([
            (datetime.datetime(2011, 1, 1), 10),
            (datetime.datetime(2011, 2, 1), 11),
            (datetime.datetime(2011, 3, 1), 9),
            (datetime.datetime(2011, 4, 1), 10),
        ])
        self.assertAlmostEqual(sigma, 0.14, places=2)
