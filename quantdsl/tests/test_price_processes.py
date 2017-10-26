import datetime
import unittest

import scipy
from dateutil.relativedelta import relativedelta
from numpy.matlib import randn
from pandas._libs.tslib import Timestamp

from quantdsl.priceprocess.blackscholes import BlackScholesPriceProcess, calc_correlation, \
    calc_historical_volatility, generate_calibration_params, pick_last_price, quandl_month_codes
from quantdsl.priceprocess.common import get_historical_data, from_csvtext, to_csvtext


class TestSimulateBlackScholesPriceProcess(unittest.TestCase):
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
            ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1), scipy.array([10., 10.]).mean()),
            ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2), scipy.array([10., 10.]).mean()),
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
            ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1), scipy.array([10., 10.]).mean()),
            ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2), scipy.array([10., 10.]).mean()),
            ('#2', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1), scipy.array([20., 20.]).mean()),
            ('#2', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2), scipy.array([20., 20.]).mean()),
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


class TestCalibrateBlackScholesPriceProcess(unittest.TestCase):
    def test_csvtext(self):
        csvtext = """2017-09-12,932.1
2017-09-13,935.0
2017-09-14,925.1
2017-09-15,920.2
"""
        series = from_csvtext(csvtext)

        self.assertEqual(series.index[0], Timestamp('2017-09-12'))
        self.assertEqual(series.index[1], Timestamp('2017-09-13'))
        self.assertEqual(series.index[2], Timestamp('2017-09-14'))
        self.assertEqual(series.index[3], Timestamp('2017-09-15'))

        self.assertEqual(series[0], 932.1)
        self.assertEqual(series[1], 935.0)
        self.assertEqual(series[2], 925.1)
        self.assertEqual(series[3], 920.2)

        self.assertEqual(to_csvtext(series), csvtext)

    def test_calc_historical_volatility(self):
        quotes = self.get_quotes()
        vol_log_returns = calc_historical_volatility(quotes)
        self.assertAlmostEqual(vol_log_returns, 0.144965, places=6)

    def test_pick_last_price(self):
        quotes = self.get_quotes()
        last_price = pick_last_price(quotes)
        self.assertEqual(last_price, 968.45)

    def test_calc_correlation(self):
        quotes = self.get_quotes()

        rho = calc_correlation(quotes)
        self.assertEqual(rho.shape, (1, 1))
        self.assertEqual(list(rho.flat), [1])

        return

        rho = calc_correlation(quotes, quotes)
        self.assertEqual(rho.shape, (2, 2))
        self.assertEqual(list(rho.flat), [1, 1, 1, 1])

        rho = calc_correlation(quotes, quotes, quotes)
        self.assertEqual(rho.shape, (3, 3))
        self.assertEqual(list(rho.flat), [1, 1, 1, 1, 1, 1, 1, 1, 1])

        rho = calc_correlation(quotes, list(map(lambda x: -x, quotes)))
        self.assertEqual(rho.shape, (2, 2))
        self.assertEqual(list(rho.flat), [1, -1, -1, 1])

        scipy.random.seed(12345)
        a = list(randn(20000).flat)
        b = list(randn(20000).flat)
        c = list(randn(20000).flat)
        rho = calc_correlation(a, b)
        self.assertEqual(rho.shape, (2, 2))
        self.assertAlmostEqual(rho[0][0], 1, places=1)
        self.assertAlmostEqual(rho[0][1], 0, places=1)
        self.assertAlmostEqual(rho[1][0], 0, places=1)
        self.assertAlmostEqual(rho[1][1], 1, places=1)

        rho = calc_correlation(a, b, c)
        self.assertEqual(rho.shape, (3, 3))
        self.assertAlmostEqual(rho[0][0], 1, places=1)
        self.assertAlmostEqual(rho[0][1], 0, places=1)
        self.assertAlmostEqual(rho[0][2], 0, places=1)
        self.assertAlmostEqual(rho[1][0], 0, places=1)
        self.assertAlmostEqual(rho[1][1], 1, places=1)
        self.assertAlmostEqual(rho[1][2], 0, places=1)
        self.assertAlmostEqual(rho[2][0], 0, places=1)
        self.assertAlmostEqual(rho[2][1], 0, places=1)
        self.assertAlmostEqual(rho[2][2], 1, places=1)

    def get_quotes(self):
        data = """
2017-09-12,932.1
2017-09-13,935.0
2017-09-14,925.1
2017-09-15,920.2
2017-09-18,915.0
2017-09-19,921.8
2017-09-20,931.5
2017-09-21,932.4
2017-09-22,928.5
2017-09-25,920.9
2017-09-26,924.8
2017-09-27,944.4
2017-09-28,949.5
2017-09-29,959.1
2017-10-02,953.2
2017-10-03,957.7
2017-10-04,951.6
2017-10-05,969.9
2017-10-06,978.8
2017-10-09,977.0
2017-10-10,972.6
2017-10-11,989.2
2017-10-12,987.8
2017-10-13,989.6
2017-10-16,992.0
2017-10-17,992.1
2017-10-18,992.8
2017-10-19,984.4
2017-10-20,988.2
2017-10-23,968.45
"""
        return from_csvtext(data)


class TestGetQuotes(unittest.TestCase):
    def _test_get_google_data_goog(self):
        # NB start and end doesn't seem to be effective with the 'google' service.
        quotes = get_historical_data('google', 'GOOG', col='Close', limit=30)
        index = quotes.index
        self.assertIsInstance(index[0], Timestamp)
        self.assertEqual(len(quotes), 30, str(quotes))

    # def test_get_yahoo_data_goog(self):
    #     quotes = get_historical_data('yahoo', 'GOOG', col='Close', end=datetime.date(2017, 10, 26))
    #     index = quotes.index
    #     self.assertIsInstance(index[0], Timestamp)
    #     self.assertEqual(len(quotes), 23)
    #
    def _test_get_quandl_data_goog(self):
        quotes = get_historical_data('quandl', 'GOOG', col='Close', end=datetime.date(2017, 10, 26))
        index = quotes.index
        self.assertIsInstance(index[0], Timestamp)
        self.assertTrue(len(quotes), 23)

    def _test_get_quandl_data_wti(self):
        quotes = get_historical_data(
            service='quandl',
            sym='ICE/TX2009',
            start=datetime.datetime(2007, 1, 1),
            end=datetime.datetime(2007, 2, 1),
            col='Settle',
        )
        index = quotes.index
        self.assertIsInstance(index[0], Timestamp)
        self.assertEqual(len(quotes), 23)

    def _test_get_quandl_data_ttf(self):
        quotes = get_historical_data(
            service='quandl',
            sym='ICE/TFMF2014',
            start=datetime.datetime(2013, 1, 1),
            col='Settle'
        )
        index = quotes.index
        self.assertIsInstance(index[0], Timestamp)
        self.assertEqual(len(quotes), 60)

    def _test_get_historical_data(self):
        # product_code = 'ICE/TFM'
        product_code = 'ICE/BPB'
        start = datetime.date(2016, 1, 1)
        end = datetime.date(2017, 12, 1)
        date = start
        step = relativedelta(months=1)
        while date < end:
            year = date.year
            month = date.month
            date = date + step
            print("Date", date)
            # continue

            month_code = quandl_month_codes[month]
            symbol = '{}{}{}'.format(product_code, month_code, year)
            quotes = get_historical_data(service='quandl', sym=symbol, start=datetime.datetime(2010, 1, 1),
                                         end=datetime.datetime.now())
            quotes_settle = quotes['Settle']
            num_quotes = len(quotes_settle)
            last_price = pick_last_price(quotes_settle)
            vol = calc_historical_volatility(quotes_settle)
            print(symbol, year, month, num_quotes, last_price, vol)


class TestGenerateCalibrationParams(unittest.TestCase):
    def _test(self):
        expect = {
            'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
            'market': ['GAS'],
            'sigma': [0.1],
            'rho': [[1.0]],
            'curve': {
                'GAS': [
                    ('2011-1-1', 1),
                    ('2012-1-1', 13.5),
                    ('2012-1-2', 19.4),
                    ('2012-1-3', 10.5),
                    ('2012-1-4', 10.3),
                    ('2012-1-5', 10.1),
                    ('2012-1-6', 10.2),
                ],
                'POWER': [
                    ('2011-1-1', 11),
                    ('2012-1-1', 15.5),
                    ('2012-1-2', 14.0),
                    ('2012-1-3', 15.0),
                    ('2012-1-4', 11.0),
                    ('2012-1-5', 1.0),
                    ('2012-1-6', 15.0),
                ]
            }
        }
        actual = generate_calibration_params(
            start=datetime.datetime(2017, 1, 1),
            end=datetime.datetime(2017, 7, 1),
            markets={
                'GAS': {
                    'service': 'quandl',
                    'sym': 'ICE/T',
                    'days': 1000,
                    'col': 'Settle',
                },
            },
        )
        self.maxDiff = None
        self.assertEqual(expect['market'], actual['market'])
        for date, price in actual['curve']['GAS']:
            self.assertIsInstance(date, datetime.date)
            self.assertIsInstance(price, float)
        self.assertEqual(actual['rho'], [[1]])
