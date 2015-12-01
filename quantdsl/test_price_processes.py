import datetime
import unittest

import mock
import numpy

from quantdsl import utc
from quantdsl.domain.model.dependency_graph import DependencyGraph
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.exceptions import DslSyntaxError
from quantdsl.infrastructure.runners.multiprocess import MultiProcessingDependencyGraphRunner
from quantdsl.infrastructure.runners.singlethread import SingleThreadedDependencyGraphRunner
from quantdsl.priceprocess.blackscholes import BlackScholesPriceProcess
from quantdsl.semantics import Module, compile_dsl_module, Number, UnarySub, String, Name, Add, Sub, Mult, Div, \
    FloorDiv, Pow, Mod, Compare, IfExp, If, Max, Date, TimeDelta, On, FunctionDef, FunctionCall, Fixing
from quantdsl.services import dsl_compile, dsl_eval
from quantdsl.syntax import DslParser


class TestBlackScholesPriceProcess(unittest.TestCase):

    def setUp(self):
        self.p = BlackScholesPriceProcess()

    def test_simulate_future_prices_no_market_names_no_fixing_dates(self):
        prices = list(self.p.simulate_future_prices(
            market_names=[],
            fixing_dates=[],
            observation_date=datetime.date(2011, 1, 1),
            path_count=2, calibration_params={},
        ))
        self.assertEqual(prices, [])

    def test_simulate_future_prices_no_fixing_dates(self):
        prices = list(self.p.simulate_future_prices(
            market_names=['#1'],
            fixing_dates=[],
            observation_date=datetime.date(2011, 1, 1),
            path_count=2, calibration_params={'#1-LAST-PRICE': 10, '#1-ACTUAL-HISTORICAL-VOLATILITY': 50},
        ))
        prices = [(p[0], p[1], p[2].all()) for p in prices]  # For numpy.
        self.assertEqual(prices, [('#1', datetime.date(2011, 1, 1), numpy.array([ 10.,  10.]).all())])

    def test_simulate_future_prices_no_markets(self):
        prices = list(self.p.simulate_future_prices(
            market_names=[],
            fixing_dates=[datetime.date(2011, 1, 2)],
            observation_date=datetime.date(2011, 1, 1),
            path_count=2, calibration_params={},
        ))
        self.assertEqual(prices, [])

    def test_simulate_future_prices_one_market_zero_volatility(self):
        prices = list(self.p.simulate_future_prices(
            market_names=['#1'],
            fixing_dates=[datetime.date(2011, 1, 2)],
            observation_date=datetime.date(2011, 1, 1),
            path_count=2, calibration_params={
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 0,
            },
        ))
        prices = [(p[0], p[1], p[2].mean()) for p in prices]  # For numpy.
        self.assertEqual(prices, [
            ('#1', datetime.date(2011, 1, 1), numpy.array([ 10.,  10.]).mean()),
            ('#1', datetime.date(2011, 1, 2), numpy.array([ 10.,  10.]).mean()),
        ])

    def test_simulate_future_prices_one_market_high_volatility(self):
        prices = list(self.p.simulate_future_prices(
            market_names=['#1'],
            fixing_dates=[datetime.date(2012, 1, 1)],
            observation_date=datetime.date(2011, 1, 1),
            path_count=1000, calibration_params={
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 50,
            },
        ))
        prices = [p[2].mean() for p in prices[1:]]  # For numpy.
        expected_prices = [10]
        for price, expected_price in zip(prices, expected_prices):
            self.assertNotEqual(price, expected_price)
            self.assertAlmostEqual(price, expected_price, places=0)

    def test_simulate_future_prices_two_markets_zero_volatility(self):
        prices = list(self.p.simulate_future_prices(
            market_names=['#1', '#2'],
            fixing_dates=[datetime.date(2011, 1, 2)],
            observation_date=datetime.date(2011, 1, 1),
            path_count=200000, calibration_params={
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 0,
                '#2-LAST-PRICE': 20,
                '#2-ACTUAL-HISTORICAL-VOLATILITY': 0,
                '#1-#2-CORRELATION': 0,
            },
        ))
        prices = [(p[0], p[1], p[2].mean()) for p in prices]  # For numpy.
        self.assertEqual(prices, [
            ('#1', datetime.date(2011, 1, 1), numpy.array([ 10.,  10.]).mean()),
            ('#1', datetime.date(2011, 1, 2), numpy.array([ 10.,  10.]).mean()),
            ('#2', datetime.date(2011, 1, 1), numpy.array([ 20.,  20.]).mean()),
            ('#2', datetime.date(2011, 1, 2), numpy.array([ 20.,  20.]).mean()),
        ])

    def test_simulate_future_prices_two_markets_high_volatility_zero_correlation(self):
        prices = list(self.p.simulate_future_prices(
            market_names=['#1', '#2'],
            fixing_dates=[datetime.date(2012, 1, 1)],
            observation_date=datetime.date(2011, 1, 1),
            path_count=1000, calibration_params={
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 50,
                '#2-LAST-PRICE': 20,
                '#2-ACTUAL-HISTORICAL-VOLATILITY': 50,
                '#1-#2-CORRELATION': 0,
            },
        ))
        prices = [p[2].mean() for p in prices]  # For numpy.
        expected_prices = [10, 10, 20, 20]
        for price, expected_price in zip(prices, expected_prices):
            self.assertAlmostEqual(price, expected_price, places=0)

    def test_simulate_future_prices_two_markets_high_volatility_positive_correlation(self):
        prices = list(self.p.simulate_future_prices(
            market_names=['#1', '#2'],
            fixing_dates=[datetime.date(2012, 1, 1)],
            observation_date=datetime.date(2011, 1, 1),
            path_count=1000, calibration_params={
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 50,
                '#2-LAST-PRICE': 20,
                '#2-ACTUAL-HISTORICAL-VOLATILITY': 50,
                '#1-#2-CORRELATION': 0.5,
            },
        ))
        prices = [p[2].mean() for p in prices]  # For numpy.
        expected_prices = [10, 10, 20, 20]
        for price, expected_price in zip(prices, expected_prices):
            self.assertAlmostEqual(price, expected_price, places=0)
