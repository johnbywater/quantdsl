from __future__ import division


# Todo: Review this module, and reestablish tests that have not already been reproduced.
import datetime
import sys
import unittest

from quantdsl.priceprocess.blackscholes import BlackScholesPriceProcess
from quantdsl.semantics import DslExpression, DslNamespace
from quantdsl.tests.test_parser import dsl_eval, dsl_compile


def suite():
    return unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])


# class MockImage(object):
#
#     def __init__(self, price_process):
#         self.price_process = price_process
#
#
# class MockPriceProcess(object): pass


class DslTestCase(unittest.TestCase):

    def assertValuation(self, dsl_source=None, expected_value=None, expected_delta=None, expected_gamma=None,
            tolerance_value=0.05, tolerance_delta = 0.1, tolerance_gamma=0.1):

        # Check option value.
        observation_date = datetime.date(2011, 1, 1)
        estimated_value = self.calc_value(dsl_source, observation_date)['mean']
        self.assertTolerance(estimated_value, expected_value, tolerance_value)

        # Todo: Reinstate the delta tests.
        return
        # Check deltas.
        markets = self.pricer.get_markets()
        if not markets:
            assert self.expected_delta == None
            return
        market = list(markets)[0]
        # Check option delta.
        estimated_delta = self.pricer.calc_delta(market)
        self.assertTolerance(estimated_delta, expected_delta, tolerance_delta)

        # Todo: Decide what to do with gamma (too much noise to pass tests consistently at the mo). Double-side differentials?
        # Check option gamma.
        #estimatedGamma = self.pricer.calcGamma(market)
        #roundedGamma = round(estimatedGamma, self.DECIMALS)
        #expected_gamma = round(self.expected_gamma, self.DECIMALS)
        #msg = "Value: %s  Expected: %s" % (roundedGamma, expected_gamma)
        #self.assertEqual(roundedGamma, expected_gamma, msg)

    def assertTolerance(self, estimated, expected, tolerance):
        upper = expected + tolerance
        lower = expected - tolerance
        assert lower <= estimated <= upper, "Estimated '%s' not close enough to expected '%s' (tolerance '%s')." % (estimated, expected, tolerance)

    def calc_value(self, dsl_source, observation_date):
        # Todo: Rename 'allRvs' to 'simulatedPrices'?
        evaluation_kwds = DslNamespace({
            'observation_date': observation_date,
            'interest_rate': '2.5',
            'market_calibration': {
                '#1-LAST-PRICE': 10,
                '#1-ACTUAL-HISTORICAL-VOLATILITY': 50,
                '#2-LAST-PRICE': 10,
                '#2-ACTUAL-HISTORICAL-VOLATILITY': 50,
                '#1-#2-CORRELATION': 0.0,
                'NBP-LAST-PRICE': 10,
                'NBP-ACTUAL-HISTORICAL-VOLATILITY': 50,
                'TTF-LAST-PRICE': 11,
                'TTF-ACTUAL-HISTORICAL-VOLATILITY': 40,
                'BRENT-LAST-PRICE': 90,
                'BRENT-ACTUAL-HISTORICAL-VOLATILITY': 60,
                'NBP-TTF-CORRELATION': 0.4,
                'BRENT-TTF-CORRELATION': 0.5,
                'BRENT-NBP-CORRELATION': 0.3,
            },
            'path_count': 200000,
            # 'simulated_price_repo': {
            #     'sim1#12011-01-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1#12011-01-03': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1#12011-06-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1#12012-01-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1#12012-01-02': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1#12012-01-03': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1#12012-06-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1#12013-01-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #
            #     'sim1#22011-01-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1#22012-01-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #
            #     'sim1TTF2012-01-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1NBP2012-01-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1TTF2013-01-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            #     'sim1NBP2013-01-01': Mock(spec=SimulatedPrice, value=scipy.array([10])),
            # },
            'simulation_id': 'sim1',
            'first_commodity_name': '#1',
        })
        return dsl_eval(dsl_source, evaluation_kwds=evaluation_kwds)


        # dsl_expr = dsl_compile(dsl_source)
        #
        # evaluation_kwds = DslNamespace({
        #     'observation_date': observation_date,
        #     'present_time': observation_date,
        #     'simulated_price_repo': {},
        #     'interest_rate': '2.5',
        #     'calibration': {
        #         '#1-LAST-PRICE': 10,
        #         '#1-ACTUAL-HISTORICAL-VOLATILITY': 50,
        #         '#2-LAST-PRICE': 10,
        #         '#2-ACTUAL-HISTORICAL-VOLATILITY': 50,
        #     },
        #     'allRvs': BlackScholesPriceProcess().getAllRvs(dsl_expr, observation_date, path_count=100000),
        # })
        # assert isinstance(dsl_expr, DslExpression)
        # value = dsl_expr.evaluate(**evaluation_kwds)
        # if hasattr(value, 'mean'):
        #     value = value.mean()
        # return value
        #

class TestDslMarket(DslTestCase):

    def testValuation(self):
        specification = "Market('#1')"
        self.assertValuation(specification, 10, 1, 0)

#
# class TestDslFixing(DslTestCase):
#
#     def testValuation(self):
#         specification = "Fixing(Date('2012-01-01'), Market('#1'))"
#         self.assertValuation(specification, 10, 1, 0)
#
#
# class TestDslWait(DslTestCase):
#
#     def testValuation(self):
#         specification = "Wait(Date('2012-01-01'), Market('#1'))"
#         self.assertValuation(specification, 9.753, 0.975, 0)
#
#
# class TestDslSettlement(DslTestCase):
#
#     def testValuation(self):
#         specification = "Settlement(Date('2012-01-01'), Market('#1'))"
#         self.assertValuation(specification, 9.753, 0.975, 0)
#
#
# class TestDslChoice(DslTestCase):
#
#     def testValuation(self):
#         specification = "Fixing(Date('2012-01-01'), Choice( Market('#1') - 9, 0))"
#         self.assertValuation(specification, 2.416, 0.677, 0.07)
#
#
# class TestDslMax(DslTestCase):
#
#     def testValuation(self):
#         specification = "Fixing(Date('2012-01-01'), Max(Market('#1'), Market('#2')))"
#         self.assertValuation(specification, 12.766, 0.636, 0)
#         #self.assertValuation(specification, 11.320, 0.636, 0)
#

# class TestDslAdd(DslTestCase):
#
#     def test_valuation(self):
#         specification = "10 + Market('#1')"
#         self.assertValuation(specification, 20, 1, 0)
#
#     def test_valuation2(self):
#         specification = "10 + Market('#2')"
#         self.assertValuation(specification, 20, 1, 0)
#
#
# class TestDslSubtract(DslTestCase):
#
#     def testValuation(self):
#         specification = "Market('#1') - 10"
#         self.assertValuation(specification, 0, 1, 0)
#
#
# class TestDslMultiply(DslTestCase):
#
#     def testValuation(self):
#         specification = "Market('#1') * Market('#2')"
#         self.assertValuation(specification, 100, 10, 0)
#
#
# class TestDslDivide(DslTestCase):
#
#     def testValuation(self):
#         specification = "Market('#1') / 10"
#         self.assertValuation(specification, 1, 0.1, 0)
#
#
# class TestDslIdenticalFixings(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# Fixing(Date('2012-01-01'), Market('#1')) - Fixing(Date('2012-01-01'), Market('#1'))
# """
#         self.assertValuation(specification, 0, 0, 0)
#
#
# class TestDslBrownianIncrements(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# Wait(
#     Date('2012-03-15'),
#     Max(
#         Fixing(
#             Date('2012-01-01'),
#             Market('#1')
#         ) /
#         Fixing(
#             Date('2011-01-01'),
#             Market('#1')
#         ),
#         1.0
#     ) -
#     Max(
#         Fixing(
#             Date('2013-01-01'),
#             Market('#1')
#         ) /
#         Fixing(
#             Date('2012-01-01'),
#             Market('#1')
#         ),
#         1.0
#     )
# )"""
#         self.assertValuation(specification, 0, 0, 0)
#
#
# class TestDslUncorrelatedMarkets(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# Max(
#     Fixing(
#         Date('2012-01-01'),
#         Market('#1')
#     ) *
#     Fixing(
#         Date('2012-01-01'),
#         Market('#2')
#     ) / 10.0,
#     0.0
# ) - Max(
#     Fixing(
#         Date('2013-01-01'),
#         Market('#1')
#     ), 0
# )"""
#         self.assertValuation(specification, 0, 0, 0, 0.07, 0.2, 0.2)  # Todo: Figure out why the delta sometimes evaluates to 1 for a period of time and then
#
#
# class TestDslCorrelatedMarkets(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# Max(
#     Fixing(
#         Date('2012-01-01'),
#         Market('TTF')
#     ) *
#     Fixing(
#         Date('2012-01-01'),
#         Market('NBP')
#     ) / 10.0,
#     0.0
# ) - Max(
#     Fixing(
#         Date('2013-01-01'),
#         Market('TTF')
#     ), 0
# )"""
#         self.assertValuation(specification, 0.92, 0, 0, 0.15, 0.2, 0.2)
#
#
# class TestDslFutures(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# Wait( Date('2012-01-01'),
#     Market('#1') - 9
# ) """
#         self.assertValuation(specification, 0.9753, 0.9753, 0)
#
#
# class TestDslEuropean(DslTestCase):
#
#     def testValuation(self):
#         specification = "Wait(Date('2012-01-01'), Choice(Market('#1') - 9, 0))"
#         self.assertValuation(specification, 2.356, 0.660, 0.068)
#
#
# class TestDslBermudan(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# Fixing( Date('2011-06-01'), Choice( Market('#1') - 9,
#     Fixing( Date('2012-01-01'), Choice( Market('#1') - 9, 0))
# ))
# """
#         self.assertValuation(specification, 2.401, 0.677, 0.0001)
#
#
# class TestDslSumContracts(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# Fixing(
#     Date('2011-06-01'),
#     Choice(
#         Market('#1') - 9,
#         Fixing(
#             Date('2012-01-01'),
#             Choice(
#                 Market('#1') - 9,
#                 0
#             )
#         )
#     )
# ) + Fixing(
#     Date('2011-06-01'),
#     Choice(
#         Market('#1') - 9,
#         Fixing(
#             Date('2012-01-01'),
#             Choice(
#                 Market('#1') - 9,
#                 0
#             )
#         )
#     )
# )
# """
#         self.assertValuation(specification, 4.812, 2 * 0.677, 2*0.07, 0.09, 0.2, 0.2)
#
#
# class TestDslAddition(DslTestCase):
#
#     def testValuation2(self):
#         specification = """
# Fixing( Date('2012-01-01'),
#     Max(Market('#1') - 9, 0) + Market('#1') - 9
# )
# """
#         self.assertValuation(specification, 3.416, 1.677, 0.07, 0.07, 0.2, 0.2)
#
#
# class TestDslFunctionDefSwing(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# def Swing(starts, ends, underlying, quantity):
#     if (quantity != 0) and (starts < ends):
#         return Choice(
#             Swing(starts + TimeDelta('1d'), ends, underlying, quantity-1) \
#             + Fixing(starts, underlying),
#             Swing(starts + TimeDelta('1d'), ends, underlying, quantity)
#         )
#     else:
#         return 0
# Swing(Date('2012-01-01'), Date('2012-01-03'), Market('#1'), 2)
# """
#         self.assertValuation(specification, 20.0, 2.0, 0.07, 0.06, 0.2, 0.2)
#
#
# class TestDslFunctionDefOption(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# def Option(date, strike, x, y):
#     return Wait(date, Choice(x - strike, y))
# Option(Date('2012-01-01'), 9, Underlying(Market('#1')), 0)
# """
#         self.assertValuation(specification, 2.356, 0.660, 0.068, 0.04, 0.2, 0.2)


# class TestDslFunctionDefEuropean(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# def Option(date, strike, underlying, alternative):
#     return Wait(date, Choice(underlying - strike, alternative))
#
# def European(date, strike, underlying):
#     return Option(date, strike, underlying, 0)
#
# European(Date('2012-01-01'), 9, Market('#1'))
# """
#         self.assertValuation(specification, 2.356, 0.660, 0.068, 0.04, 0.2, 0.2)


# class TestDslFunctionDefAmerican(DslTestCase):
#
#     def testValuation(self):
#         specification = """
# def Option(date, strike, underlying, alternative):
#     return Wait(date, Choice(underlying - strike, alternative))
#
# def American(starts, ends, strike, underlying, step):
#     Option(starts, strike, underlying, 0) if starts == ends else \
#     Option(starts, strike, underlying, American(starts + step, ends, strike, underlying, step))
#
# American(Date('2012-01-01'), Date('2012-01-3'), 9, Market('#1'), TimeDelta('1d'))
# """
#         self.assertValuation(specification, 2.356, 0.660, 0.068, 0.04, 0.2, 0.2)
