import datetime
import unittest

from abc import ABCMeta
from time import sleep

import scipy
from eventsourcing.domain.model.events import assert_event_handlers_empty
from six import with_metaclass

from quantdsl.application.with_pythonobjects import QuantDslApplicationWithPythonObjects
from quantdsl.domain.model.call_result import make_call_result_id, CallResult
from quantdsl.domain.model.contract_valuation import create_contract_valuation_id
from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.semantics import Market
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME


class ApplicationTestCaseMixin(with_metaclass(ABCMeta)):
    skip_assert_event_handers_empty = False
    NUMBER_DAYS = 5
    NUMBER_MARKETS = 2
    NUMBER_WORKERS = 30
    PATH_COUNT = 2000

    def setUp(self):
        if not self.skip_assert_event_handers_empty:
            assert_event_handlers_empty()
        # super(ContractValuationTestCase, self).setUp()

        scipy.random.seed(1354802735)

        self.setup_application()

    def tearDown(self):
        if self.app is not None:
            self.app.close()
        if not self.skip_assert_event_handers_empty:
            assert_event_handlers_empty()
        # super(ContractValuationTestCase, self).tearDown()

    def setup_application(self):
        self.app = QuantDslApplicationWithPythonObjects()


class TestCase(ApplicationTestCaseMixin, unittest.TestCase):

    def setUp(self):
        super(TestCase, self).setUp()

    def tearDown(self):
        super(TestCase, self).tearDown()


class ContractValuationTestCase(ApplicationTestCaseMixin):

    def setup_market_simulation(self, contract_specification):
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        calibration_params = {
            '#1-LAST-PRICE': 10,
            '#2-LAST-PRICE': 10,
            '#1-ACTUAL-HISTORICAL-VOLATILITY': 50,
            '#2-ACTUAL-HISTORICAL-VOLATILITY': 50,
            '#1-#2-CORRELATION': 0.0,
            'NBP-LAST-PRICE': 10,
            'TTF-LAST-PRICE': 11,
            'NBP-ACTUAL-HISTORICAL-VOLATILITY': 50,
            'TTF-ACTUAL-HISTORICAL-VOLATILITY': 40,
            'NBP-TTF-CORRELATION': 0.4,
        }
        market_calibration =  self.app.register_market_calibration(price_process_name, calibration_params)
        market_names, fixing_dates = self.app.list_market_names_and_fixing_dates(contract_specification)
        observation_date = datetime.date(2011, 1, 1)
        path_count = self.PATH_COUNT
        market_simulation = self.app.register_market_simulation(
            market_calibration_id=market_calibration.id,
            market_names=market_names,
            fixing_dates=fixing_dates,
            observation_date=observation_date,
            path_count=path_count,
            interest_rate='2.5',
        )
        return market_simulation

    def assert_contract_value(self, specification, expected_value, expected_deltas=None, expected_call_count=None):
        # Register the specification (creates call dependency graph).
        contract_specification = self.app.register_contract_specification(specification=specification)

        # Check the call count (the number of nodes of the call dependency graph).
        call_count = len(list(regenerate_execution_order(contract_specification.id, self.app.call_link_repo)))

        if expected_call_count is not None:
            self.assertEqual(call_count, expected_call_count)

        # Generate the market simulation.
        market_simulation = self.setup_market_simulation(contract_specification)

        # Generate the contract valuation ID.
        contract_valuation_id = create_contract_valuation_id()
        call_result_id = make_call_result_id(contract_valuation_id, contract_specification.id)

        # Listen for the call result, if possible.
        # Todo: Listen for results, rather than polling for results - there will be less lag.
        # call_result_listener = None

        # Start the contract valuation.
        self.app.start_contract_valuation(contract_valuation_id, contract_specification.id, market_simulation)

        # # Get the call result.
        # if call_result_listener:
        #     call_result_listener.wait()

        main_result = self.get_result(call_result_id, call_count)

        # Check the call result.
        assert isinstance(main_result, CallResult)
        self.assertAlmostEqual(self.scalar(main_result.result_value), expected_value, places=2)

        if expected_deltas is None:
            return

        # Generate the contract valuation deltas.
        assert isinstance(market_simulation, MarketSimulation)
        for market_name in expected_deltas.keys():

            # Compute the delta.
            perturbed_value = main_result.perturbed_values[market_name].mean()
            market_calibration = self.app.market_calibration_repo[market_simulation.market_calibration_id]
            assert isinstance(market_calibration, MarketCalibration)
            last_price = market_calibration.calibration_params['%s-LAST-PRICE' % market_name]
            price_perturbation = Market.PERTURBATION_FACTOR * last_price
            contract_delta = (perturbed_value - main_result.result_value) / price_perturbation

            # Check the delta.
            self.assertAlmostEqual(contract_delta.mean(), expected_deltas[market_name], places=2, msg=market_name)

    def scalar(self, contract_value):
        if isinstance(contract_value, scipy.ndarray):
            contract_value = contract_value.mean()
        return contract_value

    def get_result(self, call_result_id, call_count):
        patience = max(call_count, 10) * 1.5 * (max(self.PATH_COUNT, 2000) / 1000)  # Guesses.
        # while patience > 0:
        while True:
            if call_result_id in self.app.call_result_repo:
                break
            interval = 0.1
            self.sleep(interval)
            patience -= interval
        else:
            self.fail("Timeout whilst waiting for result")
        call_result = self.app.call_result_repo[call_result_id]
        return call_result

    def sleep(self, interval):
        sleep(interval)


class ExpressionTests(ContractValuationTestCase):

    def test_generate_valuation_addition(self):
        self.assert_contract_value("""1 + 2""", 3)
        self.assert_contract_value("""2 + 4""", 6)

    def test_market(self):
        self.assert_contract_value("Market('#1')", 10, {'#1': 1})
        self.assert_contract_value("Market('#2')", 10, {'#2': 1})

    def test_market_plus(self):
        self.assert_contract_value("Market('#1') + 10", 20)
        self.assert_contract_value("Market('#2') + 20", 30)

    def test_market_minus(self):
        self.assert_contract_value("Market('#1') - 10", 0)
        self.assert_contract_value("Market('#2') - 10", 0)

    def test_market_multiply_market(self):
        self.assert_contract_value("Market('#1') * Market('#2')", 100)

    def test_market_divide(self):
        self.assert_contract_value("Market('#1') / 10", 1)

    def test_fixing(self):
        specification = "Fixing(Date('2012-01-01'), Market('NBP'))"
        self.assert_contract_value(specification, 10.1083)

    def test_wait(self):
        specification = "Wait(Date('2012-01-01'), Market('NBP'))"
        self.assert_contract_value(specification, 9.8587)

    def test_settlement(self):
        specification = "Settlement(Date('2012-01-01'), Market('NBP'))"
        self.assert_contract_value(specification, 9.753)

    def test_choice(self):
        specification = "Fixing(Date('2012-01-01'), Choice( Market('NBP') - 9, 0))"
        self.assert_contract_value(specification, 2.5178)

    def test_max(self):
        specification = "Fixing(Date('2011-01-01'), Max(Market('#1'), Market('#2')))"
        self.assert_contract_value(specification, 10.0000)
        specification = "Fixing(Date('2012-01-01'), Max(Market('#1'), Market('#2')))"
        self.assert_contract_value(specification, 13.0250)

    def test_bermudan(self):
        specification = """
Fixing(Date('2011-06-01'), Choice(Market('NBP') - 9,
    Fixing(Date('2012-01-01'), Choice(Market('NBP') - 9, 0))))
"""
        self.assert_contract_value(specification, 2.6093, expected_deltas={'NBP': 0.7123})

    def test_identical_fixings(self):
        specification = "Fixing(Date('2012-01-02'), Market('#1')) - Fixing(Date('2012-01-02'), Market('#1'))"
        self.assert_contract_value(specification, 0)

    def test_uncorrelated_markets(self):
        specification = """
Max(
    Fixing(
        Date('2012-01-01'),
        Market('#1')
    ) *
    Fixing(
        Date('2012-01-01'),
        Market('#2')
    ) / 10.0,
    0.0
) - Max(
    Fixing(
        Date('2013-01-01'),
        Market('#1')
    ), 0
)"""
        self.assert_contract_value(specification, -0.264)

    def test_correlated_markets(self):
        specification = """
Max(
    Fixing(
        Date('2012-01-01'),
        Market('TTF')
    ) *
    Fixing(
        Date('2012-01-01'),
        Market('NBP')
    ) / 10.0,
    0.0
) - Max(
    Fixing(
        Date('2013-01-01'),
        Market('TTF')
    ), 0
)"""
        self.assert_contract_value(specification, 0.9355)

    def test_futures(self):
        specification = "Wait(Date('2012-01-01'), Market('#1') - 9)"
        self.assert_contract_value(specification, 1.0809)

    def test_european_zero_volatility(self):
        self.assert_contract_value("Wait(Date('2012-01-01'), Choice(Market('#1') - 9, 0))", 2.4557)

    def test_european_high_volatility(self):
        self.assert_contract_value("Wait(Date('2012-01-01'), Choice(Market('NBP') - 9, 0))", 2.4557)

    def test_sum_contracts(self):
        specification = """
Fixing(
    Date('2011-06-01'),
    Choice(
        Market('NBP') - 9,
        Fixing(
            Date('2012-01-01'),
            Choice(
                Market('NBP') - 9,
                0
            )
        )
    )
) + Fixing(
    Date('2011-06-01'),
    Choice(
        Market('NBP') - 9,
        Fixing(
            Date('2012-01-01'),
            Choice(
                Market('NBP') - 9,
                0
            )
        )
    )
)
"""
        self.assert_contract_value(specification, 5.2187)

    def test_brownian_increments(self):
        specification = """
Wait(
    Date('2012-03-15'),
    Max(
        Fixing(
            Date('2012-01-01'),
            Market('#1')
        ) /
        Fixing(
            Date('2011-01-01'),
            Market('#1')
        ),
        1.0
    ) -
    Max(
        Fixing(
            Date('2013-01-01'),
            Market('#1')
        ) /
        Fixing(
            Date('2012-01-01'),
            Market('#1')
        ),
        1.0
    )
)"""
        self.assert_contract_value(specification, 0.005, expected_deltas={'#1': 0})


class FunctionTests(ContractValuationTestCase):

    def test_functional_fibonacci_numbers(self):
        fib_tmpl = """
def fib(n): return fib(n-1) + fib(n-2) if n > 1 else n
fib(%d)
"""
        # self.assert_contract_value(fib_tmpl % 0, 0, expected_call_count=2)
        # self.assert_contract_value(fib_tmpl % 1, 1, expected_call_count=2)
        # self.assert_contract_value(fib_tmpl % 2, 1, expected_call_count=4)
        # self.assert_contract_value(fib_tmpl % 3, 2, expected_call_count=5)
        self.assert_contract_value(fib_tmpl % 4, 3, expected_call_count=6)
        # self.assert_contract_value(fib_tmpl % 5, 5, expected_call_count=7)
        # self.assert_contract_value(fib_tmpl % 6, 8, expected_call_count=8)
        # self.assert_contract_value(fib_tmpl % 7, 13, expected_call_count=9)
        # self.assert_contract_value(fib_tmpl % 17, 1597, expected_call_count=19)

    def test_functional_derivative_option_definition(self):
        specification = """
def Option(date, strike, x, y):
    return Wait(date, Choice(x - strike, y))

Option(Date('2012-01-01'), 9, Underlying(Market('NBP')), 0)
"""
        self.assert_contract_value(specification, 2.4557, expected_call_count=2)

    def test_functional_european_option_definition(self):
        specification = """
def Option(date, strike, underlying, alternative):
    return Wait(date, Choice(underlying - strike, alternative))

def European(date, strike, underlying):
    return Option(date, strike, underlying, 0)

European(Date('2012-01-01'), 9, Market('NBP'))
"""
        self.assert_contract_value(specification, 2.4557, {'NBP': 0.6743}, expected_call_count=3)

    def test_generate_valuation_american_option(self):
        american_option_tmpl = """
def American(starts, ends, strike, underlying):
    if starts < ends:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )
    else:
        Option(starts, strike, underlying, 0)

@nostub
def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

American(Date('%(starts)s'), Date('%(ends)s'), %(strike)s, Market('%(underlying)s'))
"""
        self.assert_contract_value(american_option_tmpl % {
            'starts':'2011-01-02',
            'ends': '2011-01-04',
            'strike': 9,
            'underlying': '#1'
        }, 1.1874, {'#1': 1.0185}, expected_call_count=4)

    def test_generate_valuation_swing_option(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-01-05'), Market('NBP'), 3)
"""
        self.assert_contract_value(specification, 30.20756, {'NBP': 30.2076}, expected_call_count=15)


class LongerTests(ContractValuationTestCase):

    def test_value_swing_option(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, Market(underlying)),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-1-1'), Date('2011-1-5'), 'NBP', 3)
"""
        self.assert_contract_value(specification, 30.2081, expected_call_count=15)

    def _test_value_swing_option_with_forward_markets(self):
        specification = """
def Swing(start_date, end_date, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, quantity-1) + Fixing(start_date, ForwardMarket('NBP', start_date + TimeDelta('1d'))),
            Swing(start_date + TimeDelta('1d'), end_date, quantity)
        )
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-1-4'), 30)
"""
        self.assert_contract_value(specification, 20.00, expected_call_count=11)

    def _test_generate_valuation_power_plant_option(self):
        specification = """
def PowerPlant(start_date, end_date, underlying, time_since_off):
    if (start_date < end_date):
        Choice(
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, 0) + ProfitFromRunning(start_date, underlying, time_since_off),
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, NextTime(time_since_off)),
        )
    else:
        return 0

@nostub
def NextTime(time_since_off):
    if time_since_off == 2:
        return 2
    else:
        return time_since_off + 1

@nostub
def ProfitFromRunning(start_date, underlying, time_since_off):
    if time_since_off == 0:
        return Fixing(start_date, underlying)
    elif time_since_off == 1:
        return 0.9 * Fixing(start_date, underlying)
    else:
        return 0.8 * Fixing(start_date, underlying)

PowerPlant(Date('2012-01-01'), Date('2013-06-01'), Market('#1'), 30)
"""
        self.assert_contract_value(specification, 48, expected_call_count=2067)


class SpecialTests(ContractValuationTestCase):

    def test_simple_expression_with_market(self):
        dsl = "Market('NBP') + 2 * Market('TTF')"
        self.assert_contract_value(dsl, 32, {'NBP': 1, 'TTF': 2}, expected_call_count=1)

    def test_simple_function_with_market(self):
        dsl = """
def F():
  Market('NBP') + 2 * Market('TTF')

F()
"""
        self.assert_contract_value(dsl, 32, {'NBP': 1, 'TTF': 2}, expected_call_count=2)

    def test_generate_valuation_swing_option(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-01-05'), Market('NBP'), 3)
"""
        self.assert_contract_value(specification, 30.20756, {'NBP': 3.0207}, expected_call_count=15)
        # self.assert_contract_value(specification, 30.20756, {}, expected_call_count=15)

    def test_reuse_unperturbed_call_results(self):
        specification = """
def SumTwoMarkets(market_name1, market_name2):
    GetMarket(market_name1) + GetMarket(market_name2)

def GetMarket(market_name):
    Market(market_name)

SumTwoMarkets('NBP', 'TTF')
"""
        self.assert_contract_value(specification,
                                   expected_value=21,
                                   expected_deltas={'NBP': 1, 'TTF': 1},
                                   expected_call_count=4,
                                   )

    def test_reuse_unperturbed_call_results2(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Exercise(Swing, start_date, end_date, underlying, quantity),
            Hold(Swing, start_date, end_date, underlying, quantity)
        )
    else:
        return 0

@nostub
def Exercise(f, start_date, end_date, underlying, quantity):
    return Hold(f, start_date, end_date, underlying, quantity - 1) + Fixing(start_date, underlying)

@nostub
def Hold(f, start_date, end_date, underlying, quantity):
    return f(start_date + TimeDelta('1d'), end_date, underlying, quantity)

Swing(Date('2011-1-1'), Date('2011-1-4'), Market('#1'), 2) * 1 + \
Swing(Date('2011-1-1'), Date('2011-1-4'), Market('#2'), 2) * 2
"""
        self.assert_contract_value(specification,
                                   expected_call_count=19,
                                   expected_value=60.4826,
                                   expected_deltas={'#1': 2.0168, '#2': 4.0313},
                                   )

class ExperimentalTests(ContractValuationTestCase):

    def test_simple_expression_with_market(self):
        dsl = """
def Swing(start, end, step, market, quantity):
    if (quantity != 0) and (start <= end):
        Max(
            HoldSwing(start, end, step, market, quantity),
            ExerciseSwing(start, end, step, market, quantity, 1)
        )
    else:
        0

@nostub
def HoldSwing(start, end, step, market, quantity):
    On(start, Swing(start+step, end, step, market, quantity))

@nostub
def ExerciseSwing(start, end, step, market, quantity, vol):
    Settlement(start, vol*market) + HoldSwing(start, end, step, market, quantity-vol)

Swing(Date('2011-01-01'), Date('2011-01-02'), TimeDelta('1d'), Market('NBP'), 1)
"""

        self.assert_contract_value(dsl, 10, {'NBP': 1}, expected_call_count=6)


class ContractValuationTests(
    # ExperimentalTests,
    SpecialTests,
    # ExpressionTests,
    # FunctionTests,
    # LongerTests
): pass

