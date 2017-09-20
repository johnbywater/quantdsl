import datetime
import unittest
from abc import ABCMeta
from time import sleep

import scipy
from eventsourcing.domain.model.events import assert_event_handlers_empty
from six import with_metaclass

from quantdsl.application.with_pythonobjects import QuantDslApplicationWithPythonObjects
from quantdsl.domain.model.call_result import CallResult
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME


class ApplicationTestCaseMixin(with_metaclass(ABCMeta)):
    skip_assert_event_handers_empty = False
    NUMBER_DAYS = 5
    NUMBER_MARKETS = 2
    NUMBER_WORKERS = 20
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
    price_process_name = DEFAULT_PRICE_PROCESS_NAME
    calibration_params = {
        'market': ['#1', '#2', 'NBP', 'TTF', 'SPARKSPREAD'],
        'sigma': [0.5, 0.5, 0.5, 0.4, 0.4],
        'curve': {
            '#1': [
                ('2011-1-1', 10),
            ],
            '#2': [
                ('2011-1-1', 10),
            ],
            'NBP': [
                ('2011-1-1', 10),
            ],
            'TTF': [
                ('2011-1-1', 11),
            ],
            'SPARKSPREAD': [
                ('2011-1-1', 1),
            ],
        },
        'rho': [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.4, 0.0],
            [0.0, 0.0, 0.4, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        # '#1-LAST-PRICE': 10,
        # '#2-LAST-PRICE': 10,
        # '#1-ACTUAL-HISTORICAL-VOLATILITY': 50,
        # '#2-ACTUAL-HISTORICAL-VOLATILITY': 50,
        # '#1-#2-CORRELATION': 0.0,
        # 'NBP-LAST-PRICE': 10,
        # 'TTF-LAST-PRICE': 11,
        # 'NBP-ACTUAL-HISTORICAL-VOLATILITY': 50,
        # 'TTF-ACTUAL-HISTORICAL-VOLATILITY': 40,
        # 'NBP-TTF-CORRELATION': 0.4,
        # 'NBP-2011-01-LAST-PRICE': 10,
        # 'NBP-2011-01-ACTUAL-HISTORICAL-VOLATILITY': 10,
        # 'NBP-2012-01-LAST-PRICE': 10,
        # 'NBP-2012-01-ACTUAL-HISTORICAL-VOLATILITY': 10,
        # 'NBP-2012-01-NBP-2012-02-CORRELATION': 0.4,
        # 'NBP-2012-02-NBP-2012-03-CORRELATION': 0.4,
        # 'SPARKSPREAD-LAST-PRICE': 1,
        # 'SPARKSPREAD-ACTUAL-HISTORICAL-VOLATILITY': 40,
    }

    def assert_contract_value(self, specification, expected_value, expected_deltas=None, expected_call_count=None):

        # Register the specification (creates call dependency graph).
        contract_specification = self.app.compile(source_code=specification)

        # Check the call count (the number of nodes of the call dependency graph).
        call_count = self.calc_call_count(contract_specification.id)

        if expected_call_count is not None:
            self.assertEqual(call_count, expected_call_count)

        # Generate the market simulation.
        market_calibration = self.app.register_market_calibration(self.price_process_name, self.calibration_params)
        observation_date = datetime.datetime(2011, 1, 1)
        market_simulation = self.app.simulate(
            contract_specification=contract_specification,
            market_calibration=market_calibration,
            observation_date=observation_date,
            path_count=self.PATH_COUNT,
            interest_rate='2.5',
            perturbation_factor=0.001,
        )

        # Generate the contract valuation ID.

        # Listen for the call result, if possible.
        # Todo: Listen for results, rather than polling for results - there will be less lag.
        # call_result_listener = None

        # Start the contract valuation.
        contract_valuation = self.app.evaluate(contract_specification.id, market_simulation.id)
        assert isinstance(contract_valuation, ContractValuation)

        # # Get the call result.
        # if call_result_listener:
        #     call_result_listener.wait()

        main_result = self.get_result(contract_valuation)

        # Check the call result.
        assert isinstance(main_result, CallResult)
        self.assertAlmostEqual(self.scalar(main_result.result_value), expected_value, places=2)

        if expected_deltas is None:
            return

        # Generate the contract valuation deltas.
        assert isinstance(market_simulation, MarketSimulation)
        for perturbation in expected_deltas.keys():

            # Get the deltas.
            if perturbation not in main_result.perturbed_values:
                self.fail("There isn't a perturbed value for '{}': {}"
                          "".format(perturbation, list(main_result.perturbed_values.keys())))

            perturbed_value = main_result.perturbed_values[perturbation].mean()
            market_calibration = self.app.market_calibration_repo[market_simulation.market_calibration_id]
            assert isinstance(market_calibration, MarketCalibration)
            commodity_name = perturbation.split('-')[0]
            simulated_price_id = make_simulated_price_id(market_simulation.id, commodity_name,
                                                         market_simulation.observation_date,
                                                         market_simulation.observation_date)
            simulated_price = self.app.simulated_price_repo[simulated_price_id]

            dy = perturbed_value - main_result.result_value
            dx = market_simulation.perturbation_factor * simulated_price.value
            contract_delta = dy / dx

            # Check the delta.
            actual_value = contract_delta.mean()
            expected_value = expected_deltas[perturbation]
            error_msg = "{}: {} != {}".format(perturbation, actual_value, expected_value)
            self.assertAlmostEqual(actual_value, expected_value, places=2, msg=error_msg)

    def calc_call_count(self, contract_specification_id):
        return self.app.calc_call_count(contract_specification_id)

    def scalar(self, contract_value):
        if isinstance(contract_value, scipy.ndarray):
            contract_value = contract_value.mean()
        return contract_value

    def get_result(self, contract_valuation):
        assert isinstance(contract_valuation, ContractValuation)
        call_costs = self.app.calc_call_costs(contract_valuation.contract_specification_id)
        total_cost = sum(call_costs.values())

        patience = max(total_cost, 10) * 1.5 * (max(self.PATH_COUNT, 2000) / 1000)  # Guesses.
        while True:
            try:
                return self.app.get_result(contract_valuation)
            except:
                interval = 0.1
                self.sleep(interval)
                patience -= interval
                if not patience:
                    self.fail("Timeout whilst waiting for result")

    def sleep(self, interval):
        sleep(interval)


class ExpressionTests(ContractValuationTestCase):
    def test_generate_valuation_addition(self):
        self.assert_contract_value("""1 + 2""", 3)
        self.assert_contract_value("""2 + 4""", 6)

    def test_market(self):
        self.assert_contract_value("Lift('#1', Market('#1'))", 10, {'#1': 1})
        self.assert_contract_value("Lift('#2', Market('#2'))", 10, {'#2': 1})

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
        self.assert_contract_value(specification, 2.6093)

    def test_bermudan_with_alltime_delta(self):
        specification = """
Fixing(Date('2011-06-01'), Choice(Lift('NBP', Market('NBP')) - 9,
    Fixing(Date('2012-01-01'), Choice(Lift('NBP', Market('NBP')) - 9, 0))))
"""
        self.assert_contract_value(specification, 2.6093, expected_deltas={'NBP': 0.71})

    def test_bermudan_with_monthly_deltas(self):
        specification = """
Fixing(Date('2011-06-01'), Choice(Lift('NBP', 'monthly', Market('NBP')) - 9,
    Fixing(Date('2012-01-01'), Choice(Lift('NBP', 'monthly', Market('NBP')) - 9, 0))))
"""
        self.assert_contract_value(specification, 2.6093, expected_deltas={'NBP-2011-6': 0.2208})

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
Max(
    Fixing(
        Date('2012-01-01'),
        Lift('#1', Market('#1'))
    ) /
    Fixing(
        Date('2011-01-01'),
        Lift('#1', Market('#1'))
    ),
    1.0
) - Max(
    Fixing(
        Date('2013-01-01'),
        Lift('#1', Market('#1'))
    ) /
    Fixing(
        Date('2012-01-01'),
        Lift('#1', Market('#1'))
    ),
    1.0
)
"""
        # NB: Expected value should be 0.0000. It is slightly
        # off due to small path count, and consistently at the
        # slightly negative value due to the seed being fixed.
        self.assert_contract_value(specification, -0.01, expected_deltas={'#1': 0.00})


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

    def test_two_defs(self):
        dsl_source = """
def add(a, b):
    a + b

def mul(a, b):
    a if b == 1 else add(a, mul(a, b - 1))
mul(3, 3)
    """
        self.assert_contract_value(dsl_source, 9, expected_call_count=6)

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

European(Date('2012-01-01'), 9, Lift('NBP', Market('NBP')))
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

@inline
def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

American(Date('%(starts)s'), Date('%(ends)s'), %(strike)s, Lift('%(underlying)s', Market('%(underlying)s')))
"""
        self.assert_contract_value(american_option_tmpl % {
            'starts': '2011-01-02',
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
        self.assert_contract_value(specification, 30.20756, {'NBP': 3.02076}, expected_call_count=15)


class LongerTests(ContractValuationTestCase):
    def test_value_swing_option(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, 
            Market(underlying)),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-1-1'), Date('2011-1-5'), 'NBP', 3)
"""
        self.assert_contract_value(specification, 30.2081, expected_call_count=15)

    def test_generate_valuation_power_plant_option(self):
        specification = """
PowerPlant(Date('2012-01-01'), Date('2012-01-13'), Market('SPARKSPREAD'), 2)

def PowerPlant(start_date, end_date, underlying, time_since_off):
    if (start_date < end_date):
        Wait(start_date, Choice(
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, Running()) \
               + ProfitFromRunning(start_date, underlying, time_since_off),
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, Stopped(time_since_off)),
        ))
    else:
        return 0

@inline
def Running():
    return 0

@inline
def Stopped(time_since_off):
    return Min(2, time_since_off + 1)

@inline
def ProfitFromRunning(start_date, underlying, time_since_off):
    if time_since_off == 0:
        return Fixing(start_date, underlying)
    elif time_since_off == 1:
        return 0.9 * Fixing(start_date, underlying)
    else:
        return 0.8 * Fixing(start_date, underlying)

"""
        self.assert_contract_value(specification, 11.57, expected_call_count=37)


class SpecialTests(ContractValuationTestCase):
    def test_simple_expression_with_market(self):
        dsl = "Market('NBP') + 2 * Market('TTF')"
        self.assert_contract_value(dsl, 32, {('NBP', 2011, 1): 1, ('TTF', 2011, 1): 2}, expected_call_count=1)

    def test_simple_function_with_market(self):
        dsl = """
def F():
  Lift('NBP', Market('NBP')) + 2 * Lift('TTF', Market('TTF'))

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

Swing(Date('2011-01-01'), Date('2011-01-05'), Lift('NBP', Market('NBP')), 3)
"""
        self.assert_contract_value(specification, 30.20756, {'NBP': 3.0207}, expected_call_count=15)
        # self.assert_contract_value(specification, 30.20756, {}, expected_call_count=15)

    def test_reuse_unperturbed_call_results(self):
        specification = """
def SumTwoMarkets(market_name1, market_name2):
    GetMarket(market_name1) + GetMarket(market_name2)

def GetMarket(market_name):
    Lift(market_name, Market(market_name))

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

@inline
def Exercise(f, start_date, end_date, underlying, quantity):
    return Hold(f, start_date, end_date, underlying, quantity - 1) + Fixing(start_date, underlying)

@inline
def Hold(f, start_date, end_date, underlying, quantity):
    return f(start_date + TimeDelta('1d'), end_date, underlying, quantity)

Swing(Date('2011-1-1'), Date('2011-1-4'), Lift('#1', Market('#1')), 2) * 1 + \
Swing(Date('2011-1-1'), Date('2011-1-4'), Lift('#2', Market('#2')), 2) * 2
"""
        self.assert_contract_value(specification,
                                   expected_call_count=19,
                                   expected_value=60.4826,
                                   expected_deltas={'#1': 2.0168, '#2': 4.0313},
                                   )


class ExperimentalTests(ContractValuationTestCase):
    def test_value_swing_option_with_fixing_on_market(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, 
            Market(underlying)),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-1-1'), Date('2011-1-5'), 'NBP', 3)
"""
        self.assert_contract_value(specification, 30.2075, expected_call_count=15)

    def test_value_swing_option_without_fixings_or_settlements(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Market(underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-1-1'), Date('2011-1-5'), 'NBP', 3)
"""
        self.assert_contract_value(specification, 30.0000, expected_call_count=15)

    def test_value_swing_option_with_settlements_and_fixings_on_choice(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Wait(start_date, Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Market(underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        ))
    else:
        return 0

Swing(Date('2011-1-1'), Date('2011-1-5'), 'NBP', 3)
"""
        self.assert_contract_value(specification, 30.2399, expected_call_count=15)

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

@inline
def HoldSwing(start, end, step, market, quantity):
    On(start, Swing(start+step, end, step, market, quantity))

@inline
def ExerciseSwing(start, end, step, market, quantity, vol):
    Settlement(start, vol*market) + HoldSwing(start, end, step, market, quantity-vol)

Swing(Date('2011-01-01'), Date('2011-01-02'), TimeDelta('1d'), Lift('NBP', Market('NBP')), 1)
"""
        self.assert_contract_value(dsl, 10, {'NBP': 1}, expected_call_count=6)

    def test_value_swing_option_with_forward_markets(self):
        specification = """
def Swing(start_date, end_date, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Settlement(start_date, Fixing(start_date, Choice(
            Swing(start_date + TimeDelta('1m'), end_date, quantity-1) + ForwardMarket('NBP', start_date),
            Swing(start_date + TimeDelta('1m'), end_date, quantity)
        )))
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-4-1'), 30)
"""
        self.assert_contract_value(specification, 29.9575, {
            ('NBP', 2011, 2): 1.0,
            ('NBP', 2011, 3): 1.0,
            ('NBP', 2011, 4): 1.0,
        }, expected_call_count=11)

    def test_simple_forward_market(self):
        specification = """Lift('NBP', ForwardMarket('NBP', '2011-1-1'))"""
        self.assert_contract_value(specification, 10.00, {'NBP': 1.0}, expected_call_count=1)

    def test_gas_storage_option(self):
        specification_tmpl = """
def GasStorage(start, end, commodity_name, quantity, limit, step):
    if ((start < end) and (limit > 0)):
        if quantity <= 0:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Inject(start, end, commodity_name, quantity, limit, step, 1),
            ))
        elif quantity < limit:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Inject(start, end, commodity_name, quantity, limit, step, -1),
            ))
        else:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Inject(start, end, commodity_name, quantity, limit, step, 1),
                Inject(start, end, commodity_name, quantity, limit, step, -1),
            ))
    else:
        0

@inline
def Continue(start, end, commodity_name, quantity, limit, step):
    GasStorage(start + step, end, commodity_name, quantity, limit, step)

@inline
def Inject(start, end, commodity_name, quantity, limit, step, vol):
    Continue(start, end, commodity_name, quantity + vol, limit, step) - \
    Settlement(start, vol * ForwardMarket(commodity_name, start))

GasStorage(Date('%(start_date)s'), Date('%(end_date)s'), '%(commodity)s', %(quantity)s, %(limit)s, TimeDelta('1m'))
"""
        # No capacity.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 0
        }
        self.assert_contract_value(specification, 0.00, {}, expected_call_count=2)

        # Capacity, zero inventory.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 10
        }
        self.assert_contract_value(specification, 0.00, {}, expected_call_count=6)

        # Capacity, zero inventory, option in the future.
        specification = specification_tmpl % {
            'start_date': '2013-1-1',
            'end_date': '2013-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 10
        }
        self.assert_contract_value(specification, 0.0270, {}, expected_call_count=6)

        # Capacity, and inventory to discharge.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-3-1',
            'commodity': 'NBP',
            'quantity': 2,
            'limit': 2
        }
        self.assert_contract_value(specification, 19.9857, {}, expected_call_count=10)

        # Capacity, and inventory to discharge in future.
        specification = specification_tmpl % {
            'start_date': '2021-1-1',
            'end_date': '2021-3-1',
            'commodity': 'NBP',
            'quantity': 2,
            'limit': 2
        }
        self.assert_contract_value(specification, 15.3496, {}, expected_call_count=10)

        # Capacity, zero inventory, in future.
        specification = specification_tmpl % {
            'start_date': '2021-1-1',
            'end_date': '2021-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 2
        }
        self.assert_contract_value(specification, 0.0123, {}, expected_call_count=6)


class SingleTests(ContractValuationTestCase):
    def test_value_swing_option_with_forward_markets(self):
        specification = """
def Swing(start_date, end_date, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Settlement(start_date, Fixing(start_date,
            Choice(
                Swing(start_date + TimeDelta('1m'), end_date, quantity-1) + \
                    Lift('NBP', 'monthly', ForwardMarket('NBP', start_date)),
                Swing(start_date + TimeDelta('1m'), end_date, quantity)
            )
        ))
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-4-1'), 30)
"""
        self.assert_contract_value(specification, 29.9575, {
            'NBP-2011-1': 0.9939,
            'NBP-2011-3': 1.0,
            # ('NBP', 2011, 4): 1.0,
        }, expected_call_count=11)


class ContractValuationTests(
    SingleTests,
    ExperimentalTests,
    SpecialTests,
    ExpressionTests,
    FunctionTests,
    LongerTests
): pass
