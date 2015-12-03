import datetime
import unittest
from time import sleep

from six.moves import queue
from threading import Thread

import scipy
from eventsourcing.domain.model.events import assert_event_handlers_empty

from quantdsl.application.with_pythonobjects import QuantDslApplicationWithPythonObjects
from quantdsl.domain.model.call_result import CallResult
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.simulated_price import SimulatedPrice, make_simulated_price_id
from quantdsl.domain.services.contract_valuations import evaluate_call_requirement
from quantdsl.domain.services.fixing_dates import list_fixing_dates
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.market_names import list_market_names
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME


# Specification. Calibration. Simulation. Evaluation.




class ApplicationTestCase(unittest.TestCase):

    def setUp(self):
        assert_event_handlers_empty()
        super(ApplicationTestCase, self).setUp()

        self.call_evaluation_queue = queue.Queue(maxsize=0)
        num_threads = 1

        self.app = QuantDslApplicationWithPythonObjects(
            call_evaluation_queue=self.call_evaluation_queue,
            # call_result_queue=Queue(),
        )

        scipy.random.seed(1354802735)

        def do_stuff(q):
            while True:
                item = q.get()
                try:
                    dependency_graph_id, contract_valuation_id, call_id = item
                    evaluate_call_requirement(
                        self.app.contract_valuation_repo[contract_valuation_id],
                        self.app.call_requirement_repo[call_id],
                        self.app.market_simulation_repo,
                        self.app.call_dependencies_repo,
                        self.app.call_result_repo,
                        self.app.simulated_price_repo
                    )
                finally:
                    q.task_done()

        for i in range(num_threads):
            worker = Thread(target=do_stuff, args=(self.call_evaluation_queue,))
            worker.setDaemon(True)
            worker.start()

    def tearDown(self):
        self.app.close()
        self.call_evaluation_queue.join()
        assert_event_handlers_empty()
        super(ApplicationTestCase, self).tearDown()


# Todo: More about market calibration, especially generating the calibration params from historical data.
class TestMarketCalibration(ApplicationTestCase):

    pass


class TestMarketSimulation(ApplicationTestCase):

    NUMBER_MARKETS = 2
    NUMBER_DAYS = 5
    PATH_COUNT = 200

    def test_register_market_simulation(self):
        # Set up the market calibration.
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        calibration_params = {
            '#1-LAST-PRICE': 10,
            '#2-LAST-PRICE': 20,
            '#1-ACTUAL-HISTORICAL-VOLATILITY': 10,
            '#2-ACTUAL-HISTORICAL-VOLATILITY': 20,
            '#1-#2-CORRELATION': 0.5,
        }
        market_calibration = self.app.register_market_calibration(price_process_name, calibration_params)

        # Create a market simulation for a list of markets and fixing times.
        market_names = ['#%d' % (i+1) for i in range(self.NUMBER_MARKETS)]
        date_range = [datetime.date(2011, 1, 1) + datetime.timedelta(days=i) for i in range(self.NUMBER_DAYS)]
        fixing_dates = date_range[1:]
        observation_date = date_range[0]
        path_count = self.PATH_COUNT

        market_simulation = self.app.register_market_simulation(
            market_calibration_id=market_calibration.id,
            market_names=market_names,
            fixing_dates=fixing_dates,
            observation_date=observation_date,
            path_count=path_count,
            interest_rate=2.5,
        )

        assert isinstance(market_simulation, MarketSimulation)
        assert market_simulation.id
        market_simulation = self.app.market_simulation_repo[market_simulation.id]
        assert isinstance(market_simulation, MarketSimulation)
        self.assertEqual(market_simulation.market_calibration_id, market_calibration.id)
        self.assertEqual(market_simulation.market_names, ['#1', '#2'])
        self.assertEqual(market_simulation.fixing_dates, [datetime.date(2011, 1, i) for i in range(2, 6)])
        self.assertEqual(market_simulation.observation_date, datetime.date(2011, 1, 1))
        self.assertEqual(market_simulation.path_count, self.PATH_COUNT)

        # Check there are simulated prices for all markets at all fixing times.
        for market_name in market_names:
            for fixing_date in fixing_dates:
                simulated_price_id = make_simulated_price_id(market_simulation.id, market_name, fixing_date)
                simulated_price = self.app.simulated_price_repo[simulated_price_id]
                self.assertIsInstance(simulated_price, SimulatedPrice)
                self.assertTrue(simulated_price.value.mean())


class TestContractValuation(ApplicationTestCase):

    NUMBER_MARKETS = 2
    NUMBER_DAYS = 5
    PATH_COUNT = 2000

    def test_generate_valuation_simple_addition(self):
        self.assert_contract_value("""1 + 2""", 3)

    def test_market(self):
        self.assert_contract_value("Market('#1')", 10)
        self.assert_contract_value("Market('#2')", 20)

    def test_market_plus(self):
        self.assert_contract_value("Market('#1') + 10", 20)
        self.assert_contract_value("Market('#2') + 10", 30)

    def test_market_minus(self):
        self.assert_contract_value("Market('#1') - 10", 0)
        self.assert_contract_value("Market('#2') - 10", 10)

    def test_market_multiply_market(self):
        self.assert_contract_value("Market('#1') * Market('#2')", 200)

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
        self.assert_contract_value(specification, 20.0000)
        specification = "Fixing(Date('2012-01-01'), Max(Market('#1'), Market('#2')))"
        self.assert_contract_value(specification, 20.1191)

    def test_identical_fixings(self):
        specification = "Fixing(Date('2012-01-02'), Market('#1')) - Fixing(Date('2012-01-02'), Market('#1'))"
        self.assert_contract_value(specification, 0)

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
) - 2 * Max(
    Fixing(
        Date('2013-01-01'),
        Market('#1')
    ), 0
)"""
        self.assert_contract_value(specification, -0.0714)

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
        self.assert_contract_value(specification, 1.1923)

    def test_futures(self):
        specification = "Wait(Date('2012-01-01'), Market('#1') - 9)"
        self.assert_contract_value(specification, 0.9753)

    def test_european_zero_volatility(self):
        self.assert_contract_value("Wait(Date('2012-01-01'), Choice(Market('#1') - 9, 0))", 0.9753)

    def test_european_high_volatility(self):
        self.assert_contract_value("Wait(Date('2012-01-01'), Choice(Market('NBP') - 9, 0))", 2.4557)

    def test_bermudan(self):
        specification = """
Fixing(Date('2011-06-01'), Choice(Market('NBP') - 9,
    Fixing(Date('2012-01-01'), Choice(Market('NBP') - 9, 0))))
"""
        self.assert_contract_value(specification, 2.6093)

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

    def test_functional_fibonacci_numbers(self):
        fib_tmpl = """
def fib(n): return fib(n-1) + fib(n-2) if n > 1 else n
fib(%d)
"""
        self.assert_contract_value(fib_tmpl % 0, 0, expected_call_count=2)
        self.assert_contract_value(fib_tmpl % 1, 1, expected_call_count=2)
        self.assert_contract_value(fib_tmpl % 2, 1, expected_call_count=4)
        self.assert_contract_value(fib_tmpl % 3, 2, expected_call_count=5)
        self.assert_contract_value(fib_tmpl % 4, 3, expected_call_count=6)
        self.assert_contract_value(fib_tmpl % 5, 5, expected_call_count=7)
        self.assert_contract_value(fib_tmpl % 6, 8, expected_call_count=8)
        self.assert_contract_value(fib_tmpl % 7, 13, expected_call_count=9)
        self.assert_contract_value(fib_tmpl % 17, 1597, expected_call_count=19)

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
        self.assert_contract_value(specification, 2.4557, expected_call_count=3)

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
        }, 1, expected_call_count=4)

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
        self.assert_contract_value(specification, 30.2081, expected_call_count=15)

    def test_generate_valuation_power_plant_option(self):
        specification = """
def PowerPlant(start_date, end_date, underlying, time_since_off):
    if (start_date < end_date):
        Choice(
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, 0)
                + ProfitFromRunning(start_date, underlying, time_since_off),
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, NextTime(time_since_off))
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

PowerPlant(Date('2012-01-01'), Date('2012-01-06'), Market('#1'), 2)
"""
        self.assert_contract_value(specification, 48, expected_call_count=16)

    def assert_contract_value(self, specification, expected_value, expected_call_count=None):
        contract_specification = self.app.register_contract_specification(specification=specification)

        # Check the call count (the number of nodes of the call dependency graph).
        if expected_call_count is not None:
            call_count = len(list(regenerate_execution_order(contract_specification.id, self.app.call_link_repo)))
            self.assertEqual(call_count, expected_call_count)

        # Generate the market simulation.
        market_simulation = self.setup_market_simulation(contract_specification)

        # Generate the contract valuation.
        self.app.start_contract_valuation(contract_specification.id, market_simulation)

        count = 0
        while count < 600:
            # Check the result.
                try:
                    call_result = self.app.call_result_repo[contract_specification.id]
                    break
                except KeyError:
                    count += 1
                    sleep(0.1)

        assert isinstance(call_result, CallResult)
        self.assertAlmostEqual(call_result.scalar_result_value, expected_value, places=2)

    def setup_market_simulation(self, contract_specification):
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        calibration_params = {
            '#1-LAST-PRICE': 10,
            '#2-LAST-PRICE': 20,
            '#1-ACTUAL-HISTORICAL-VOLATILITY': 0,
            '#2-ACTUAL-HISTORICAL-VOLATILITY': 20,
            '#1-#2-CORRELATION': 0,
            'NBP-LAST-PRICE': 10,
            'TTF-LAST-PRICE': 10,
            'NBP-ACTUAL-HISTORICAL-VOLATILITY': 50,
            'TTF-ACTUAL-HISTORICAL-VOLATILITY': 50,
            'NBP-TTF-CORRELATION': 0.5,
        }
        market_calibration =  self.app.register_market_calibration(price_process_name, calibration_params)

        market_names = list_market_names(contract_specification)
        fixing_dates = list_fixing_dates(contract_specification.id, self.app.call_requirement_repo, self.app.call_link_repo)
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
