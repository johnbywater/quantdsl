import datetime
import unittest

import numpy
import six
from eventsourcing.domain.model.events import assert_event_handlers_empty

from quantdsl.application.with_sqlalchemy import QuantDslApplicationWithSQLAlchemy
from quantdsl.domain.model.call_dependencies import CallDependencies
from quantdsl.domain.model.call_dependents import CallDependents
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_result import CallResult
from quantdsl.domain.model.contract_specification import ContractSpecification
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.dependency_graph import DependencyGraph, register_dependency_graph
from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.simulated_price import SimulatedPrice, make_simulated_price_id, register_simulated_price
from quantdsl.domain.services.uuids import create_uuid4
from quantdsl.domain.services.fixing_dates import list_fixing_dates
from quantdsl.domain.services.market_names import list_market_names
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME


# Specification. Calibration. Simulation. Evaluation.

class ApplicationTestCase(unittest.TestCase):

    def setUp(self):
        assert_event_handlers_empty()
        super(ApplicationTestCase, self).setUp()
        self.app = get_app()

    def tearDown(self):
        super(ApplicationTestCase, self).tearDown()
        self.app.close()
        assert_event_handlers_empty()


class TestEventSourcedRepos(ApplicationTestCase):

    def test_register_market_calibration(self):
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        calibration_params = {'param1': 10, 'param2': 20}

        market_calibration = self.app.register_market_calibration(price_process_name, calibration_params)

        assert isinstance(market_calibration, MarketCalibration)
        assert market_calibration.id
        market_calibration = self.app.market_calibration_repo[market_calibration.id]
        assert isinstance(market_calibration, MarketCalibration)
        self.assertEqual(market_calibration.price_process_name, DEFAULT_PRICE_PROCESS_NAME)
        self.assertEqual(market_calibration.calibration_params['param1'], 10)
        self.assertEqual(market_calibration.calibration_params['param2'], 20)

    def test_register_contract_specification(self):
        contract_spec = self.app.register_contract_specification('1 + 1')
        self.assertIsInstance(contract_spec, ContractSpecification)
        self.assertIsInstance(contract_spec.id, six.string_types)
        contract_spec = self.app.contract_specification_repo[contract_spec.id]
        assert isinstance(contract_spec, ContractSpecification)
        self.assertEqual(contract_spec.specification, '1 + 1')

    def test_register_dependency_graph(self):
        contract_specification_id = create_uuid4()
        dependency_graph = register_dependency_graph(contract_specification_id)
        self.assertIsInstance(dependency_graph, DependencyGraph)
        assert isinstance(dependency_graph, DependencyGraph)
        self.assertEqual(dependency_graph.contract_specification_id, contract_specification_id)

    def test_register_call_requirements(self):
        call_id = create_uuid4()

        self.assertRaises(KeyError, self.app.call_requirement_repo.__getitem__, call_id)

        dsl_source = '1 + 1'
        effective_present_time = datetime.datetime(2015, 9, 7, 0, 0, 0)

        self.app.register_call_requirement(call_id=call_id, dsl_source=dsl_source,
                                      effective_present_time=effective_present_time)

        call_requirement = self.app.call_requirement_repo[call_id]
        assert isinstance(call_requirement, CallRequirement)
        self.assertEqual(call_requirement.dsl_source, dsl_source)
        self.assertEqual(call_requirement.effective_present_time, effective_present_time)

    def test_register_call_dependencies(self):
        call_id = create_uuid4()

        self.assertRaises(KeyError, self.app.call_dependencies_repo.__getitem__, call_id)

        dependencies = ['123', '456']

        self.app.register_call_dependencies(call_id=call_id, dependencies=dependencies)

        call_dependencies = self.app.call_dependencies_repo[call_id]
        assert isinstance(call_dependencies, CallDependencies)
        self.assertEqual(call_dependencies.dependencies, dependencies)

    def test_register_call_dependents(self):
        call_id = create_uuid4()

        self.assertRaises(KeyError, self.app.call_dependents_repo.__getitem__, call_id)

        dependents = ['123', '456']

        self.app.register_call_dependents(call_id=call_id, dependents=dependents)

        call_dependents = self.app.call_dependents_repo[call_id]
        assert isinstance(call_dependents, CallDependents)
        self.assertEqual(call_dependents.dependents, dependents)

    def test_register_call_result(self):
        call_id = create_uuid4()

        self.assertRaises(KeyError, self.app.call_result_repo.__getitem__, call_id)

        self.app.register_call_result(call_id=call_id, result_value=123)

        call_result = self.app.call_result_repo[call_id]
        assert isinstance(call_result, CallResult)
        self.assertEqual(call_result.result_value, 123)

    def test_register_simulated_price(self):

        price_time = datetime.datetime(2011, 1, 1)
        price_value = numpy.array([1.1, 1.2, 1.367345987359734598734598723459872345987235698237459862345])
        simulation_id = create_uuid4()
        self.assertRaises(KeyError, self.app.simulated_price_repo.__getitem__, simulation_id)

        price = register_simulated_price(simulation_id, '#1', price_time, price_value)

        assert isinstance(price, SimulatedPrice), price
        assert price.id
        price = self.app.simulated_price_repo[make_simulated_price_id(simulation_id, '#1', price_time)]
        assert isinstance(price, SimulatedPrice)
        numpy.testing.assert_equal(price.value, price_value)

    def test_register_contract_valuation(self):
        v = self.app.register_contract_valuation(dependency_graph_id='123456')
        self.assertIsInstance(v, ContractValuation)
        v = self.app.contract_valuation_repo[v.id]
        self.assertIsInstance(v, ContractValuation)


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

        specification = """1 + 2"""

        # Register the specification.
        contract_specification = self.app.register_contract_specification(specification=specification)

        # C.
        market_calibration = self.get_calibration_fixture()
        market_simulation = self.get_market_simulation_fixture(market_calibration)
        self.app.generate_contract_valuation(contract_specification.id, market_simulation)

        # Check the result.
        self.assertIn(contract_specification.id, self.app.call_result_repo)
        call_result = self.app.call_result_repo[contract_specification.id]
        assert isinstance(call_result, CallResult)
        self.assertEqual(call_result.result_value, 3)

    def test_generate_valuation_fibonacci_numbers(self):

        # Define the contract.
        contract_specification = self.app.register_contract_specification(specification="""
def fib(n): return fib(n-1) + fib(n-2) if n > 2 else n
fib(6)
""")
        market_simulation = self.create_market_simulation_from_contract_specification(contract_specification)

        # Generate the contract valuation.
        self.app.generate_contract_valuation(contract_specification.id, market_simulation)

        # Check the result.
        self.assertIn(contract_specification.id, self.app.call_result_repo)
        call_result = self.app.call_result_repo[contract_specification.id]
        assert isinstance(call_result, CallResult)
        self.assertEqual(call_result.result_value, 13)

    def test_generate_valuation_american_option(self):

        # Define the contract.
        contract_specification = self.app.register_contract_specification(specification="""
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

American(Date('2011-01-02'), Date('2011-01-03'), 9, Market('#1'))
""")

        # Generate the market simulation.
        market_simulation = self.create_market_simulation_from_contract_specification(contract_specification)

        # Generate the contract valuation.
        self.app.generate_contract_valuation(contract_specification.id, market_simulation)

        # Check the result.
        self.assertIn(contract_specification.id, self.app.call_result_repo)
        call_result = self.app.call_result_repo[contract_specification.id]
        assert isinstance(call_result, CallResult)
        self.assertAlmostEqual(call_result.result_value.mean(), 1.0, places=1)

    def create_market_simulation_from_contract_specification(self, contract_specification):
        market_calibration = self.get_calibration_fixture()
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
        )
        return market_simulation

    def test_generate_valuation_swing_option(self):

        # Define the contract.
        contract_specification = self.app.register_contract_specification(specification="""
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-01-10'), Market('#1'), 5)
""")

        # Generate the market simulation.
        market_simulation = self.create_market_simulation_from_contract_specification(contract_specification)

        # Generate the contract valuation.
        self.app.generate_contract_valuation(contract_specification.id, market_simulation)

        # Check the result.
        self.assertIn(contract_specification.id, self.app.call_result_repo)
        call_result = self.app.call_result_repo[contract_specification.id]
        assert isinstance(call_result, CallResult)
        self.assertAlmostEqual(call_result.result_value.mean(), 50, places=1)

    def test_generate_valuation_power_plant_option(self):

        # Define the contract.
        contract_specification = self.app.register_contract_specification(specification="""
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

def PowerPlant(start_date, end_date, underlying, time_since_off):
    if (start_date < end_date):
        Choice(
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, 0)
                + ProfitFromRunning(start_date, underlying, time_since_off),
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, NextTime(time_since_off))
        )
    else:
        return 0

PowerPlant(Date('2012-01-01'), Date('2012-01-06'), Market('#1'), 2)

""")

        # Generate the market simulation.
        market_simulation = self.create_market_simulation_from_contract_specification(contract_specification)

        # Generate the contract valuation.
        self.app.generate_contract_valuation(contract_specification.id, market_simulation)

        # Check the result.
        self.assertIn(contract_specification.id, self.app.call_result_repo)
        call_result = self.app.call_result_repo[contract_specification.id]
        assert isinstance(call_result, CallResult)
        self.assertAlmostEqual(call_result.result_value.mean(), 48, places=2)

    #
    # Fixture methods...

    def get_calibration_fixture(self):
        """
        :rtype: MarketSimulation
        """
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        calibration_params = {
            '#1-LAST-PRICE': 10,
            '#2-LAST-PRICE': 20,
            '#1-ACTUAL-HISTORICAL-VOLATILITY': .10,
            '#2-ACTUAL-HISTORICAL-VOLATILITY': 20,
            '#1-#2-CORRELATION': 0.5,
        }
        return self.app.register_market_calibration(price_process_name, calibration_params)

    def get_market_simulation_fixture(self, market_calibration):
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
        )
        return market_simulation


def get_app():
    return QuantDslApplicationWithSQLAlchemy(db_uri='sqlite:///:memory:')
