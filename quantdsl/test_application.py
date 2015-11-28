import datetime
import unittest

import numpy

from quantdsl import utc
from quantdsl.application.with_sqlalchemy import QuantDslApplicationWithSQLAlchemy
from quantdsl.domain.model.call_dependencies import CallDependencies
from quantdsl.domain.model.call_dependents import CallDependents
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_result import CallResult
from quantdsl.domain.model.contract_specification import ContractSpecification
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.dependency_graph import DependencyGraph
from quantdsl.domain.model.leaf_calls import LeafCalls
from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.simulated_price import SimulatedPrice, make_simulated_price_id
from quantdsl.domain.services.create_uuid4 import create_uuid4
from quantdsl.domain.services.fixing_times import fixing_times_from_call_order
from quantdsl.domain.services.market_names import market_names_from_contract_specification
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME


class TestApplication(unittest.TestCase):

    NUMBER_MARKETS = 2
    NUMBER_DAYS = 5
    PATH_COUNT = 2000

    def test_register_contract_specification(self):
        app = self.get_app()
        specification = '1 + 1'

        contract_spec = app.register_contract_specification(specification)

        assert isinstance(contract_spec, ContractSpecification), contract_spec
        assert contract_spec.id
        contract_spec = app.contract_specification_repo[contract_spec.id]
        assert isinstance(contract_spec, ContractSpecification)
        self.assertEqual(contract_spec.specification, specification)

    def test_register_dependency_graph(self):
        app = self.get_app()
        dependency_graph = app.register_dependency_graph()
        assert isinstance(dependency_graph, DependencyGraph)

    def test_register_call_requirements(self):
        app = self.get_app()
        call_id = create_uuid4()

        self.assertRaises(KeyError, app.call_requirement_repo.__getitem__, call_id)

        dsl_source = '1 + 1'
        effective_present_time = datetime.datetime(2015, 9, 7, 0, 0, 0)

        app.register_call_requirement(call_id=call_id, dsl_source=dsl_source,
                                      effective_present_time=effective_present_time)

        call_requirement = app.call_requirement_repo[call_id]
        assert isinstance(call_requirement, CallRequirement)
        self.assertEqual(call_requirement.dsl_source, dsl_source)
        self.assertEqual(call_requirement.effective_present_time, effective_present_time)

    def test_register_call_dependencies(self):
        app = self.get_app()
        call_id = create_uuid4()

        self.assertRaises(KeyError, app.call_dependencies_repo.__getitem__, call_id)

        dependencies = ['123', '456']

        app.register_call_dependencies(call_id=call_id, dependencies=dependencies)

        call_dependencies = app.call_dependencies_repo[call_id]
        assert isinstance(call_dependencies, CallDependencies)
        self.assertEqual(call_dependencies.dependencies, dependencies)

    def test_register_call_dependents(self):
        app = self.get_app()
        call_id = create_uuid4()

        self.assertRaises(KeyError, app.call_dependents_repo.__getitem__, call_id)

        dependents = ['123', '456']

        app.register_call_dependents(call_id=call_id, dependents=dependents)

        call_dependents = app.call_dependents_repo[call_id]
        assert isinstance(call_dependents, CallDependents)
        self.assertEqual(call_dependents.dependents, dependents)

    def test_register_leaf_calls(self):
        app = self.get_app()
        dependency_graph_id = create_uuid4()

        self.assertRaises(KeyError, app.leaf_calls_repo.__getitem__, dependency_graph_id)

        leaf_call_ids = ['123', '456']

        app.register_leaf_calls(dependency_graph_id=dependency_graph_id, call_ids=leaf_call_ids)

        leaf_calls = app.leaf_calls_repo[dependency_graph_id]
        assert isinstance(leaf_calls, LeafCalls)
        self.assertEqual(leaf_calls.call_ids, leaf_call_ids)

    def test_register_call_result(self):
        app = self.get_app()
        call_id = create_uuid4()

        self.assertRaises(KeyError, app.call_result_repo.__getitem__, call_id)

        app.register_call_result(call_id=call_id, result_value=123)

        call_result = app.call_result_repo[call_id]
        assert isinstance(call_result, CallResult)
        self.assertEqual(call_result.result_value, 123)

    def test_generate_dependency_graph_no_root_dependencies(self):
        app = self.get_app()
        contract_specification = app.register_contract_specification(specification="""
1 + 1
""")
        dependency_graph = app.generate_dependency_graph(contract_specification=contract_specification)

        root_dependencies = app.call_dependencies_repo[dependency_graph.id]
        assert isinstance(root_dependencies, CallDependencies)
        self.assertEqual(len(root_dependencies.dependencies), 0)

    def test_generate_dependency_graph_with_function_call(self):
        app = self.get_app()
        contract_specification = app.register_contract_specification(specification="""
def double(x):
    return x * 2

double(1 + 1)
""")
        dependency_graph = app.generate_dependency_graph(contract_specification=contract_specification)

        root_dependencies = app.call_dependencies_repo[dependency_graph.id]
        assert isinstance(root_dependencies, CallDependencies)
        self.assertEqual(len(root_dependencies.dependencies), 1)

        root_dependency = root_dependencies.dependencies[0]
        call_dependencies = app.call_dependencies_repo[root_dependency]
        self.assertEqual(len(call_dependencies.dependencies), 0)

        dependency = app.call_dependents_repo[root_dependency]
        assert isinstance(dependency, CallDependents)
        self.assertEqual(len(dependency.dependents), 1)

        self.assertEqual(dependency.dependents[0], dependency_graph.id)

    def test_generate_dependency_graph_recursive_indefinite(self):
        app = self.get_app()
        contract_specification = app.register_contract_specification(specification="""
def inc(x):
    if x < 10:
        return inc(x+1)
    else:
        return 100

inc(1 + 2)
""")

        dependency_graph = app.generate_dependency_graph(contract_specification=contract_specification)
        call_dependencies = app.call_dependencies_repo[dependency_graph.id]
        self.assertEqual(len(call_dependencies.dependencies), 1)
        dependency_id = call_dependencies.dependencies[0]
        dependents = app.call_dependents_repo[dependency_id].dependents
        self.assertEqual(len(dependents), 2)
        self.assertIn(dependency_graph.id, dependents)
        # A circular dependency...
        self.assertIn(dependency_id, dependents)


    # def test_compile_dependency_graph(self, contract_specification):
    #     app.compute_dependency_graph(contract_specification)
    #
    # def test_compute_contract_value(self, contract_specification, market_simulation, contract_state):
    #     app.compute_contract_value(dependency_graph, market_simulation)

    def test_register_market_calibration(self):
        app = self.get_app()

        market_calibration = self.get_calibration_fixture(app)

        assert isinstance(market_calibration, MarketCalibration)
        assert market_calibration.id
        market_calibration = app.market_calibration_repo[market_calibration.id]
        assert isinstance(market_calibration, MarketCalibration)
        self.assertEqual(market_calibration.calibration_params['#1-LAST-PRICE'], 10)
        self.assertEqual(market_calibration.calibration_params['#2-LAST-PRICE'], 20)
        self.assertEqual(market_calibration.calibration_params['#1-ACTUAL-HISTORICAL-VOLATILITY'], 0.1)
        self.assertEqual(market_calibration.calibration_params['#2-ACTUAL-HISTORICAL-VOLATILITY'], 0.2)
        self.assertEqual(market_calibration.calibration_params['#1-#2-CORRELATION'], 0.5)

    def test_register_market_simulation(self):
        app = self.get_app()

        market_calibration = self.get_calibration_fixture(app)

        market_simulation = self.get_market_simulation_fixture(app, market_calibration)

        assert isinstance(market_simulation, MarketSimulation)
        assert market_simulation.id
        market_simulation = app.market_simulation_repo[market_simulation.id]
        assert isinstance(market_simulation, MarketSimulation)
        self.assertEqual(market_simulation.market_calibration_id, market_calibration.id)
        self.assertEqual(market_simulation.price_process_name, DEFAULT_PRICE_PROCESS_NAME)
        self.assertEqual(market_simulation.market_names, ['#1', '#2'])
        self.assertEqual(market_simulation.fixing_times, [datetime.date(2011, 1, 2), datetime.date(2011, 1, 3)])
        self.assertEqual(market_simulation.observation_time, datetime.date(2011, 1, 1))
        self.assertEqual(market_simulation.path_count, self.PATH_COUNT)

    def test_register_simulated_price(self):
        app = self.get_app()

        price_time = datetime.datetime(2011, 1, 1)
        price_value = numpy.array([1.1, 1.2, 1.367345987359734598734598723459872345987235698237459862345])
        simulation_id = create_uuid4()
        self.assertRaises(KeyError, app.simulated_price_repo.__getitem__, simulation_id)

        price = app.register_simulated_price(simulation_id, '#1', price_time, price_value)

        assert isinstance(price, SimulatedPrice), price
        assert price.id
        price = app.simulated_price_repo[make_simulated_price_id(simulation_id, '#1', price_time)]
        assert isinstance(price, SimulatedPrice)
        numpy.testing.assert_equal(price.value, price_value)

    def test_simulate_future_prices(self):
        app = self.get_app()
        market_calibration = self.get_calibration_fixture(app)
        market_simulation = self.get_market_simulation_fixture(app, market_calibration)

        simulated_prices = app.simulate_future_prices(market_calibration, market_simulation)

        market_name, fixing_time, price_value = list(simulated_prices)[2]
        self.assertEqual(market_name, '#1')
        self.assertEqual(fixing_time, datetime.date(2011, 1, 3))
        self.assertAlmostEqual(price_value.mean(), 10, places=2)

    def test_generate_simulated_prices(self):
        app = self.get_app()
        market_calibration = self.get_calibration_fixture(app)
        market_simulation = self.get_market_simulation_fixture(app, market_calibration)

        app.generate_simulated_prices(market_calibration, market_simulation)

        simulated_price_id = make_simulated_price_id(market_simulation.id, '#1', datetime.date(2011, 1, 3))
        price = app.simulated_price_repo[simulated_price_id]
        assert isinstance(price, SimulatedPrice)
        self.assertAlmostEqual(price.value.mean(), 10, places=2)

    def test_generate_market_simulation(self):
        app = self.get_app()
        market_calibration = self.get_calibration_fixture(app)

        market_names=['#%d' % (i+1) for i in range(self.NUMBER_MARKETS)]
        date_range=[datetime.date(2011, 1, 1) + datetime.timedelta(days=i) for i in range(self.NUMBER_DAYS)]
        market_simulation = app.generate_market_simulation(
            market_calibration,
            price_process_name=DEFAULT_PRICE_PROCESS_NAME,
            market_names=market_names,
            fixing_times=date_range[1:],
            observation_time=date_range[0],
            path_count=self.PATH_COUNT,
        )

        simulated_price_id = make_simulated_price_id(market_simulation.id, '#1', datetime.datetime(2011, 1, 3))
        price = app.simulated_price_repo[simulated_price_id]
        assert isinstance(price, SimulatedPrice)
        self.assertAlmostEqual(price.value.mean(), 10, places=2)

    def test_register_contract_valuation(self):
        app = self.get_app()
        v = app.register_contract_valuation(dependency_graph_id='123456')
        assert isinstance(v, ContractValuation)
        app.contract_valuation_repo[v.id]
        # self.assertIn(v.id, app.contract_valuation_repo)

    def test_generate_valuation_simple_addition(self):
        app = self.get_app()

        # Define the contract.
        contract_specification = app.register_contract_specification(specification="""1 + 2""")

        # Generate the dependency graph.
        dependency_graph = app.generate_dependency_graph(contract_specification=contract_specification)

        call_order = app.generate_call_order(dependency_graph)

        market_calibration = self.get_calibration_fixture(app)
        market_simulation = self.get_market_simulation_fixture(app, market_calibration)

        # Generate the contract valuation.
        app.generate_contract_valuation(dependency_graph, market_simulation, call_order)

        # Check the result.
        self.assertIn(dependency_graph.id, app.call_result_repo)
        call_result = app.call_result_repo[dependency_graph.id]
        assert isinstance(call_result, CallResult)
        self.assertEqual(call_result.result_value, 3)

    def test_generate_valuation_fibonacci_numbers(self):
        app = self.get_app()

        # Define the contract.
        contract_specification = app.register_contract_specification(specification="""
def fib(n): return fib(n-1) + fib(n-2) if n > 2 else n
fib(6)
""")
        # Generate the dependency graph.
        dependency_graph = app.generate_dependency_graph(contract_specification=contract_specification)

        call_order = app.generate_call_order(dependency_graph)

        market_calibration = self.get_calibration_fixture(app)
        market_simulation = self.get_market_simulation_fixture(app, market_calibration)

        # Generate the contract valuation.
        app.generate_contract_valuation(dependency_graph, market_simulation, call_order)

        # Check the result.
        self.assertIn(dependency_graph.id, app.call_result_repo)
        call_result = app.call_result_repo[dependency_graph.id]
        assert isinstance(call_result, CallResult)
        self.assertEqual(call_result.result_value, 13)

    def test_generate_valuation_american_option(self):
        app = self.get_app()

        # Define the contract.
        contract_specification = app.register_contract_specification(specification="""
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

American(Date('2011-01-01'), Date('2011-01-03'), 9, Market('#1'))
""")

        # Generate the dependency graph from the contract specification.
        dependency_graph = app.generate_dependency_graph(contract_specification=contract_specification)

        # Generate call order from the dependency graph.
        call_order = list(app.generate_call_order(dependency_graph))

        # Generate the market simulation.
        market_calibration = self.get_calibration_fixture(app)
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        market_names = market_names_from_contract_specification(contract_specification)
        fixing_times = fixing_times_from_call_order(call_order, app.call_requirement_repo)
        observation_time = datetime.date(2010, 1, 1)
        path_count = self.PATH_COUNT

        market_simulation = app.generate_market_simulation(
            market_calibration=market_calibration,
            price_process_name=price_process_name,
            market_names=market_names,
            fixing_times=fixing_times,
            observation_time=observation_time,
            path_count=path_count,
        )

        # Generate the contract valuation.
        app.generate_contract_valuation(dependency_graph, market_simulation, call_order)

        # Check the result.
        self.assertIn(dependency_graph.id, app.call_result_repo)
        call_result = app.call_result_repo[dependency_graph.id]
        assert isinstance(call_result, CallResult)
        self.assertAlmostEqual(call_result.result_value.mean(), 1, places=2)

    def test_generate_valuation_swing_option(self):
        app = self.get_app()

        # Define the contract.
        contract_specification = app.register_contract_specification(specification="""
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-01-10'), Market('#1'), 10)
""")

        # Generate the dependency graph from the contract specification.
        dependency_graph = app.generate_dependency_graph(contract_specification=contract_specification)

        # Generate call order from the dependency graph.
        call_order = list(app.generate_call_order(dependency_graph))

        # self.assertEqual(len(call_order), 10)

        # Generate the market simulation.
        market_calibration = self.get_calibration_fixture(app)
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        market_names = market_names_from_contract_specification(contract_specification)
        fixing_times = fixing_times_from_call_order(call_order, app.call_requirement_repo)
        observation_time = datetime.date(2010, 1, 1)
        path_count = 20000

        market_simulation = app.generate_market_simulation(
            market_calibration=market_calibration,
            price_process_name=price_process_name,
            market_names=market_names,
            fixing_times=fixing_times,
            observation_time=observation_time,
            path_count=path_count,
        )

        # Generate the contract valuation.
        app.generate_contract_valuation(dependency_graph, market_simulation, call_order)

        # Check the result.
        self.assertIn(dependency_graph.id, app.call_result_repo)
        call_result = app.call_result_repo[dependency_graph.id]
        assert isinstance(call_result, CallResult)
        self.assertAlmostEqual(call_result.result_value.mean(), 20, places=2)

    def test_generate_valuation_power_plant_option(self):
        app = self.get_app()

        # Define the contract.
        contract_specification = app.register_contract_specification(specification="""
@nostub
def NextTime(time_since_off):
    if time_since_off == 2:
        return 2
    else:
        return time_since_off + 1

@nostub
def ProfitFromRunning(start_date, underlying, time_since_off):
    if time_since_off == 0:
        return Fixing(start_date, underlying) - 9
    elif time_since_off == 1:
        return 0.9 * Fixing(start_date, underlying) - 9
    else:
        return 0.8 * Fixing(start_date, underlying) - 9

def PowerPlant(start_date, end_date, underlying, time_since_off):
    if (start_date < end_date):
        Choice(
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, 0)
                + ProfitFromRunning(start_date, underlying, time_since_off),
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, NextTime(time_since_off))
        )
    else:
        return 0

PowerPlant(Date('2012-02-01'), Date('2012-02-04'), Market('#1'), 2)
""")

        # Generate the dependency graph from the contract specification.
        dependency_graph = app.generate_dependency_graph(contract_specification=contract_specification)

        # Generate call order from the dependency graph.
        call_order = list(app.generate_call_order(dependency_graph))

        # self.assertEqual(len(call_order), 10)

        # Generate the market simulation.
        market_calibration = self.get_calibration_fixture(app)
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        market_names = market_names_from_contract_specification(contract_specification)
        fixing_times = fixing_times_from_call_order(call_order, app.call_requirement_repo)
        observation_time = datetime.date(2010, 1, 1)
        path_count = 20000

        market_simulation = app.generate_market_simulation(
            market_calibration=market_calibration,
            price_process_name=price_process_name,
            market_names=market_names,
            fixing_times=fixing_times,
            observation_time=observation_time,
            path_count=path_count,
        )

        # Generate the contract valuation.
        app.generate_contract_valuation(dependency_graph, market_simulation, call_order)

        # Check the result.
        self.assertIn(dependency_graph.id, app.call_result_repo)
        call_result = app.call_result_repo[dependency_graph.id]
        assert isinstance(call_result, CallResult)
        self.assertAlmostEqual(call_result.result_value.mean(), 20, places=2)

    #
    # Fixture methods...

    def get_calibration_fixture(self, app):
        """
        :rtype: MarketSimulation
        """
        calibration_params = {
            '#1-LAST-PRICE': 10,
            '#2-LAST-PRICE': 20,
            '#1-ACTUAL-HISTORICAL-VOLATILITY': 1.0,
            '#2-ACTUAL-HISTORICAL-VOLATILITY': 0.2,
            '#1-#2-CORRELATION': 0.5,
        }
        return app.register_market_calibration(calibration_params=calibration_params)

    def get_market_simulation_fixture(self, app, market_calibration):
        market_names = ['#%d' % (i+1) for i in range(self.NUMBER_MARKETS)]
        date_range = [datetime.date(2011, 1, 1) + datetime.timedelta(days=i) for i in range(self.NUMBER_DAYS)]
        fixing_times = date_range[1:]
        observation_time = date_range[0]
        path_count = self.PATH_COUNT
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        market_simulation = app.register_market_simulation(
            market_calibration_id=market_calibration.id,
            price_process_name=price_process_name,
            market_names=market_names,
            fixing_times=fixing_times,
            observation_time=observation_time,
            path_count=path_count,
        )
        return market_simulation

    def get_app(self):
        return QuantDslApplicationWithSQLAlchemy(db_uri='sqlite:///:memory:')
