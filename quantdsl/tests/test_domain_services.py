import datetime
import unittest

import scipy
import six
from eventsourcing.infrastructure.event_store import EventStore
from eventsourcing.infrastructure.persistence_subscriber import PersistenceSubscriber
from eventsourcing.infrastructure.stored_events.python_objects_stored_events import PythonObjectsStoredEventRepository
from mock import MagicMock, Mock, patch

from quantdsl.domain.model.call_dependencies import CallDependencies, CallDependenciesRepository
from quantdsl.domain.model.call_dependents import CallDependents, CallDependentsRepository
from quantdsl.domain.model.call_link import CallLink, CallLinkRepository
from quantdsl.domain.model.call_requirement import CallRequirement, CallRequirementRepository
from quantdsl.domain.model.call_result import CallResult, CallResultRepository
from quantdsl.domain.model.contract_specification import ContractSpecification, register_contract_specification
from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.contract_valuations import get_dependency_results
from quantdsl.domain.services.dependency_graphs import generate_dependency_graph, generate_execution_order
from quantdsl.domain.services.price_processes import get_price_process
from quantdsl.domain.services.simulated_prices import generate_simulated_prices, identify_simulation_requirements, \
    simulate_future_prices
from quantdsl.domain.services.uuids import create_uuid4
from quantdsl.exceptions import DslError
from quantdsl.infrastructure.event_sourced_repos.call_dependencies_repo import CallDependenciesRepo
from quantdsl.infrastructure.event_sourced_repos.call_dependents_repo import CallDependentsRepo
from quantdsl.infrastructure.event_sourced_repos.call_leafs_repo import CallLeafsRepo
from quantdsl.infrastructure.event_sourced_repos.call_requirement_repo import CallRequirementRepo
from quantdsl.infrastructure.event_sourced_repos.perturbation_dependencies_repo import PerturbationDependenciesRepo
from quantdsl.priceprocess.blackscholes import BlackScholesPriceProcess
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME


class TestUUIDs(unittest.TestCase):
    def test_create_uuid4(self):
        id = create_uuid4()
        self.assertIsInstance(id, six.string_types)


class TestDependencyGraph(unittest.TestCase):
    def setUp(self):
        self.es = EventStore(stored_event_repo=PythonObjectsStoredEventRepository())
        self.ps = PersistenceSubscriber(self.es)
        self.call_dependencies_repo = CallDependenciesRepo(self.es)
        self.call_dependents_repo = CallDependentsRepo(self.es)
        self.call_leafs_repo = CallLeafsRepo(self.es)
        self.call_requirement_repo = CallRequirementRepo(self.es)

    def tearDown(self):
        self.ps.close()

    def test_generate_dependency_graph_with_function_call(self):
        contract_specification = register_contract_specification(source_code="""
def double(x):
    return x * 2

double(1 + 1)
""")

        generate_dependency_graph(
            contract_specification=contract_specification,
            call_dependencies_repo=self.call_dependencies_repo,
            call_dependents_repo=self.call_dependents_repo,
            call_leafs_repo=self.call_leafs_repo,
            call_requirement_repo=self.call_requirement_repo,
        )

        root_dependencies = self.call_dependencies_repo[contract_specification.id]
        assert isinstance(root_dependencies, CallDependencies)
        self.assertEqual(len(root_dependencies.dependencies), 1)

        root_dependency = root_dependencies.dependencies[0]
        call_dependencies = self.call_dependencies_repo[root_dependency]
        self.assertEqual(len(call_dependencies.dependencies), 0)

        dependency = self.call_dependents_repo[root_dependency]
        assert isinstance(dependency, CallDependents)
        self.assertEqual(len(dependency.dependents), 1)

        self.assertEqual(dependency.dependents[0], contract_specification.id)

    #     def test_generate_dependency_graph_recursive_functional_call(self):
    #         contract_specification = self.app.register_contract_specification(specification="""
    # def inc(x):
    #     if x < 10:
    #         return inc(x+1) + inc(x+2)
    #     else:
    #         return 100
    #
    # inc(1 + 2)
    # """)
    #
    #         dependency_graph = self.app.generate_dependency_graph(contract_specification=contract_specification)
    #
    #         call_dependencies = self.app.call_dependencies_repo[dependency_graph.id]
    #         self.assertEqual(len(call_dependencies.requirements), 1)
    #         dependency_id = call_dependencies.requirements[0]
    #         dependents = self.app.call_dependents_repo[dependency_id].dependents
    #         self.assertEqual(len(dependents), 1)
    #         self.assertIn(dependency_graph.id, dependents)
    #         # A circular dependency...
    #         self.assertIn(dependency_id, dependents)



    def test_generate_execution_order(self):
        # Dependencies:
        # 1 -> 2
        # 1 -> 3
        # 3 -> 2

        # Therefore dependents:
        # 1 = []
        # 2 = [1, 3]
        # 3 = [1]

        # 2 depends on nothing, so 2 is a leaf.
        # 1 depends on 3 and 2, so 1 is not next.
        # 3 depends only on 2, so is next.
        # Therefore 1 is last.
        # Hence evaluation order: 2, 3, 1

        call_dependencies_repo = MagicMock(spec=CallDependenciesRepository,
                                           __getitem__=lambda self, x: {
                                               1: [2, 3],
                                               2: [],
                                               3: [2]
                                           }[x])
        call_dependents_repo = MagicMock(spec=CallDependentsRepository,
                                         __getitem__=lambda self, x: {
                                             1: [],
                                             2: [1, 3],
                                             3: [1]
                                         }[x])

        leaf_call_ids = [2]
        execution_order_gen = generate_execution_order(leaf_call_ids, call_dependents_repo, call_dependencies_repo)
        execution_order = list(execution_order_gen)
        self.assertEqual(execution_order, [2, 3, 1])

    def test_get_dependency_values(self):
        call_dependencies_repo = MagicMock(spec=CallDependenciesRepository,
                                           __getitem__=lambda self, x: {
                                               '1': CallDependencies(dependencies=['2', '3'], entity_id=123,
                                                                     entity_version=0, timestamp=1),
                                           }[x])
        call_result_repo = MagicMock(spec=CallResultRepository,
                                     __getitem__=lambda self, x: {
                                         'valuation2': Mock(spec=CallResult, result_value=12, perturbed_values={}),
                                         'valuation3': Mock(spec=CallResult, result_value=13, perturbed_values={}),
                                     }[x])
        values = get_dependency_results('valuation', '1', call_dependencies_repo, call_result_repo)
        self.assertEqual(values, {'2': (12, {}), '3': (13, {})})


class TestCallLinks(unittest.TestCase):
    def test_regenerate_execution_order(self):
        contract_specification = Mock(spec=ContractSpecification, id=1)
        call_link_repo = MagicMock(spec=CallLinkRepository,
                                   __getitem__=lambda self, x: {
                                       1: Mock(spec=CallLink, call_id=2),
                                       2: Mock(spec=CallLink, call_id=3),
                                       3: Mock(spec=CallLink, call_id=1),
                                   }[x])
        order = regenerate_execution_order(contract_specification.id, call_link_repo)
        order = list(order)
        self.assertEqual(order, [2, 3, 1])


# Todo: Fix up this test to check the new behaviour: setting the market requirements.
class TestListMarketNamesAndFixingDates(unittest.TestCase):
    def test_list_market_names_and_fixing_dates(self):
        contract_specification = Mock(spec=ContractSpecification, id=1)
        call_requirement_repo = MagicMock(spec=CallRequirementRepository,
                                          __getitem__=lambda self, x: {
                                              1: Mock(spec=CallRequirement,
                                                      dsl_source="Fixing('2011-01-01', Market('1'))",
                                                      effective_present_time=datetime.datetime(2011, 1, 1),
                                                      _dsl_expr=None),
                                              2: Mock(spec=CallRequirement,
                                                      dsl_source="Fixing('2012-02-02', Market('2'))",
                                                      effective_present_time=datetime.datetime(2011, 2, 2),
                                                      _dsl_expr=None),
                                              3: Mock(spec=CallRequirement,
                                                      dsl_source="Fixing('2013-03-03', Market('3'))",
                                                      effective_present_time=datetime.datetime(2011, 3, 3),
                                                      _dsl_expr=None),
                                          }[x])
        call_link_repo = MagicMock(spec=CallLinkRepository,
                                   __getitem__=lambda self, x: {
                                       1: Mock(spec=CallLink, call_id=2),
                                       2: Mock(spec=CallLink, call_id=3),
                                       3: Mock(spec=CallLink, call_id=1),
                                   }[x])
        call_dependencies_repo = MagicMock(spec=CallDependenciesRepo,
                                           __getitem__=lambda self, x: {
                                               1: Mock(spec=CallDependencies, dependencies=[]),
                                               2: Mock(spec=CallDependencies, dependencies=[]),
                                               3: Mock(spec=CallDependencies, dependencies=[]),
                                           }[x])
        market_dependencies_repo = MagicMock(spec=PerturbationDependenciesRepo)

        observation_date = datetime.datetime(2011, 1, 1)

        requirements = set()
        identify_simulation_requirements(contract_specification.id, call_requirement_repo, call_link_repo,
                                         call_dependencies_repo, market_dependencies_repo, observation_date,
                                         requirements)

        self.assertEqual(requirements, {
            ('1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1)),
            ('2', datetime.datetime(2012, 2, 2), datetime.datetime(2012, 2, 2)),
            ('3', datetime.datetime(2013, 3, 3), datetime.datetime(2013, 3, 3)),
        })


class TestSimulatedPrices(unittest.TestCase):
    @patch('quantdsl.domain.services.simulated_prices.register_simulated_price')
    @patch('quantdsl.domain.services.simulated_prices.simulate_future_prices', return_value=[
        ('#1', datetime.datetime(2011, 1, 1), datetime.datetime(2011, 1, 1), 10),
        ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2), 10),
        ('#1', datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2), 10),
    ])
    def test_generate_simulated_prices(self, simulate_future_prices, register_simuated_price):
        market_calibration = Mock(spec=MarketCalibration)
        market_simulation = Mock(spec=MarketSimulation)
        prices = generate_simulated_prices(market_simulation=market_simulation, market_calibration=market_calibration)
        prices = list(prices)
        self.assertEqual(register_simuated_price.call_count, len(simulate_future_prices.return_value))

    @patch('quantdsl.domain.services.simulated_prices.get_price_process', new=lambda name: Mock(
        spec=BlackScholesPriceProcess,
        simulate_future_prices=lambda observation_date, requirements, path_count, calibration_params: [
            ('#1', datetime.date(2011, 1, 1), scipy.array([10.])),
            ('#1', datetime.date(2011, 1, 2), scipy.array([10.])),
        ]
    ))
    def test_simulate_future_prices(self):
        ms = Mock(spec=MarketSimulation)
        mc = Mock(spec=MarketCalibration)
        prices = simulate_future_prices(market_simulation=ms, market_calibration=mc)
        self.assertEqual(list(prices), [
            ('#1', datetime.date(2011, 1, 1), scipy.array([10.])),
            ('#1', datetime.date(2011, 1, 2), scipy.array([10.])),
        ])


class TestPriceProcesses(unittest.TestCase):
    def test_get_price_process(self):
        price_process = get_price_process(DEFAULT_PRICE_PROCESS_NAME)
        self.assertIsInstance(price_process, BlackScholesPriceProcess)

        # Test the error paths.
        # - can't import the Python module
        self.assertRaises(DslError, get_price_process, 'x' + DEFAULT_PRICE_PROCESS_NAME)
        # - can't find the price process class
        self.assertRaises(DslError, get_price_process, DEFAULT_PRICE_PROCESS_NAME + 'x')
