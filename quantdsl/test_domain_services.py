import datetime

import numpy
import six
from mock import Mock
from mock.mock import MagicMock
from twisted.trial import unittest

from quantdsl.domain.model.call_dependencies import CallDependenciesRepository, CallDependencies
from quantdsl.domain.model.call_dependents import CallDependentsRepository
from quantdsl.domain.model.call_link import CallLinkRepository, CallLink
from quantdsl.domain.model.call_requirement import CallRequirementRepository, CallRequirement
from quantdsl.domain.model.call_result import CallResultRepository, CallResult
from quantdsl.domain.model.contract_specification import ContractSpecification
from quantdsl.domain.model.dependency_graph import DependencyGraph
from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.services.dependency_graph import generate_execution_order, get_dependency_values
from quantdsl.domain.services.fixing_times import regenerate_execution_order, list_fixing_times
from quantdsl.domain.services.market_names import list_market_names
from quantdsl.domain.services.simulated_prices import simulate_future_prices
from quantdsl.domain.services.uuids import create_uuid4
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME


class TestUUIDs(unittest.TestCase):

    def test_create_uuid4(self):
        id = create_uuid4()
        self.assertIsInstance(id, six.string_types)


class TestDependencyGraph(unittest.TestCase):

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

        leaf_call_ids = [3]
        execution_order_gen = generate_execution_order(leaf_call_ids, call_dependents_repo, call_dependencies_repo)
        execution_order = list(execution_order_gen)
        self.assertEqual(execution_order, [2, 3, 1])

    def test_get_dependency_values(self):
        call_dependencies_repo = MagicMock(spec=CallDependenciesRepository,
                                         __getitem__=lambda self, x: {
                                             1: CallDependencies(dependencies=[2, 3], entity_id=123, entity_version=0, timestamp=1),
                                         }[x])
        call_result_repo = MagicMock(spec=CallResultRepository,
                                     __getitem__=lambda self, x: {
                                        2: Mock(spec=CallResult, result_value=12),
                                        3: Mock(spec=CallResult, result_value=13),
                                     }[x])
        values = get_dependency_values(1, call_dependencies_repo, call_result_repo)
        self.assertEqual(values, {2: 12, 3: 13})


class TestFixingTimes(unittest.TestCase):

    def test_regenerate_execution_order(self):
        dependency_graph = Mock(spec=DependencyGraph, id=1)
        call_link_repo = MagicMock(spec=CallLinkRepository,
                                   __getitem__=lambda self, x: {
                                       1: Mock(spec=CallLink, call_id=2),
                                       2: Mock(spec=CallLink, call_id=3),
                                       3: Mock(spec=CallLink, call_id=1),
                                   }[x])
        order = regenerate_execution_order(dependency_graph=dependency_graph, call_link_repo=call_link_repo)
        order = list(order)
        self.assertEqual(order, [2, 3, 1])

    def test_list_fixing_times(self):
        dependency_graph = Mock(spec=DependencyGraph, id=1)
        call_requirement_repo = MagicMock(spec=CallRequirementRepository,
                                   __getitem__=lambda self, x: {
                                       1: Mock(spec=CallRequirement, dsl_source="Fixing('2011-01-01', 1)"),
                                       2: Mock(spec=CallRequirement, dsl_source="Fixing('2012-02-02', 1)"),
                                       3: Mock(spec=CallRequirement, dsl_source="Fixing('2013-03-03', 1)"),
                                   }[x])
        call_link_repo = MagicMock(spec=CallLinkRepository,
                                   __getitem__=lambda self, x: {
                                       1: Mock(spec=CallLink, call_id=2),
                                       2: Mock(spec=CallLink, call_id=3),
                                       3: Mock(spec=CallLink, call_id=1),
                                   }[x])
        dates = list_fixing_times(dependency_graph, call_requirement_repo, call_link_repo)
        self.assertEqual(dates, [datetime.date(2011, 1, 1), datetime.date(2012, 2, 2), datetime.date(2013, 3, 3)])


class TestMarketNames(unittest.TestCase):

    def test_list_market_names(self):
        contract_specification = Mock(spec=ContractSpecification, specification="Market('#1') + Market('#2')")
        names = list_market_names(contract_specification=contract_specification)
        self.assertEqual(names, ['#1', '#2'])


class TestSimulatedPrices(unittest.TestCase):

    def test_simulate_future_prices(self):
        ms = Mock(spec=MarketSimulation,
                  observation_time=datetime.date(2011, 1, 1),
                  path_count=1,
                  market_names=['#1', '#2'],
                  fixing_times=[datetime.date(2011, 1, 2), datetime.date(2011, 1, 3)]
                  )
        mc = MagicMock(spec=MarketCalibration,
                       calibration_params={
                           '#1-#2-CORRELATION': 0.5,
                           '#1-LAST-PRICE': 10,
                           '#1-ACTUAL-HISTORICAL-VOLATILITY': 0,
                           '#2-LAST-PRICE': 9,
                           '#2-ACTUAL-HISTORICAL-VOLATILITY': 0,
                       })
        ms.price_process_name = DEFAULT_PRICE_PROCESS_NAME
        prices = simulate_future_prices(market_simulation=ms, market_calibration=mc)
        self.assertEqual(list(prices), [
            ('#1', datetime.date(2011, 1, 1), numpy.array([ 10.])),
            ('#1', datetime.date(2011, 1, 2), numpy.array([ 10.])),
            ('#1', datetime.date(2011, 1, 3), numpy.array([ 10.])),
            ('#2', datetime.date(2011, 1, 1), numpy.array([ 9.])),
            ('#2', datetime.date(2011, 1, 2), numpy.array([ 9.])),
            ('#2', datetime.date(2011, 1, 3), numpy.array([ 9.])),
        ])
