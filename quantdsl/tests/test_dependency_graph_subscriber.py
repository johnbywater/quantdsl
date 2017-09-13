from eventsourcing.domain.model.events import publish
from mock import patch
from mock import MagicMock
import unittest

from quantdsl.domain.model.call_dependencies import CallDependenciesRepository
from quantdsl.domain.model.call_dependents import CallDependentsRepository
from quantdsl.domain.model.call_leafs import CallLeafsRepository
from quantdsl.domain.model.call_requirement import CallRequirementRepository
from quantdsl.domain.model.contract_specification import ContractSpecificationRepository, ContractSpecification
from quantdsl.infrastructure.dependency_graph_subscriber import DependencyGraphSubscriber


class TestDependencyGraphSubscriber(unittest.TestCase):

    def setUp(self):
        contract_specification_repo = MagicMock(spec=ContractSpecificationRepository)
        call_dependencies_repo = MagicMock(spec=CallDependenciesRepository)
        call_dependents_repo = MagicMock(spec=CallDependentsRepository)
        call_leafs_repo = MagicMock(spec=CallLeafsRepository)
        call_requirement_repo = MagicMock(spec=CallRequirementRepository)
        self.dependency_graph_subscriber = DependencyGraphSubscriber(
            contract_specification_repo,
            call_dependencies_repo,
            call_dependents_repo,
            call_leafs_repo,
            call_requirement_repo
        )

    def tearDown(self):
        self.dependency_graph_subscriber.close()

    @patch('quantdsl.infrastructure.dependency_graph_subscriber.generate_dependency_graph')
    def test_dependency_graph_subscriber(self, generate_dependency_graph):
        market_simulation_created = ContractSpecification.Created(entity_id='1', specification='1')
        publish(market_simulation_created)
        self.assertEqual(generate_dependency_graph.call_count, 1)
