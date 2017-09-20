import unittest

from eventsourcing.domain.model.events import assert_event_handlers_empty, publish
from mock import MagicMock, Mock, patch

from quantdsl.domain.model.call_dependencies import CallDependenciesRepository
from quantdsl.domain.model.call_dependents import CallDependentsRepository
from quantdsl.domain.model.call_leafs import CallLeafs, CallLeafsRepository
from quantdsl.domain.model.call_link import CallLinkRepository
from quantdsl.domain.model.call_requirement import CallRequirementRepository
from quantdsl.domain.model.call_result import CallResultRepository
from quantdsl.domain.model.contract_valuation import ContractValuation, ContractValuationRepository
from quantdsl.domain.model.market_simulation import MarketSimulation, MarketSimulationRepository
from quantdsl.domain.model.simulated_price import SimulatedPriceRepository
from quantdsl.infrastructure.evaluation_subscriber import EvaluationSubscriber
from quantdsl.infrastructure.event_sourced_repos.perturbation_dependencies_repo import PerturbationDependenciesRepo
from quantdsl.infrastructure.event_sourced_repos.simulated_price_dependencies_repo import \
    SimulatedPriceRequirementsRepo


class TestEvaluationSubscriber(unittest.TestCase):
    def setUp(self):
        assert_event_handlers_empty()
        contract_valuation_repo = MagicMock(spec=ContractValuationRepository)
        contract_valuation_repo.__getitem__.return_value = Mock(spec=ContractValuation)
        market_simulation_repo = MagicMock(spec=MarketSimulationRepository)
        market_simulation_repo.__getitem__.return_value = Mock(spec=MarketSimulation)
        call_leafs_repo = MagicMock(spec=CallLeafsRepository)
        call_leafs_repo.__getitem__.return_value = Mock(spec=CallLeafs)
        call_dependents_repo = MagicMock(spec=CallDependentsRepository)
        perturbation_dependencies_repo = MagicMock(spec=PerturbationDependenciesRepo)
        simulated_price_requirements_repo = MagicMock(spec=SimulatedPriceRequirementsRepo)

        self.evaluation_subscriber = EvaluationSubscriber(
            contract_valuation_repo=contract_valuation_repo,
            call_link_repo=MagicMock(spec=CallLinkRepository),
            call_dependencies_repo=MagicMock(spec=CallDependenciesRepository),
            call_requirement_repo=MagicMock(spec=CallRequirementRepository),
            call_result_repo=MagicMock(spec=CallResultRepository),
            simulated_price_repo=MagicMock(spec=SimulatedPriceRepository),
            market_simulation_repo=market_simulation_repo,
            call_leafs_repo=call_leafs_repo,
            call_evaluation_queue=None,
            call_dependents_repo=call_dependents_repo,
            perturbation_dependencies_repo=perturbation_dependencies_repo,
            simulated_price_requirements_repo=simulated_price_requirements_repo
        )

    def tearDown(self):
        self.evaluation_subscriber.close()
        assert_event_handlers_empty()

    @patch('quantdsl.infrastructure.evaluation_subscriber.generate_contract_valuation')
    def test_evaluation_subscriber(self, evaluate_contract_in_series):
        # Check that when an event is published, the domain service is called.
        contract_valuation_created = ContractValuation.Created(
            entity_id='1',
            market_calibration_id='1',
        )
        self.assertEqual(evaluate_contract_in_series.call_count, 0)
        publish(contract_valuation_created)
        self.assertEqual(evaluate_contract_in_series.call_count, 1)
