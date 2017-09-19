from eventsourcing.domain.model.events import subscribe, unsubscribe

from quantdsl.domain.model.call_dependencies import CallDependenciesRepository
from quantdsl.domain.model.call_dependents import CallDependentsRepository
from quantdsl.domain.model.call_link import CallLinkRepository
from quantdsl.domain.model.call_requirement import CallRequirementRepository
from quantdsl.domain.model.call_result import CallResultRepository
from quantdsl.domain.model.contract_valuation import ContractValuation, ContractValuationRepository
from quantdsl.domain.model.market_simulation import MarketSimulationRepository
from quantdsl.domain.model.simulated_price import SimulatedPriceRepository
from quantdsl.domain.services.contract_valuations import generate_contract_valuation
from quantdsl.infrastructure.event_sourced_repos.perturbation_dependencies_repo import PerturbationDependenciesRepo
from quantdsl.infrastructure.event_sourced_repos.simulated_price_dependencies_repo import \
    SimulatedPriceRequirementsRepo


class EvaluationSubscriber(object):

    def __init__(self, contract_valuation_repo, call_link_repo, call_dependencies_repo, call_requirement_repo,
                 call_result_repo, simulated_price_repo, market_simulation_repo, call_leafs_repo,
                 call_evaluation_queue, call_dependents_repo,
                 perturbation_dependencies_repo, simulated_price_requirements_repo):
        assert isinstance(contract_valuation_repo, ContractValuationRepository), contract_valuation_repo
        assert isinstance(call_link_repo, CallLinkRepository), call_link_repo
        assert isinstance(call_dependencies_repo, CallDependenciesRepository), call_dependencies_repo
        assert isinstance(call_requirement_repo, CallRequirementRepository), call_requirement_repo
        assert isinstance(call_result_repo, (CallResultRepository, dict)), call_result_repo
        # assert isinstance(simulated_price_repo, SimulatedPriceRepository), simulated_price_repo
        assert isinstance(market_simulation_repo, MarketSimulationRepository), market_simulation_repo
        assert isinstance(call_dependents_repo, CallDependentsRepository), call_dependents_repo
        assert isinstance(perturbation_dependencies_repo, PerturbationDependenciesRepo), perturbation_dependencies_repo
        assert isinstance(simulated_price_requirements_repo, SimulatedPriceRequirementsRepo), simulated_price_requirements_repo
        self.contract_valuation_repo = contract_valuation_repo
        self.call_link_repo = call_link_repo
        self.call_dependencies_repo = call_dependencies_repo
        self.call_requirement_repo = call_requirement_repo
        self.call_result_repo = call_result_repo
        self.simulated_price_repo = simulated_price_repo
        self.market_simulation_repo = market_simulation_repo
        self.call_leafs_repo = call_leafs_repo
        self.call_evaluation_queue = call_evaluation_queue
        self.call_dependents_repo = call_dependents_repo
        self.perturbation_dependencies_repo = perturbation_dependencies_repo
        self.simulated_price_dependencies_repo = simulated_price_requirements_repo
        subscribe(self.is_contract_valuation_created, self.generate_contract_valuation)

    def close(self):
        unsubscribe(self.is_contract_valuation_created, self.generate_contract_valuation)

    @staticmethod
    def is_contract_valuation_created(event):
        return isinstance(event, ContractValuation.Created)

    def generate_contract_valuation(self, event):
        assert isinstance(event, ContractValuation.Created)
        generate_contract_valuation(contract_valuation_id=event.entity_id,
                                    call_dependencies_repo=self.call_dependencies_repo,
                                    call_evaluation_queue=self.call_evaluation_queue,
                                    call_leafs_repo=self.call_leafs_repo,
                                    call_link_repo=self.call_link_repo,
                                    call_requirement_repo=self.call_requirement_repo,
                                    call_result_repo=self.call_result_repo,
                                    contract_valuation_repo=self.contract_valuation_repo,
                                    market_simulation_repo=self.market_simulation_repo,
                                    simulated_price_repo=self.simulated_price_repo,
                                    perturbation_dependencies_repo=self.perturbation_dependencies_repo,
                                    simulated_price_dependencies_repo=self.simulated_price_dependencies_repo,
                                    )
