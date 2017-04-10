from abc import abstractmethod

import six

from eventsourcing.application.base import EventSourcingApplication

from quantdsl.domain.model.call_dependencies import register_call_dependencies
from quantdsl.domain.model.call_dependents import register_call_dependents
from quantdsl.domain.model.call_link import register_call_link
from quantdsl.domain.model.call_requirement import register_call_requirement
from quantdsl.domain.model.contract_specification import register_contract_specification, ContractSpecification
from quantdsl.domain.model.contract_valuation import start_contract_valuation
from quantdsl.domain.model.dependency_graph import register_dependency_graph
from quantdsl.domain.model.market_calibration import register_market_calibration, compute_market_calibration_params
from quantdsl.domain.model.market_simulation import register_market_simulation, MarketSimulation
from quantdsl.domain.services.contract_valuations import loop_on_evaluation_queue, evaluate_call_and_queue_next_calls
from quantdsl.domain.services.simulated_prices import identify_simulation_requirements
from quantdsl.infrastructure.dependency_graph_subscriber import DependencyGraphSubscriber
from quantdsl.infrastructure.evaluation_subscriber import EvaluationSubscriber
from quantdsl.infrastructure.event_sourced_repos.call_dependencies_repo import CallDependenciesRepo
from quantdsl.infrastructure.event_sourced_repos.call_dependents_repo import CallDependentsRepo
from quantdsl.infrastructure.event_sourced_repos.call_leafs_repo import CallLeafsRepo
from quantdsl.infrastructure.event_sourced_repos.call_link_repo import CallLinkRepo
from quantdsl.infrastructure.event_sourced_repos.call_requirement_repo import CallRequirementRepo
from quantdsl.infrastructure.event_sourced_repos.call_result_repo import CallResultRepo
from quantdsl.infrastructure.event_sourced_repos.contract_specification_repo import ContractSpecificationRepo
from quantdsl.infrastructure.event_sourced_repos.contract_valuation_repo import ContractValuationRepo
from quantdsl.infrastructure.event_sourced_repos.market_calibration_repo import MarketCalibrationRepo
from quantdsl.infrastructure.event_sourced_repos.perturbation_dependencies_repo import PerturbationDependenciesRepo
from quantdsl.infrastructure.event_sourced_repos.market_simulation_repo import MarketSimulationRepo
from quantdsl.infrastructure.event_sourced_repos.simulated_price_dependencies_repo import \
    SimulatedPriceRequirementsRepo
from quantdsl.infrastructure.event_sourced_repos.simulated_price_repo import SimulatedPriceRepo
from quantdsl.infrastructure.simulation_subscriber import SimulationSubscriber


class QuantDslApplication(EventSourcingApplication):
    """

    Flow of user stories:

    Register contract specification (DSL text).  --> gives required market names
    Generate compile call dependency graph using contract specification (and observation time?).  --> gives required fixing times

    Register price histories.
    Generate market calibration for required market names using available price histories and observation time.

    Generate market simulation for required market names from market calibration, observation time, and fixing times.

    Evaluate contract given call dependency graph and market simulation.
    """

    def __init__(self, call_evaluation_queue=None, result_counters=None, usage_counters=None, *args, **kwargs):
        super(QuantDslApplication, self).__init__(*args, **kwargs)
        self.contract_specification_repo = ContractSpecificationRepo(event_store=self.event_store, use_cache=True)
        self.contract_valuation_repo = ContractValuationRepo(event_store=self.event_store, use_cache=True)
        self.market_calibration_repo = MarketCalibrationRepo(event_store=self.event_store, use_cache=True)
        self.market_simulation_repo = MarketSimulationRepo(event_store=self.event_store, use_cache=True)
        self.perturbation_dependencies_repo = PerturbationDependenciesRepo(event_store=self.event_store, use_cache=True)
        self.simulated_price_requirements_repo = SimulatedPriceRequirementsRepo(event_store=self.event_store, use_cache=True)
        self.simulated_price_repo = SimulatedPriceRepo(event_store=self.event_store, use_cache=True)
        self.call_requirement_repo = CallRequirementRepo(event_store=self.event_store, use_cache=True)
        self.call_dependencies_repo = CallDependenciesRepo(event_store=self.event_store, use_cache=True)
        self.call_dependents_repo = CallDependentsRepo(event_store=self.event_store, use_cache=True)
        self.call_leafs_repo = CallLeafsRepo(event_store=self.event_store, use_cache=True)
        self.call_link_repo = CallLinkRepo(event_store=self.event_store, use_cache=True)
        self.call_result_repo = CallResultRepo(event_store=self.event_store, use_cache=True)
        self.call_evaluation_queue = call_evaluation_queue
        self.result_counters = result_counters
        self.usage_counters = usage_counters

        self.simulation_subscriber = SimulationSubscriber(
            market_calibration_repo=self.market_calibration_repo,
            market_simulation_repo=self.market_simulation_repo,
        )
        self.dependency_graph_subscriber = DependencyGraphSubscriber(
            contract_specification_repo=self.contract_specification_repo,
            call_dependencies_repo=self.call_dependencies_repo,
            call_dependents_repo=self.call_dependents_repo,
            call_leafs_repo=self.call_leafs_repo,
            call_requirement_repo=self.call_requirement_repo,
        )
        self.evaluation_subscriber = EvaluationSubscriber(
            contract_valuation_repo=self.contract_valuation_repo,
            call_link_repo=self.call_link_repo,
            call_dependencies_repo=self.call_dependencies_repo,
            call_requirement_repo=self.call_requirement_repo,
            call_result_repo=self.call_result_repo,
            simulated_price_repo=self.simulated_price_repo,
            market_simulation_repo=self.market_simulation_repo,
            call_evaluation_queue=self.call_evaluation_queue,
            call_leafs_repo=self.call_leafs_repo,
            result_counters=self.result_counters,
            usage_counters=self.usage_counters,
            call_dependents_repo=self.call_dependents_repo,
            perturbation_dependencies_repo=self.perturbation_dependencies_repo,
            simulated_price_requirements_repo=self.simulated_price_requirements_repo,
        )

    @abstractmethod
    def create_stored_event_repo(self, **kwargs):
        raise NotImplementedError()

    def close(self):
        self.evaluation_subscriber.close()
        self.dependency_graph_subscriber.close()
        self.simulation_subscriber.close()
        super(QuantDslApplication, self).close()

    # Todo: Register historical data.

    def compute_market_calibration_params(self, price_process_name, historical_data):
        """
        Returns market calibration params for given price process name and historical data.
        """
        return compute_market_calibration_params(price_process_name, historical_data)

    def register_contract_specification(self, specification):
        """
        The contract specification is a Quant DSL module.
        """
        return register_contract_specification(specification=specification)

    def register_market_calibration(self, price_process_name, calibration_params):
        """
        Calibration params result from fitting a model of market dynamics to historical data.
        """
        assert isinstance(price_process_name, six.string_types)
        assert isinstance(calibration_params, (dict, list))
        return register_market_calibration(price_process_name, calibration_params)

    def register_market_simulation(self, market_calibration_id, observation_date, requirements, path_count, interest_rate):
        """
        A market simulation has simulated prices at specified times across a set of markets.
        """
        return register_market_simulation(market_calibration_id, observation_date, requirements, path_count, interest_rate)

    def register_dependency_graph(self, contract_specification_id):
        return register_dependency_graph(contract_specification_id)

    def register_call_requirement(self, call_id, dsl_source, effective_present_time):
        """
        A call requirement is a node of the dependency graph.
        """
        return register_call_requirement(
            call_id=call_id,
            dsl_source=dsl_source,
            effective_present_time=effective_present_time
        )

    def register_call_dependencies(self, call_id, dependencies):
        return register_call_dependencies(call_id=call_id, dependencies=dependencies)

    def register_call_dependents(self, call_id, dependents):
        return register_call_dependents(call_id=call_id, dependents=dependents)

    def register_call_link(self, link_id, call_id):
        return register_call_link(link_id, call_id)

    def identify_simulation_requirements(self, contract_specification, observation_date, requirements):
        assert isinstance(contract_specification, ContractSpecification), contract_specification
        assert isinstance(requirements, set)
        return identify_simulation_requirements(contract_specification.id,
                                                self.call_requirement_repo,
                                                self.call_link_repo,
                                                self.call_dependencies_repo,
                                                self.perturbation_dependencies_repo,
                                                observation_date,
                                                requirements)

    def start_contract_valuation(self, entity_id, dependency_graph_id, market_simulation):
        assert isinstance(dependency_graph_id, six.string_types), dependency_graph_id
        assert isinstance(market_simulation, MarketSimulation)
        return start_contract_valuation(entity_id, dependency_graph_id, market_simulation.id)

    def loop_on_evaluation_queue(self, call_result_lock, compute_pool=None, result_counters=None, usage_counters=None):
        loop_on_evaluation_queue(
            call_evaluation_queue=self.call_evaluation_queue,
            contract_valuation_repo=self.contract_valuation_repo,
            call_requirement_repo=self.call_requirement_repo,
            market_simulation_repo=self.market_simulation_repo,
            call_dependencies_repo=self.call_dependencies_repo,
            call_result_repo=self.call_result_repo,
            simulated_price_repo=self.simulated_price_repo,
            call_dependents_repo=self.call_dependents_repo,
            perturbation_dependencies_repo=self.perturbation_dependencies_repo,
            simulated_price_requirements_repo=self.simulated_price_requirements_repo,
            call_result_lock=call_result_lock,
            compute_pool=compute_pool,
            result_counters=result_counters,
            usage_counters=usage_counters,
        )

    def evaluate_call_and_queue_next_calls(self, contract_valuation_id, dependency_graph_id, call_id, lock):
        evaluate_call_and_queue_next_calls(
            contract_valuation_id=contract_valuation_id,
            dependency_graph_id=dependency_graph_id,
            call_id=call_id,
            call_evaluation_queue=self.call_evaluation_queue,
            contract_valuation_repo=self.contract_valuation_repo,
            call_requirement_repo=self.call_requirement_repo,
            market_simulation_repo=self.market_simulation_repo,
            call_dependencies_repo=self.call_dependencies_repo,
            call_result_repo=self.call_result_repo,
            simulated_price_repo=self.simulated_price_repo,
            call_dependents_repo=self.call_dependents_repo,
            perturbation_dependencies_repo=self.perturbation_dependencies_repo,
            simulated_price_requirements_repo=self.simulated_price_requirements_repo,
            call_result_lock=lock,
        )
