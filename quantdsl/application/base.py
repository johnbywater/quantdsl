import six
from eventsourcing.application.base import EventSourcingApplication

from quantdsl.application.call_result_policy import CallResultPolicy
from quantdsl.application.persistence_policy import PersistencePolicy
from quantdsl.domain.model.call_dependencies import register_call_dependencies
from quantdsl.domain.model.call_dependents import register_call_dependents
from quantdsl.domain.model.call_link import register_call_link
from quantdsl.domain.model.call_result import make_call_result_id
from quantdsl.domain.model.contract_specification import ContractSpecification, register_contract_specification
from quantdsl.domain.model.contract_valuation import start_contract_valuation
from quantdsl.domain.model.market_calibration import register_market_calibration
from quantdsl.domain.model.market_simulation import register_market_simulation
from quantdsl.domain.model.perturbation_dependencies import PerturbationDependencies
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.contract_valuations import evaluate_call_and_queue_next_calls, loop_on_evaluation_queue
from quantdsl.domain.services.simulated_prices import identify_simulation_requirements
from quantdsl.infrastructure.dependency_graph_subscriber import DependencyGraphSubscriber
from quantdsl.infrastructure.evaluation_subscriber import EvaluationSubscriber
from quantdsl.infrastructure.event_sourced_repos.call_dependencies_repo import CallDependenciesRepo
from quantdsl.infrastructure.event_sourced_repos.call_dependents_repo import CallDependentsRepo
from quantdsl.infrastructure.event_sourced_repos.call_leafs_repo import CallLeafsRepo
from quantdsl.infrastructure.event_sourced_repos.call_link_repo import CallLinkRepo
from quantdsl.infrastructure.event_sourced_repos.call_requirement_repo import CallRequirementRepo
from quantdsl.infrastructure.event_sourced_repos.contract_specification_repo import ContractSpecificationRepo
from quantdsl.infrastructure.event_sourced_repos.contract_valuation_repo import ContractValuationRepo
from quantdsl.infrastructure.event_sourced_repos.market_calibration_repo import MarketCalibrationRepo
from quantdsl.infrastructure.event_sourced_repos.market_simulation_repo import MarketSimulationRepo
from quantdsl.infrastructure.event_sourced_repos.perturbation_dependencies_repo import PerturbationDependenciesRepo
from quantdsl.infrastructure.event_sourced_repos.simulated_price_dependencies_repo import \
    SimulatedPriceRequirementsRepo
from quantdsl.infrastructure.event_sourced_repos.simulated_price_repo import SimulatedPriceRepo
from quantdsl.infrastructure.simulation_subscriber import SimulationSubscriber


class QuantDslApplication(EventSourcingApplication):
    """

    Flow of user stories:

    Register contract specification (DSL text).  --> gives required market names
    Generate compile call dependency graph using contract specification (and observation time?).  --> gives required 
    fixing times

    Register price histories.
    Generate market calibration for required market names using available price histories and observation time.

    Generate market simulation for required market names from market calibration, observation time, and fixing times.

    Evaluate contract given call dependency graph and market simulation.
    """

    def __init__(self, call_evaluation_queue=None, *args, **kwargs):
        super(QuantDslApplication, self).__init__(*args, **kwargs)
        self.contract_specification_repo = ContractSpecificationRepo(event_store=self.event_store, use_cache=True)
        self.contract_valuation_repo = ContractValuationRepo(event_store=self.event_store, use_cache=True)
        self.market_calibration_repo = MarketCalibrationRepo(event_store=self.event_store, use_cache=True)
        self.market_simulation_repo = MarketSimulationRepo(event_store=self.event_store, use_cache=True)
        self.perturbation_dependencies_repo = PerturbationDependenciesRepo(event_store=self.event_store,
                                                                           use_cache=True)
        self.simulated_price_requirements_repo = SimulatedPriceRequirementsRepo(event_store=self.event_store,
                                                                                use_cache=True)
        # self.simulated_price_repo = SimulatedPriceRepo(event_store=self.event_store, use_cache=True)
        self.simulated_price_repo = {}
        self.call_requirement_repo = CallRequirementRepo(event_store=self.event_store, use_cache=True)
        self.call_dependencies_repo = CallDependenciesRepo(event_store=self.event_store, use_cache=True)
        self.call_dependents_repo = CallDependentsRepo(event_store=self.event_store, use_cache=True)
        self.call_leafs_repo = CallLeafsRepo(event_store=self.event_store, use_cache=True)
        self.call_link_repo = CallLinkRepo(event_store=self.event_store, use_cache=True)
        # self.call_result_repo = CallResultRepo(event_store=self.event_store, use_cache=True)
        self.call_result_repo = {}
        self.call_evaluation_queue = call_evaluation_queue

        self.simulation_subscriber = SimulationSubscriber(
            market_calibration_repo=self.market_calibration_repo,
            market_simulation_repo=self.market_simulation_repo,
            simulated_price_repo=self.simulated_price_repo
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
            call_dependents_repo=self.call_dependents_repo,
            perturbation_dependencies_repo=self.perturbation_dependencies_repo,
            simulated_price_requirements_repo=self.simulated_price_requirements_repo,
        )
        self.call_result_policy = CallResultPolicy(self.call_result_repo, self.call_evaluation_queue)

    def create_persistence_subscriber(self):
        return PersistencePolicy(event_store=self.event_store)

    def close(self):
        self.evaluation_subscriber.close()
        self.dependency_graph_subscriber.close()
        self.simulation_subscriber.close()
        self.call_result_policy.close()
        super(QuantDslApplication, self).close()

    def register_contract_specification(self, source_code):
        """
        Registers a new contract specification, from given Quant DSL source code.
        """
        return register_contract_specification(source_code=source_code)

    def register_market_calibration(self, price_process_name, calibration_params):
        """
        Calibration params result from fitting a model of market dynamics to historical data.
        """
        assert isinstance(price_process_name, six.string_types)
        assert isinstance(calibration_params, (dict, list))
        return register_market_calibration(price_process_name, calibration_params)

    def register_market_simulation(self, market_calibration_id, observation_date, requirements, path_count,
                                   interest_rate, perturbation_factor=0.001):
        """
        A market simulation has simulated prices at specified times across a set of markets.
        """
        return register_market_simulation(market_calibration_id, observation_date, requirements, path_count,
                                          interest_rate, perturbation_factor)

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

    def start_contract_valuation(self, contract_specification_id, market_simulation_id):
        return start_contract_valuation(contract_specification_id, market_simulation_id)

    def loop_on_evaluation_queue(self):
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
        )

    def evaluate_call_and_queue_next_calls(self, contract_valuation_id, contract_specification_id, call_id):
        evaluate_call_and_queue_next_calls(
            contract_valuation_id=contract_valuation_id,
            contract_specification_id=contract_specification_id,
            call_id=call_id,
            contract_valuation_repo=self.contract_valuation_repo,
            call_requirement_repo=self.call_requirement_repo,
            market_simulation_repo=self.market_simulation_repo,
            call_dependencies_repo=self.call_dependencies_repo,
            call_result_repo=self.call_result_repo,
            simulated_price_repo=self.simulated_price_repo,
            perturbation_dependencies_repo=self.perturbation_dependencies_repo,
            simulated_price_requirements_repo=self.simulated_price_requirements_repo,
        )

    def compile(self, source_code):
        return self.register_contract_specification(source_code=source_code)

    def simulate(self, contract_specification, market_calibration, observation_date, path_count=20000,
                 interest_rate='2.5', perturbation_factor=0.001):
        simulation_requirements = set()
        self.identify_simulation_requirements(contract_specification, observation_date, simulation_requirements)
        market_simulation = self.register_market_simulation(
            market_calibration_id=market_calibration.id,
            requirements=list(simulation_requirements),
            observation_date=observation_date,
            path_count=path_count,
            interest_rate=interest_rate,
            perturbation_factor=perturbation_factor,
        )
        return market_simulation

    def evaluate(self, contract_specification_id, market_simulation_id):
        return self.start_contract_valuation(contract_specification_id, market_simulation_id)

    def get_result(self, contract_valuation):
        call_result_id = make_call_result_id(contract_valuation.id, contract_valuation.contract_specification_id)
        return self.call_result_repo[call_result_id]

    def calc_call_count(self, contract_specification_id):
        # Todo: Return the call count from the compilation method?
        return len(list(regenerate_execution_order(contract_specification_id, self.call_link_repo)))

    def calc_call_costs(self, contract_specification_id):
        """Returns a dict of call IDs -> perturbation requirements."""
        calls = {}
        for call_id in regenerate_execution_order(contract_specification_id, self.call_link_repo):
            # Get the perturbation requirements for this call.
            try:
                perturbation_dependencies = self.perturbation_dependencies_repo[call_id]
            except KeyError:
                calls[call_id] = 1
            else:
                assert isinstance(perturbation_dependencies, PerturbationDependencies)
                # "1 + 2 * number of dependencies" because of the double sided delta.
                calls[call_id] = 1 + 2 * len(perturbation_dependencies.dependencies)
        return calls
