import datetime

import six
from eventsourcing.application.base import EventSourcingApplication

from quantdsl import DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE
from quantdsl.application.call_result_policy import CallResultPolicy
from quantdsl.application.persistence_policy import PersistencePolicy
from quantdsl.domain.model.call_dependencies import register_call_dependencies
from quantdsl.domain.model.call_dependents import register_call_dependents
from quantdsl.domain.model.call_result import make_call_result_id
from quantdsl.domain.model.contract_specification import ContractSpecification, register_contract_specification
from quantdsl.domain.model.contract_valuation import ContractValuation, start_contract_valuation
from quantdsl.domain.model.market_calibration import register_market_calibration
from quantdsl.domain.model.market_simulation import register_market_simulation
from quantdsl.domain.model.perturbation_dependencies import PerturbationDependencies
from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.contract_valuations import loop_on_evaluation_queue
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
from quantdsl.infrastructure.simulation_subscriber import SimulationSubscriber
from quantdsl.semantics import discount



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

    def __init__(self, call_evaluation_queue=None, max_dependency_graph_size=DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE,
                 dsl_classes=None, *args, **kwargs):
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
        self.max_dependency_graph_size = max_dependency_graph_size
        self.dsl_classes = dsl_classes

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
            max_dependency_graph_size=self.max_dependency_graph_size,
            dsl_classes=self.dsl_classes,
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

    def register_contract_specification(self, source_code, observation_date=None):
        """
        Registers a new contract specification, from given Quant DSL source code.
        """
        return register_contract_specification(source_code=source_code, observation_date=observation_date)

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

    def identify_simulation_requirements(self, contract_specification, observation_date, requirements, periodisation):
        assert isinstance(contract_specification, ContractSpecification), type(contract_specification)
        assert isinstance(requirements, set)
        return identify_simulation_requirements(contract_specification.id,
                                                self.call_requirement_repo,
                                                self.call_link_repo,
                                                self.call_dependencies_repo,
                                                observation_date,
                                                requirements,
                                                periodisation)

    def start_contract_valuation(self, contract_specification_id, market_simulation_id, periodisation,
                                 is_double_sided_deltas):
        return start_contract_valuation(contract_specification_id, market_simulation_id, periodisation,
                                        is_double_sided_deltas)

    def loop_on_evaluation_queue(self):
        loop_on_evaluation_queue(
            call_evaluation_queue=self.call_evaluation_queue,
            contract_valuation_repo=self.contract_valuation_repo,
            call_requirement_repo=self.call_requirement_repo,
            market_simulation_repo=self.market_simulation_repo,
            call_dependencies_repo=self.call_dependencies_repo,
            call_result_repo=self.call_result_repo,
            simulated_price_repo=self.simulated_price_repo,
            perturbation_dependencies_repo=self.perturbation_dependencies_repo,
            simulated_price_requirements_repo=self.simulated_price_requirements_repo,
        )

    def compile(self, source_code, observation_date=None):
        return self.register_contract_specification(source_code=source_code, observation_date=observation_date)

    def simulate(self, contract_specification, price_process_name, calibration_params, observation_date,
                 path_count=20000, interest_rate='2.5', perturbation_factor=0.01, periodisation=None):

        market_calibration = self.register_market_calibration(price_process_name, calibration_params)

        simulation_requirements = set()
        self.identify_simulation_requirements(contract_specification, observation_date, simulation_requirements,
                                              periodisation)
        market_simulation = self.register_market_simulation(
            market_calibration_id=market_calibration.id,
            requirements=list(simulation_requirements),
            observation_date=observation_date,
            path_count=path_count,
            interest_rate=interest_rate,
            perturbation_factor=perturbation_factor,
        )
        return market_simulation

    def evaluate(self, contract_specification_id, market_simulation_id, periodisation=None,
                 is_double_sided_deltas=False):
        return self.start_contract_valuation(contract_specification_id, market_simulation_id, periodisation,
                                             is_double_sided_deltas)

    def get_result(self, contract_valuation):
        call_result_id = make_call_result_id(contract_valuation.id, contract_valuation.contract_specification_id)
        return self.call_result_repo[call_result_id]

    def get_periods(self, contract_valuation):
        assert isinstance(contract_valuation, ContractValuation)

        market_simulation = self.market_simulation_repo[contract_valuation.market_simulation_id]

        call_result_id = make_call_result_id(contract_valuation.id, contract_valuation.contract_specification_id)
        call_result = self.call_result_repo[call_result_id]

        fair_value = call_result.result_value

        perturbation_names = call_result.perturbed_values.keys()
        perturbation_names = [i for i in perturbation_names if not i.startswith('-')]
        perturbation_names = sorted(perturbation_names, key=lambda x: [int(i) for i in x.split('-')[1:]])

        periods = []
        for perturbation_name in perturbation_names:

            perturbed_value = call_result.perturbed_values[perturbation_name]
            if contract_valuation.is_double_sided_deltas:
                perturbed_value_negative = call_result.perturbed_values['-' + perturbation_name]
            else:
                perturbed_value_negative = None
            # Assumes format: NAME-YEAR-MONTH
            perturbed_name_split = perturbation_name.split('-')
            market_name = perturbed_name_split[0]

            if market_name == perturbation_name:
                simulated_price_id = make_simulated_price_id(market_simulation.id, market_name,
                                                             market_simulation.observation_date,
                                                             market_simulation.observation_date)

                simulated_price = self.simulated_price_repo[simulated_price_id]
                price = simulated_price.value
                if contract_valuation.is_double_sided_deltas:
                    dy = perturbed_value - perturbed_value_negative
                else:
                    dy = perturbed_value - fair_value

                dx = market_simulation.perturbation_factor * price
                if contract_valuation.is_double_sided_deltas:
                    dx *= 2

                delta = dy / dx

                hedge_units = - delta
                hedge_cost = hedge_units * price
                periods.append({
                    'market_name': market_name,
                    'delivery_date': None,
                    'delta': delta,
                    'perturbation_name': perturbation_name,
                    'hedge_units': hedge_units,
                    'price_simulated': price,
                    'price_discounted': price,
                    'hedge_cost': hedge_cost,
                    'cash': -hedge_cost,

                })

            elif len(perturbed_name_split) > 2:
                year = int(perturbed_name_split[1])
                month = int(perturbed_name_split[2])
                if len(perturbed_name_split) > 3:
                    day = int(perturbed_name_split[3])
                    delivery_date = datetime.date(year, month, day)
                    simulated_price_id = make_simulated_price_id(
                        market_simulation.id, market_name, delivery_date, delivery_date
                    )
                    simulated_price = self.simulated_price_repo[simulated_price_id]
                    simulated_price_value = simulated_price.value

                else:
                    delivery_date = datetime.date(year, month, 1)
                    sum_simulated_prices = 0
                    count_simulated_prices = 0
                    for i in range(1, 32):
                        try:
                            _delivery_date = datetime.date(year, month, i)
                        except ValueError:
                            continue
                        else:
                            simulated_price_id = make_simulated_price_id(
                                market_simulation.id, market_name, _delivery_date, _delivery_date
                            )
                            try:
                                simulated_price = self.simulated_price_repo[simulated_price_id]
                            except KeyError:
                                pass
                            else:
                                sum_simulated_prices += simulated_price.value
                                count_simulated_prices += 1
                    assert count_simulated_prices, "Can't find any simulated prices for {}-{}".format(year, month)
                    simulated_price_value = sum_simulated_prices / count_simulated_prices

                # Assume present time of perturbed values is observation date.
                if contract_valuation.is_double_sided_deltas:
                    dy = perturbed_value - perturbed_value_negative
                else:
                    dy = perturbed_value - fair_value

                discount_rate = discount(
                    value=1,
                    present_date=market_simulation.observation_date,
                    value_date=delivery_date,
                    interest_rate=market_simulation.interest_rate
                )

                discounted_simulated_price_value = simulated_price_value * discount_rate

                dx = market_simulation.perturbation_factor * simulated_price_value
                if contract_valuation.is_double_sided_deltas:
                    dx *= 2
                delta = dy / dx

                # The delta of a forward contract at the observation date
                # is the discount factor at the delivery date.
                forward_contract_delta = discount_rate

                # Flatten the book with delta hedging in forward markets.
                # delta + hedge-units * hedge-delta = 0
                # hence: hedge-units = -delta / hedge-delta
                hedge_units = -delta / forward_contract_delta

                # Present value of cost of hedge.
                hedge_cost = hedge_units * discounted_simulated_price_value

                periods.append({
                    'market_name': market_name,
                    'delivery_date': delivery_date,
                    'delta': delta,
                    'perturbation_name': perturbation_name,
                    'hedge_units': hedge_units,
                    'price_simulated': simulated_price_value,
                    'price_discounted': discounted_simulated_price_value,
                    'hedge_cost': hedge_cost,
                    'cash': -hedge_cost,
                })
        return periods

    def calc_call_count(self, contract_specification_id):
        # Todo: Return the call count from the compilation method?
        return len(list(regenerate_execution_order(contract_specification_id, self.call_link_repo)))

    def calc_counts_and_costs(self, contract_specification_id, is_double_sided_deltas):
        """Returns a dict of call IDs -> perturbation requirements."""
        costs = {}
        counts = {}
        for call_id in regenerate_execution_order(contract_specification_id, self.call_link_repo):

            # Get estimated cost of evaluating the expression once.
            call_requirement = self.call_requirement_repo[call_id]
            estimated_cost_of_expr = call_requirement.cost

            # Get the perturbation requirements for this call.
            perturbation_dependencies = self.perturbation_dependencies_repo[call_id]
            assert isinstance(perturbation_dependencies, PerturbationDependencies)
            # "1 + 2 * number of dependencies" because of the double sided delta.
            num_perturbation_dependencies = len(perturbation_dependencies.dependencies)
            num_perturbations = (2 if is_double_sided_deltas else 1) * num_perturbation_dependencies
            num_evaluations = 1 + num_perturbations

            # Cost is cost of doing it once, times the number of times it needs doing.
            costs[call_id] = num_evaluations * estimated_cost_of_expr
            counts[call_id] = num_evaluations

        return counts, costs
