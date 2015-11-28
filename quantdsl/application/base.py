from collections import defaultdict

from eventsourcing.application.base import EventSourcingApplication

from quantdsl.domain.model.call_dependencies import register_call_dependencies
from quantdsl.domain.model.call_dependents import register_call_dependents
from quantdsl.domain.model.call_requirement import register_call_requirement, CallRequirementData, CallRequirement
from quantdsl.domain.model.call_result import register_call_result
from quantdsl.domain.model.call_specification import CallSpecification
from quantdsl.domain.model.contract_specification import register_contract_specification, ContractSpecification
from quantdsl.domain.model.contract_valuation import register_contract_valuation
from quantdsl.domain.model.dependency_graph import register_dependency_graph, DependencyGraph
from quantdsl.domain.model.leaf_calls import register_leaf_calls
from quantdsl.domain.model.market_calibration import register_market_calibration, compute_market_calibration_params, \
    MarketCalibration
from quantdsl.domain.model.market_simulation import register_market_simulation, MarketSimulation
from quantdsl.domain.model.price_process import get_price_process
from quantdsl.domain.model.simulated_price import register_simulated_price
from quantdsl.domain.services.dependency_graph import get_dependency_values, generate_call_order
from quantdsl.infrastructure.event_sourced_repos.call_dependencies_repo import CallDependenciesRepo
from quantdsl.infrastructure.event_sourced_repos.call_dependents_repo import CallDependentsRepo
from quantdsl.infrastructure.event_sourced_repos.call_requirement_repo import CallRequirementRepo
from quantdsl.infrastructure.event_sourced_repos.call_result_repo import CallResultRepo
from quantdsl.infrastructure.event_sourced_repos.contract_specification_repo import ContractSpecificationRepo
from quantdsl.infrastructure.event_sourced_repos.contract_valuation_repo import ContractValuationRepo
from quantdsl.infrastructure.event_sourced_repos.leaf_calls_repo import LeafCallsRepo
from quantdsl.infrastructure.event_sourced_repos.market_calibration_repo import MarketCalibrationRepo
from quantdsl.infrastructure.event_sourced_repos.market_simulation_repo import MarketSimulationRepo
from quantdsl.infrastructure.event_sourced_repos.simulated_price_repo import SimulatedPriceRepo
from quantdsl.priceprocess.base import PriceProcess
from quantdsl.semantics import generate_stubbed_calls, extract_graph_structure, \
    extract_defs_and_exprs, DslExpression, DslNamespace, Module, StubbedCall
from quantdsl.services import dsl_parse


# def generate_call_requirements(root_stub_id, dsl_module, dsl_expr, dsl_globals, dsl_locals):
#     for stubbed_call in generate_stubbed_calls(root_stub_id, dsl_module, dsl_expr, dsl_globals, dsl_locals):


class BaseQuantDslApplication(EventSourcingApplication):
    """

    Flow of user stories:

    Register contract specification (DSL text).  --> gives required market names
    Generate compile call dependency graph using contract specification (and observation time?).  --> gives required fixing times

    Register price histories.
    Generate market calibration for required market names using available price histories and observation time.

    Generate market simulation for required market names from market calibration, observation time, and fixing times.

    Evaluate contract given call dependency graph and market simulation.
    """

    def __init__(self):
        super(BaseQuantDslApplication, self).__init__()
        self.contract_specification_repo = ContractSpecificationRepo(event_store=self.event_store)
        self.contract_valuation_repo = ContractValuationRepo(event_store=self.event_store)
        self.market_calibration_repo = MarketCalibrationRepo(event_store=self.event_store)
        self.market_simulation_repo = MarketSimulationRepo(event_store=self.event_store)
        self.simulated_price_repo = SimulatedPriceRepo(event_store=self.event_store)
        self.call_requirement_repo = CallRequirementRepo(event_store=self.event_store)
        self.call_dependencies_repo = CallDependenciesRepo(event_store=self.event_store)
        self.call_dependents_repo = CallDependentsRepo(event_store=self.event_store)
        self.leaf_calls_repo = LeafCallsRepo(event_store=self.event_store)
        self.call_result_repo = CallResultRepo(event_store=self.event_store)

    # Todo: Register historical data.

    def compute_market_calibration_params(self, price_process_name, historical_data):
        """
        Returns market calibration params for given price process name and historical data.
        """
        return compute_market_calibration_params(price_process_name, historical_data)

    def register_market_calibration(self, calibration_params):
        """
        Calibration params result from fitting a model of market dynamics to historical data.
        """
        assert isinstance(calibration_params, dict)
        return register_market_calibration(calibration_params)

    def register_market_simulation(self, market_calibration_id, price_process_name, market_names, fixing_times,
                                   observation_time, path_count):
        """
        A market simulation has simulated prices at specified times across a set of markets.
        """
        return register_market_simulation(market_calibration_id, price_process_name, market_names, fixing_times,
                                          observation_time, path_count)

    def register_simulated_price(self, market_simulation_id, market_name, fixing_time, price_value):
        """
        A simulated price is generated during a market simulation.
        """
        return register_simulated_price(market_simulation_id, market_name, fixing_time, price_value)

    def simulate_future_prices(self, market_calibration, market_simulation):
        assert isinstance(market_calibration, MarketCalibration)
        assert isinstance(market_simulation, MarketSimulation)
        price_process = get_price_process(market_simulation.price_process_name)
        assert isinstance(price_process, PriceProcess), price_process
        return price_process.simulate_future_prices(
                market_names=market_simulation.market_names,
                fixing_dates=market_simulation.fixing_times,
                observation_time=market_simulation.observation_time,
                path_count=market_simulation.path_count,
                market_calibration=market_calibration.calibration_params)

    def generate_simulated_prices(self, market_calibration, market_simulation):
        simulated_prices = self.simulate_future_prices(market_calibration, market_simulation)
        for market_name, fixing_time, price_value in simulated_prices:
            self.register_simulated_price(market_simulation.id, market_name, fixing_time, price_value)

    def generate_market_simulation(self, market_calibration, price_process_name, market_names, fixing_times,
                                   observation_time, path_count):
        market_simulation = self.register_market_simulation(market_calibration.id, price_process_name, market_names,
                                                            fixing_times, observation_time, path_count)
        self.generate_simulated_prices(market_calibration, market_simulation)
        return market_simulation

    def register_contract_specification(self, specification):
        """
        The contract specification is a Quant DSL module.
        """
        return register_contract_specification(specification=specification)

    def register_dependency_graph(self):
        return register_dependency_graph()

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

    def register_leaf_calls(self, dependency_graph_id, call_ids):
        return register_leaf_calls(dependency_graph_id, call_ids)

    def generate_dependency_graph(self, contract_specification):
        assert isinstance(contract_specification, ContractSpecification)
        dsl_module = dsl_parse(dsl_source=contract_specification.specification)
        assert isinstance(dsl_module, Module)

        dsl_globals = DslNamespace()
        function_defs, expressions = extract_defs_and_exprs(dsl_module, dsl_globals)
        dsl_expr = expressions[0]
        assert isinstance(dsl_expr, DslExpression)
        dsl_locals = DslNamespace()

        dependency_graph = self.register_dependency_graph()

        leaf_call_ids = []
        all_dependents = defaultdict(list)
        for call_id, call_requirement_data in generate_stubbed_calls(dependency_graph.id, dsl_module, dsl_expr, dsl_globals, dsl_locals):
            assert isinstance(call_requirement_data, CallRequirementData)

            dsl_source = call_requirement_data.dsl_source
            effective_present_time = call_requirement_data.effective_present_time
            dependencies = call_requirement_data.dependencies

            self.register_call_requirement(call_id, dsl_source, effective_present_time)
            self.register_call_dependencies(call_id, dependencies)
            if len(dependencies) == 0:
                leaf_call_ids.append(call_id)
            else:
                for dependency_call_id in dependencies:
                    all_dependents[dependency_call_id].append(call_id)
        for call_id, dependents in all_dependents.items():
            self.register_call_dependents(call_id, dependents)
        self.register_leaf_calls(dependency_graph.id, leaf_call_ids)

        return dependency_graph

    def register_call_result(self, call_id, result_value):
        return register_call_result(call_id=call_id, result_value=result_value)

    def register_contract_valuation(self, dependency_graph_id):
        return register_contract_valuation(dependency_graph_id)

    def generate_contract_valuation(self, dependency_graph, market_simulation, call_order):
        assert isinstance(market_simulation, MarketSimulation)
        v = self.register_contract_valuation(dependency_graph.id)

        for call_id in call_order:

            # Todo: Feed this order into a queue, don't do it like this :-)
            # Todo: Make call specification event sourced, and then have a handler respond to the Created event
            # Todo: by evaluating the call and registering a result, or by putting it on a queue for a worker.

            call = self.call_requirement_repo[call_id]
            assert isinstance(call, CallRequirement)

            # Evaluate the call requirement.

            first_market_name = market_simulation.market_names[0] if market_simulation.market_names else None
            evaluation_kwds = {
                'simulated_price_repo': self.simulated_price_repo,
                'simulation_id': market_simulation.id,
                'interest_rate': 0,
                'present_time': call.effective_present_time or market_simulation.observation_time,
                'first_market_name': first_market_name,
            }
            dependency_values = get_dependency_values(call_id, self.call_dependencies_repo, self.call_result_repo)

            # - parse the expr
            stubbed_module = dsl_parse(call.dsl_source)

            assert isinstance(stubbed_module, Module), "Parsed stubbed expr string is not a module: %s" % stubbed_module

            # - build a namespace from the dependency values
            dsl_locals = DslNamespace(dependency_values)

            # - compile the parsed expr
            dsl_expr = stubbed_module.body[0].reduce(dsl_locals=dsl_locals, dsl_globals=DslNamespace())
            assert isinstance(dsl_expr, DslExpression), dsl_expr

            # - evaluate the compiled expr
            result_value = dsl_expr.evaluate(**evaluation_kwds)

            # - store the result
            register_call_result(call_id=call_id, result_value=result_value)

    def generate_call_order(self, dependency_graph):
        assert isinstance(dependency_graph, DependencyGraph)
        return generate_call_order(dependency_graph, self.leaf_calls_repo, self.call_dependents_repo,
                                         self.call_dependencies_repo)
