from multiprocessing.pool import Pool

from eventsourcing.domain.model.events import publish

from quantdsl.domain.model.call_dependencies import CallDependencies
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_result import CallResult, CallResultRepository, ResultValueComputed, \
    make_call_result_id, register_call_result
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.perturbation_dependencies import PerturbationDependencies
from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.semantics import DslNamespace, Module


def generate_contract_valuation(contract_valuation_id, call_dependencies_repo, call_evaluation_queue, call_leafs_repo,
                                call_link_repo, call_requirement_repo, call_result_repo, contract_valuation_repo,
                                market_simulation_repo, simulated_price_repo, perturbation_dependencies_repo,
                                simulated_price_dependencies_repo):
    if not call_evaluation_queue:
        evaluate_contract_in_series(
            contract_valuation_id=contract_valuation_id,
            contract_valuation_repo=contract_valuation_repo,
            market_simulation_repo=market_simulation_repo,
            simulated_price_repo=simulated_price_repo,
            call_requirement_repo=call_requirement_repo,
            call_dependencies_repo=call_dependencies_repo,
            call_link_repo=call_link_repo,
            call_result_repo=call_result_repo,
            perturbation_dependencies_repo=perturbation_dependencies_repo,
            simulated_price_dependencies_repo=simulated_price_dependencies_repo,
        )
    else:
        evaluate_contract_in_parallel(
            contract_valuation_id=contract_valuation_id,
            contract_valuation_repo=contract_valuation_repo,
            call_leafs_repo=call_leafs_repo,
            call_evaluation_queue=call_evaluation_queue,
        )


def evaluate_contract_in_series(contract_valuation_id, contract_valuation_repo, market_simulation_repo,
                                simulated_price_repo, call_requirement_repo, call_dependencies_repo,
                                call_link_repo, call_result_repo, perturbation_dependencies_repo,
                                simulated_price_dependencies_repo):
    """
    Computes value of contract by following the series execution order of its call dependency graph
    in directly evaluating each call in turn until the whole graph has been evaluated.
    """

    # Get the contract valuation entity (it knows which call dependency graph and which market simualation to use).
    contract_valuation = contract_valuation_repo[contract_valuation_id]
    assert isinstance(contract_valuation, ContractValuation), contract_valuation

    # Get the contract specification ID.
    contract_specification_id = contract_valuation.contract_specification_id

    # Get the market simulation.
    market_simulation = market_simulation_repo[contract_valuation.market_simulation_id]

    # Follow the execution order...
    for call_id in regenerate_execution_order(contract_specification_id, call_link_repo):

        # Get the call requirement entity.
        call_requirement = call_requirement_repo[call_id]

        # Get the perturbation requirements for this call.
        perturbation_dependencies = perturbation_dependencies_repo[call_id]

        # Get the simulated price requirements for this call.
        simulation_requirements = simulated_price_dependencies_repo[call_id].requirements

        # Compute the call result.
        result_value, perturbed_values = compute_call_result(
            contract_valuation=contract_valuation,
            call_requirement=call_requirement,
            market_simulation=market_simulation,
            perturbation_dependencies=perturbation_dependencies,
            call_dependencies_repo=call_dependencies_repo,
            call_result_repo=call_result_repo,
            simulated_price_repo=simulated_price_repo,
            simulation_requirements=simulation_requirements
        )

        # Register the result.
        register_call_result(
            call_id=call_id,
            result_value=result_value,
            perturbed_values=perturbed_values,
            contract_valuation_id=contract_valuation_id,
            contract_specification_id=contract_specification_id,
        )


def evaluate_contract_in_parallel(contract_valuation_id, contract_valuation_repo, call_leafs_repo,
                                  call_evaluation_queue):
    """
    Computes value of contract by putting the dependency graph leaves on an evaluation queue and expecting
    there is at least one worker loop evaluating the queued calls and putting satisfied dependents on the queue.
    """

    contract_valuation = contract_valuation_repo[contract_valuation_id]
    # assert isinstance(contract_valuation, ContractValuation), contract_valuation

    contract_specification_id = contract_valuation.contract_specification_id

    call_leafs = call_leafs_repo[contract_specification_id]
    # assert isinstance(call_leafs, CallLeafs)

    for call_id in call_leafs.leaf_ids:
        call_evaluation_queue.put((contract_specification_id, contract_valuation_id, call_id))


def loop_on_evaluation_queue(call_evaluation_queue, contract_valuation_repo, call_requirement_repo,
                             market_simulation_repo, call_dependencies_repo, call_result_repo, simulated_price_repo,
                             call_dependents_repo, perturbation_dependencies_repo, simulated_price_requirements_repo):
    while True:
        item = call_evaluation_queue.get()
        try:
            contract_specification_id, contract_valuation_id, call_id = item

            evaluate_call_and_queue_next_calls(
                contract_valuation_id=contract_valuation_id,
                contract_specification_id=contract_specification_id,
                call_id=call_id,
                contract_valuation_repo=contract_valuation_repo,
                call_requirement_repo=call_requirement_repo,
                market_simulation_repo=market_simulation_repo,
                call_dependencies_repo=call_dependencies_repo,
                call_result_repo=call_result_repo,
                simulated_price_repo=simulated_price_repo,
                perturbation_dependencies_repo=perturbation_dependencies_repo,
                simulated_price_requirements_repo=simulated_price_requirements_repo)
        finally:
            call_evaluation_queue.task_done()


def evaluate_call_and_queue_next_calls(contract_valuation_id, contract_specification_id, call_id,
                                       contract_valuation_repo, call_requirement_repo, market_simulation_repo,
                                       call_dependencies_repo, call_result_repo, simulated_price_repo,
                                       perturbation_dependencies_repo, simulated_price_requirements_repo):
    # Get the contract valuation.
    contract_valuation = contract_valuation_repo[contract_valuation_id]
    assert isinstance(contract_valuation, ContractValuation)

    # Get the market simulation.
    market_simulation = market_simulation_repo[contract_valuation.market_simulation_id]

    # Get the call requirement.
    call_requirement = call_requirement_repo[call_id]

    # Get the perturbation dependencies for this call.
    perturbation_dependencies = perturbation_dependencies_repo[call_id]
    assert isinstance(perturbation_dependencies, PerturbationDependencies)

    # Get the simulated price requirements for this call.
    simulation_requirements = simulated_price_requirements_repo[call_id].requirements

    # Compute the call result.
    result_value, perturbed_values = compute_call_result(
        contract_valuation=contract_valuation,
        call_requirement=call_requirement,
        market_simulation=market_simulation,
        perturbation_dependencies=perturbation_dependencies,
        call_dependencies_repo=call_dependencies_repo,
        call_result_repo=call_result_repo,
        simulated_price_repo=simulated_price_repo,
        simulation_requirements=simulation_requirements,
    )

    # Register the call result.
    register_call_result(
        call_id=call_id,
        result_value=result_value,
        perturbed_values=perturbed_values,
        contract_valuation_id=contract_valuation_id,
        contract_specification_id=contract_specification_id,
    )


def compute_call_result(contract_valuation, call_requirement, market_simulation, perturbation_dependencies,
                        call_dependencies_repo, call_result_repo, simulated_price_repo, simulation_requirements):
    """
    Parses, compiles and evaluates a call requirement.
    """
    # assert isinstance(contract_valuation, ContractValuation), contract_valuation
    # assert isinstance(call_requirement, CallRequirement), call_requirement
    # assert isinstance(market_simulation, MarketSimulation), market_simulation
    # assert isinstance(call_dependencies_repo, CallDependenciesRepository), call_dependencies_repo
    # assert isinstance(call_result_repo, CallResultRepository)
    # assert isinstance(simulated_price_dict, SimulatedPriceRepository)

    # Parse the DSL source into a DSL module object (composite tree of objects that provide the DSL semantics).
    # if call_requirement._dsl_expr is not None:
    dsl_expr = call_requirement._dsl_expr
    # else:
    #     dsl_module = dsl_parse(call_requirement.dsl_source)
    #     assert isinstance(dsl_module, Module), "Parsed stubbed expr string is not a module: %s" % dsl_module
    #     dsl_expr = dsl_module.body[0]

    present_time = call_requirement.effective_present_time or market_simulation.observation_date

    simulated_value_dict = {}

    simulation_id = contract_valuation.market_simulation_id

    first_commodity_name = None

    for simulation_requirement in simulation_requirements:
        (commodity_name, fixing_date, delivery_date) = simulation_requirement
        if first_commodity_name is None:
            first_commodity_name = commodity_name
        price_id = make_simulated_price_id(simulation_id, commodity_name, fixing_date, delivery_date)
        simulated_price = simulated_price_repo[price_id]
        simulated_value_dict[price_id] = simulated_price.value

    # Get all the call results depended on by this call.
    dependency_results = get_dependency_results(
        contract_valuation_id=contract_valuation.id,
        call_id=call_requirement.id,
        dependencies_repo=call_dependencies_repo,
        result_repo=call_result_repo,
    )

    result_value, perturbed_values = evaluate_dsl_expr(dsl_expr, first_commodity_name, market_simulation.id,
                                                       market_simulation.interest_rate, present_time,
                                                       simulated_value_dict,
                                                       perturbation_dependencies.dependencies,
                                                       dependency_results, market_simulation.path_count,
                                                       market_simulation.perturbation_factor)

    # Return the result.
    return result_value, perturbed_values


def evaluate_dsl_expr(dsl_expr, first_commodity_name, simulation_id, interest_rate, present_time, simulated_value_dict,
                      perturbation_dependencies, dependency_results, path_count, perturbation_factor):
    evaluation_kwds = {
        'simulated_value_dict': simulated_value_dict,
        'simulation_id': simulation_id,
        'interest_rate': interest_rate,
        'perturbation_factor': perturbation_factor,
        'present_time': present_time,
        'first_commodity_name': first_commodity_name,
        'path_count': path_count,
    }

    result_value = None
    perturbed_values = {}

    for perturbation in [None] + perturbation_dependencies + ['-' + p for p in perturbation_dependencies]:

        evaluation_kwds['active_perturbation'] = perturbation

        # Initialise namespace with the dependency values.
        dependency_values = {}
        for stub_id in dependency_results.keys():
            (dependency_result_value, dependency_perturbed_values) = dependency_results[stub_id]
            try:
                dependency_value = dependency_perturbed_values[str(perturbation)]
            except KeyError:
                dependency_value = dependency_result_value
            dependency_values[stub_id] = dependency_value

        dsl_locals = DslNamespace(dependency_values)

        # Compile the parsed expr using the namespace to give something that can be evaluated.
        dsl_expr_reduced = dsl_expr.reduce(dsl_locals=dsl_locals, dsl_globals=DslNamespace())
        # assert isinstance(dsl_expr, DslExpression), dsl_expr

        expr_value = dsl_expr_reduced.evaluate(**evaluation_kwds)
        if perturbation is None:
            assert result_value is None
            result_value = expr_value
        else:
            perturbed_values[perturbation] = expr_value

        # Publish result value computed event.
        publish(ResultValueComputed())

    return result_value, perturbed_values


def get_dependency_results(contract_valuation_id, call_id, dependencies_repo, result_repo):
    assert isinstance(result_repo, (CallResultRepository, dict)), result_repo
    dependency_results = {}
    stub_dependencies = dependencies_repo[call_id]
    assert isinstance(stub_dependencies, CallDependencies), stub_dependencies
    for stub_id in stub_dependencies.dependencies:
        call_result_id = make_call_result_id(contract_valuation_id, stub_id)
        stub_result = result_repo[call_result_id]
        assert isinstance(stub_result, CallResult)
        dependency_results[stub_id] = (stub_result.result_value, stub_result.perturbed_values)
    return dependency_results
