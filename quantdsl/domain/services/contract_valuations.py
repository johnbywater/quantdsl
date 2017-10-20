from eventsourcing.domain.model.events import publish

from quantdsl.domain.model.call_dependencies import CallDependencies
from quantdsl.domain.model.call_result import CallResult, CallResultRepository, ResultValueComputed, \
    make_call_result_id, register_call_result
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.perturbation_dependencies import PerturbationDependencies
from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.semantics import DslNamespace
from eventsourcing.domain.model.events import publish

from quantdsl.domain.model.call_dependencies import CallDependencies
from quantdsl.domain.model.call_result import CallResult, CallResultRepository, ResultValueComputed, \
    make_call_result_id, register_call_result
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.perturbation_dependencies import PerturbationDependencies
from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.semantics import DslNamespace


def generate_contract_valuation(contract_valuation_id, call_dependencies_repo, call_evaluation_queue, call_leafs_repo,
                                call_link_repo, call_requirement_repo, call_result_repo, contract_valuation_repo,
                                market_simulation_repo, simulated_price_repo, perturbation_dependencies_repo,
                                simulated_price_dependencies_repo, is_double_sided_deltas):
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
            is_double_sided_deltas=is_double_sided_deltas,
        )
    else:
        evaluate_contract_in_parallel(
            contract_valuation_id=contract_valuation_id,
            contract_valuation_repo=contract_valuation_repo,
            call_leafs_repo=call_leafs_repo,
            call_evaluation_queue=call_evaluation_queue,
            is_double_sided_deltas=is_double_sided_deltas,
        )


def evaluate_contract_in_series(contract_valuation_id, contract_valuation_repo, market_simulation_repo,
                                simulated_price_repo, call_requirement_repo, call_dependencies_repo,
                                call_link_repo, call_result_repo, perturbation_dependencies_repo,
                                simulated_price_dependencies_repo, is_double_sided_deltas):
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
        result_value, perturbed_values, involved_market_names = compute_call_result(
            contract_valuation=contract_valuation,
            call_requirement=call_requirement,
            market_simulation=market_simulation,
            perturbation_dependencies=perturbation_dependencies,
            call_dependencies_repo=call_dependencies_repo,
            call_result_repo=call_result_repo,
            simulated_price_repo=simulated_price_repo,
            simulation_requirements=simulation_requirements,
            is_double_sided_deltas=is_double_sided_deltas,
        )

        # Register the result.
        register_call_result(
            call_id=call_id,
            result_value=result_value,
            perturbed_values=perturbed_values,
            contract_valuation_id=contract_valuation_id,
            contract_specification_id=contract_specification_id,
            involved_market_names=involved_market_names
        )


def evaluate_contract_in_parallel(contract_valuation_id, contract_valuation_repo, call_leafs_repo,
                                  call_evaluation_queue, is_double_sided_deltas):
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
                             perturbation_dependencies_repo, simulated_price_requirements_repo):
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
                simulated_price_requirements_repo=simulated_price_requirements_repo,
            )
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
    result_value, perturbed_values, involved_market_names = compute_call_result(
        contract_valuation=contract_valuation,
        call_requirement=call_requirement,
        market_simulation=market_simulation,
        perturbation_dependencies=perturbation_dependencies,
        call_dependencies_repo=call_dependencies_repo,
        call_result_repo=call_result_repo,
        simulated_price_repo=simulated_price_repo,
        simulation_requirements=simulation_requirements,
        is_double_sided_deltas=contract_valuation.is_double_sided_deltas,
    )

    # Register the call result.
    register_call_result(
        call_id=call_id,
        result_value=result_value,
        perturbed_values=perturbed_values,
        contract_valuation_id=contract_valuation_id,
        contract_specification_id=contract_specification_id,
        involved_market_names=involved_market_names
    )


def compute_call_result(contract_valuation, call_requirement, market_simulation, perturbation_dependencies,
                        call_dependencies_repo, call_result_repo, simulated_price_repo, simulation_requirements,
                        is_double_sided_deltas):
    """
    Parses, compiles and evaluates a call requirement.
    """
    assert isinstance(contract_valuation, ContractValuation), contract_valuation
    # assert isinstance(call_requirement, CallRequirement), call_requirement
    # assert isinstance(market_simulation, MarketSimulation), market_simulation
    # assert isinstance(call_dependencies_repo, CallDependenciesRepository), call_dependencies_repo
    # assert isinstance(call_result_repo, CallResultRepository)
    # assert isinstance(simulated_price_dict, SimulatedPriceRepository)

    present_time = call_requirement.present_time

    simulated_value_dict = {}

    simulation_id = contract_valuation.market_simulation_id

    involved_market_names = set()

    for simulation_requirement in simulation_requirements:
        (commodity_name, fixing_date, delivery_date) = simulation_requirement

        # Accumulate involved market names (needed in Choice for regressions).
        involved_market_names.add(commodity_name)

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

    # Accumulate market names from dependency results.
    for dependency_result in dependency_results.values():
        assert isinstance(dependency_result, CallResult), dependency_result
        involved_market_names.update(dependency_result.involved_market_names)

    result_value, perturbed_values = evaluate_dsl_expr(call_requirement._dsl_expr,
                                                       market_simulation.id,
                                                       market_simulation.interest_rate, present_time,
                                                       simulated_value_dict,
                                                       perturbation_dependencies.dependencies,
                                                       dependency_results,
                                                       market_simulation.path_count,
                                                       market_simulation.perturbation_factor,
                                                       contract_valuation.periodisation,
                                                       call_requirement.cost,
                                                       market_simulation.observation_date,
                                                       is_double_sided_deltas,
                                                       involved_market_names)

    # Return the result.
    return result_value, perturbed_values, involved_market_names


def evaluate_dsl_expr(dsl_expr, simulation_id, interest_rate, present_time, simulated_value_dict,
                      perturbation_dependencies, dependency_results, path_count, perturbation_factor, periodisation,
                      estimated_cost_of_expr, observation_date, is_double_sided_deltas, involved_market_names):

    evaluation_kwds = {
        'simulated_value_dict': simulated_value_dict,
        'simulation_id': simulation_id,
        'interest_rate': interest_rate,
        'perturbation_factor': perturbation_factor,
        'observation_date': observation_date,
        'present_time': present_time,
        'involved_market_names': involved_market_names,
        'path_count': path_count,
        'periodisation': periodisation,
    }

    result_value = None
    perturbed_values = {}

    # Decide perturbation names.
    perturbation_names = perturbation_dependencies
    if is_double_sided_deltas:
        perturbation_names += ['-' + p for p in perturbation_dependencies]


    for perturbation_name in [''] + perturbation_names:

        evaluation_kwds['active_perturbation'] = perturbation_name

        # Initialise namespace with the dependency values.
        dependency_values = {}
        for stub_id in dependency_results.keys():
            dependency_result = dependency_results[stub_id]
            try:
                dependency_value = dependency_result.perturbed_values[perturbation_name]
            except KeyError:
                dependency_value = dependency_result.result_value
            dependency_values[stub_id] = dependency_value

        # Prepare the namespace with the values the expression depends on.
        dsl_locals = DslNamespace(dependency_values)

        # Substitute Name elements, to give something that can be evaluated.
        dsl_expr_resolved = dsl_expr.substitute_names(dsl_locals)

        # Evaluate the expression.
        expr_value = dsl_expr_resolved.evaluate(**evaluation_kwds)
        if not perturbation_name:
            assert result_value is None
            result_value = expr_value
        else:
            perturbed_values[perturbation_name] = expr_value

        # Publish result value computed event.
        publish(ResultValueComputed(estimated_cost_of_expr))

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
        dependency_results[stub_id] = stub_result
    return dependency_results
