from threading import Thread, Event

import six

from quantdsl.domain.model.call_dependencies import CallDependencies, CallDependenciesRepository
from quantdsl.domain.model.call_dependents import CallDependentsRepository, CallDependents
from quantdsl.domain.model.call_leafs import CallLeafs
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_result import register_call_result, CallResultRepository, make_call_result_id
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.simulated_price import SimulatedPriceRepository
from quantdsl.domain.services.dependency_graphs import get_dependency_values
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.semantics import Module, DslNamespace, DslExpression


def generate_contract_valuation(contract_valuation_id, call_dependencies_repo, call_evaluation_queue, call_leafs_repo, call_link_repo,
                                call_requirement_repo, call_result_repo, contract_valuation_repo,
                                market_simulation_repo, simulated_price_repo):
    if call_evaluation_queue:
        evaluate_contract_in_parallel(
            contract_valuation_id=contract_valuation_id,
            contract_valuation_repo=contract_valuation_repo,
            call_leafs_repo=call_leafs_repo,
            call_evaluation_queue=call_evaluation_queue,
            call_link_repo=call_link_repo,
        )
    else:
        evaluate_contract_in_series(
            contract_valuation_id=contract_valuation_id,
            contract_valuation_repo=contract_valuation_repo,
            market_simulation_repo=market_simulation_repo,
            simulated_price_repo=simulated_price_repo,
            call_requirement_repo=call_requirement_repo,
            call_dependencies_repo=call_dependencies_repo,
            call_link_repo=call_link_repo,
            call_result_repo=call_result_repo,
        )


def evaluate_contract_in_series(contract_valuation_id, contract_valuation_repo, market_simulation_repo,
                                simulated_price_repo, call_requirement_repo, call_dependencies_repo, call_link_repo,
                                call_result_repo):
    """
    Computes value of contract by following the series execution order of its call dependency graph
    in directly evaluating each call in turn until the whole graph has been evaluated.
    """

    # Get the contract valuation entity (it knows which call dependency graph and which market simualation to use).
    contract_valuation = contract_valuation_repo[contract_valuation_id]
    assert isinstance(contract_valuation, ContractValuation), contract_valuation

    # Get the dependency graph ID.
    dependency_graph_id = contract_valuation.dependency_graph_id

    # Follow the execution order...
    for call_id in regenerate_execution_order(dependency_graph_id, call_link_repo):

        # Get the call requirement.
        call = call_requirement_repo[call_id]
        assert isinstance(call, CallRequirement)

        # Get the market simulation.
        market_simulation = market_simulation_repo[contract_valuation.market_simulation_id]

        # Compute the call result.
        result_value = compute_call_result(contract_valuation=contract_valuation, call=call,
                                           market_simulation=market_simulation,
                                           call_dependencies_repo=call_dependencies_repo,
                                           call_result_repo=call_result_repo,
                                           simulated_price_repo=simulated_price_repo)
        # Register the result.
        register_call_result(
            call_id=call_id,
            result_value=result_value,
            contract_valuation_id=contract_valuation_id,
            dependency_graph_id=dependency_graph_id,
        )


def evaluate_contract_in_parallel(contract_valuation_id, contract_valuation_repo, call_leafs_repo, call_link_repo,
                                  call_evaluation_queue):
    """
    Computes value of contract by putting the dependency graph leaves on an evaluation queue and expecting
    there is at least one worker loop evaluating the queued calls and putting satisfied dependents on the queue.
    """

    contract_valuation = contract_valuation_repo[contract_valuation_id]
    assert isinstance(contract_valuation, ContractValuation), contract_valuation

    dependency_graph_id = contract_valuation.dependency_graph_id

    call_leafs = call_leafs_repo[dependency_graph_id]
    assert isinstance(call_leafs, CallLeafs)

    for call_id in call_leafs.leaf_ids:
        call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, call_id))


def loop_on_evaluation_queue(call_evaluation_queue, contract_valuation_repo, call_requirement_repo, market_simulation_repo,
                             call_dependencies_repo, call_result_repo, simulated_price_repo, call_dependents_repo, call_result_lock):
    while True:
        item = call_evaluation_queue.get()
        dependency_graph_id, contract_valuation_id, call_id = item

        evaluate_call_and_queue_next_calls(
            contract_valuation_id=contract_valuation_id,
            dependency_graph_id=dependency_graph_id,
            call_id=call_id,
            call_evaluation_queue=call_evaluation_queue,
            contract_valuation_repo=contract_valuation_repo,
            call_requirement_repo=call_requirement_repo,
            market_simulation_repo=market_simulation_repo,
            call_dependencies_repo=call_dependencies_repo,
            call_result_repo=call_result_repo,
            simulated_price_repo=simulated_price_repo,
            call_dependents_repo=call_dependents_repo,
            call_result_lock=call_result_lock,
        )


def evaluate_call_and_queue_next_calls(contract_valuation_id, dependency_graph_id, call_id, call_evaluation_queue, contract_valuation_repo, call_requirement_repo, market_simulation_repo,
                                       call_dependencies_repo, call_result_repo, simulated_price_repo, call_dependents_repo, call_result_lock):

    # Get the contract valuation.
    contract_valuation = contract_valuation_repo[contract_valuation_id]

    # Get the market simulation.
    market_simulation = market_simulation_repo[contract_valuation.market_simulation_id]

    result_value = compute_call_result(contract_valuation,
                                       call_requirement_repo[call_id], market_simulation,
                                       call_dependencies_repo, call_result_repo, simulated_price_repo)

    # Lock the results.
    if call_result_lock is not None:
        call_result_lock.acquire()

    try:
        # Register this result.
        register_call_result(
            call_id=call_id,
            result_value=result_value,
            contract_valuation_id=contract_valuation_id,
            dependency_graph_id=dependency_graph_id,
        )

        # Find next calls.
        next_call_ids = list(find_dependents_ready_to_be_evaluated(
            contract_valuation_id=contract_valuation_id,
            call_id=call_id,
            call_dependencies_repo=call_dependencies_repo,
            call_dependents_repo=call_dependents_repo,
            call_result_repo=call_result_repo))

    finally:
        # Unlock the results.
        if call_result_lock is not None:
            call_result_lock.release()

    # Queue the next calls.
    for next_call_id in next_call_ids:
        call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, next_call_id))


def find_dependents_ready_to_be_evaluated(contract_valuation_id, call_id, call_dependents_repo, call_dependencies_repo,
                                          call_result_repo):
    assert isinstance(contract_valuation_id, six.string_types), contract_valuation_id
    assert isinstance(call_id, six.string_types), call_id
    assert isinstance(call_dependents_repo, CallDependentsRepository)
    assert isinstance(call_dependencies_repo, CallDependenciesRepository)
    assert isinstance(call_result_repo, CallResultRepository)

    # Get dependents (if any exist).
    try:
        call_dependents = call_dependents_repo[call_id]

    # Don't worry if there are none.
    except KeyError:
        pass

    else:
        # Check if any dependents are ready to be evaluated.
        assert isinstance(call_dependents, CallDependents)

        # Todo: Maybe speed this up by prepreparing the dependent-dependencies (so it's just one query).

        dependent_threads = []

        ready_dependents = []

        for dependent_id in call_dependents.dependents:

            thread = Thread(target=get_dependencies_of_this_dependent,
                            args=(call_dependencies_repo,
                                  call_id, call_result_repo,
                                  contract_valuation_id, dependent_id, ready_dependents))
            thread.start()
            dependent_threads.append(thread)

        [t.join() for t in dependent_threads]

        return ready_dependents


def get_dependencies_of_this_dependent(call_dependencies_repo, call_id, call_result_repo, contract_valuation_id,
                                       dependent_id, ready_dependents):

    # Get the dependencies of this dependent.
    dependent_dependencies = call_dependencies_repo[dependent_id]
    assert isinstance(dependent_dependencies, CallDependencies)
    dependency_threads = []
    is_unsatisfied = Event()
    # Dependent is ready if each of its dependencies already has a result.
    for dependent_dependency_id in dependent_dependencies.dependencies:

        # Skip if this dependent dependency is the given call.
        if dependent_dependency_id == call_id:
            continue

        # Look for any unsatisfied dependencies....
        thread = Thread(
            target=check_result_missing,
            args=(contract_valuation_id, dependent_dependency_id, call_result_repo, is_unsatisfied)
        )
        thread.start()
        dependency_threads.append(thread)
    [thread.join() for thread in dependency_threads]
    if not is_unsatisfied.is_set():
        # ....append all dependents that do not have a dependency without a result.
        ready_dependents.append(dependent_id)


def check_result_missing(contract_valuation_id, dependent_dependency_id, call_result_repo, is_unsatisfied):
    call_result_id = make_call_result_id(contract_valuation_id, dependent_dependency_id)
    if call_result_id not in call_result_repo:
        is_unsatisfied.set()


def compute_call_result(contract_valuation, call, market_simulation, call_dependencies_repo, call_result_repo,
                        simulated_price_repo):
    """
    Parses, compiles and evaluates a call requirement.
    """
    assert isinstance(contract_valuation, ContractValuation), contract_valuation
    assert isinstance(call, CallRequirement), call
    assert isinstance(market_simulation, MarketSimulation), market_simulation
    assert isinstance(call_dependencies_repo, CallDependenciesRepository), call_dependencies_repo
    assert isinstance(call_result_repo, CallResultRepository)
    assert isinstance(simulated_price_repo, SimulatedPriceRepository)


    # Todo: Put getting the dependency values in a separate thread, and perhaps make each call a separate thread.
    # Get all the call results depended on by this call.
    dependency_values = get_dependency_values(contract_valuation.id, call.id, call_dependencies_repo, call_result_repo)

    # Initialise namespace with the dependency values.
    dsl_locals = DslNamespace(dependency_values)

    # Parse the DSL source into a DSL module object (composite tree of objects that provide the DSL semantics).
    stubbed_module = dsl_parse(call.dsl_source)
    assert isinstance(stubbed_module, Module), "Parsed stubbed expr string is not a module: %s" % stubbed_module

    # Todo: Join on getting the dependency values...

    # Compile the parsed expr using the namespace to give something that can be evaluated.
    dsl_expr = stubbed_module.body[0].reduce(dsl_locals=dsl_locals, dsl_globals=DslNamespace())
    assert isinstance(dsl_expr, DslExpression), dsl_expr

    # Evaluate the compiled DSL into a result.
    first_market_name = market_simulation.market_names[0] if market_simulation.market_names else None
    evaluation_kwds = {
        'simulated_price_repo': simulated_price_repo,
        'simulation_id': market_simulation.id,
        'interest_rate': market_simulation.interest_rate,
        'present_time': call.effective_present_time or market_simulation.observation_date,
        'first_market_name': first_market_name,
    }
    result_value = dsl_expr.evaluate(**evaluation_kwds)

    # Return the result.
    return result_value
