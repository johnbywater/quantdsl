from multiprocessing.pool import Pool
from multiprocessing.sharedctypes import SynchronizedArray
from threading import Thread, Event

import gevent
import scipy
import six
from multiprocessing import Value, Array

from gevent.queue import Queue

from quantdsl.domain.model.call_dependencies import CallDependencies
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_result import register_call_result, make_call_result_id, CallResult, \
    CallResultRepository
from quantdsl.domain.model.contract_specification import make_simulated_price_id
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.semantics import DslNamespace, list_fixing_dates


def generate_contract_valuation(contract_valuation_id, call_dependencies_repo, call_evaluation_queue, call_leafs_repo,
                                call_link_repo, call_requirement_repo, call_result_repo, contract_valuation_repo,
                                market_simulation_repo, simulated_price_repo, result_counters, usage_counters,
                                call_dependents_repo, market_dependencies_repo):
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
            market_dependencies_repo=market_dependencies_repo,
        )
    else:
        evaluate_contract_in_parallel(
            contract_valuation_id=contract_valuation_id,
            contract_valuation_repo=contract_valuation_repo,
            call_leafs_repo=call_leafs_repo,
            call_evaluation_queue=call_evaluation_queue,
            call_link_repo=call_link_repo,
            result_counters=result_counters,
            usage_counters=usage_counters,
            call_dependencies_repo=call_dependencies_repo,
            call_dependents_repo=call_dependents_repo,
            market_dependencies_repo=market_dependencies_repo,
        )


def evaluate_contract_in_series(contract_valuation_id, contract_valuation_repo, market_simulation_repo,
                                simulated_price_repo, call_requirement_repo, call_dependencies_repo, call_link_repo,
                                call_result_repo, market_dependencies_repo):
    """
    Computes value of contract by following the series execution order of its call dependency graph
    in directly evaluating each call in turn until the whole graph has been evaluated.
    """

    # Get the contract valuation entity (it knows which call dependency graph and which market simualation to use).
    contract_valuation = contract_valuation_repo[contract_valuation_id]
    assert isinstance(contract_valuation, ContractValuation), contract_valuation

    # Get the dependency graph ID.
    dependency_graph_id = contract_valuation.dependency_graph_id

    # Get the market simulation.
    market_simulation = market_simulation_repo[contract_valuation.market_simulation_id]

    # Follow the execution order...
    for call_id in regenerate_execution_order(dependency_graph_id, call_link_repo):

        # Get the call requirement entity.
        call_requirement = call_requirement_repo[call_id]

        # Get the market dependencies for this call.
        perturbed_market_names = market_dependencies_repo[call_id].dependencies

        # Compute the call result.
        result_value, perturbed_values = compute_call_result(
            contract_valuation=contract_valuation,
            call_requirement=call_requirement,
            market_simulation=market_simulation,
            perturbed_market_names=perturbed_market_names,
            call_dependencies_repo=call_dependencies_repo,
            call_result_repo=call_result_repo,
            simulated_price_repo=simulated_price_repo,
        )

        # Register the result.
        register_call_result(
            call_id=call_id,
            result_value=result_value,
            perturbed_values=perturbed_values,
            contract_valuation_id=contract_valuation_id,
            dependency_graph_id=dependency_graph_id,
        )


def evaluate_contract_in_parallel(contract_valuation_id, contract_valuation_repo, call_leafs_repo, call_link_repo,
                                  call_evaluation_queue, result_counters, usage_counters, call_dependencies_repo,
                                  call_dependents_repo, market_dependencies_repo):
    """
    Computes value of contract by putting the dependency graph leaves on an evaluation queue and expecting
    there is at least one worker loop evaluating the queued calls and putting satisfied dependents on the queue.
    """

    contract_valuation = contract_valuation_repo[contract_valuation_id]
    # assert isinstance(contract_valuation, ContractValuation), contract_valuation

    dependency_graph_id = contract_valuation.dependency_graph_id

    if result_counters is not None:
        assert usage_counters is not None
        for call_id in regenerate_execution_order(dependency_graph_id, call_link_repo):
            call_dependencies = call_dependencies_repo[call_id]
            call_dependents = call_dependents_repo[call_id]
            # assert isinstance(call_dependencies, CallDependencies)
            count_dependencies = len(call_dependencies.dependencies)
            count_dependents = len(call_dependents.dependents)
            # Crude attempt to count down using atomic operations, so we get an exception when we can't pop off the last one.
            call_result_id = make_call_result_id(contract_valuation_id, call_id)
            result_counters[call_result_id] = [None] * (count_dependencies - 1)
            usage_counters[call_result_id] = [None] * (count_dependents - 1)

    call_leafs = call_leafs_repo[dependency_graph_id]
    # assert isinstance(call_leafs, CallLeafs)

    for call_id in call_leafs.leaf_ids:
        call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, call_id))
        gevent.sleep(0)


def loop_on_evaluation_queue(call_evaluation_queue, contract_valuation_repo, call_requirement_repo,
                             market_simulation_repo, call_dependencies_repo, call_result_repo, simulated_price_repo,
                             call_dependents_repo, market_dependencies_repo, call_result_lock, compute_pool=None,
                             result_counters=None, usage_counters=None):
    while True:
        item = call_evaluation_queue.get()
        if isinstance(call_evaluation_queue, gevent.queue.Queue):
            gevent.sleep(0)
        try:
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
                market_dependencies_repo=market_dependencies_repo,
                call_result_lock=call_result_lock,
                compute_pool=compute_pool,
                result_counters=result_counters,
                usage_counters=usage_counters,
            )
        finally:
            call_evaluation_queue.task_done()


def evaluate_call_and_queue_next_calls(contract_valuation_id, dependency_graph_id, call_id, call_evaluation_queue,
                                       contract_valuation_repo, call_requirement_repo, market_simulation_repo,
                                       call_dependencies_repo, call_result_repo, simulated_price_repo,
                                       call_dependents_repo, market_dependencies_repo, call_result_lock, compute_pool=None,
                                       result_counters=None, usage_counters=None):

    # Get the contract valuation.
    contract_valuation = contract_valuation_repo[contract_valuation_id]
    assert isinstance(contract_valuation, ContractValuation)

    # Get the market simulation.
    market_simulation = market_simulation_repo[contract_valuation.market_simulation_id]

    call_requirement = call_requirement_repo[call_id]

    perturbed_market_names = market_dependencies_repo[call_id].dependencies

    result_value, perturbed_values = compute_call_result(
            contract_valuation,
            call_requirement,
            market_simulation,
            perturbed_market_names,
            call_dependencies_repo,
            call_result_repo,
            simulated_price_repo,
            compute_pool,
            )

    # Lock the results.
    if call_result_lock is not None:
        call_result_lock.acquire()

    try:
        # Register this result.
        register_call_result(
            call_id=call_id,
            result_value=result_value,
            perturbed_values=perturbed_values,
            contract_valuation_id=contract_valuation_id,
            dependency_graph_id=dependency_graph_id,
        )

        # Find next calls.
        ready_generator = find_dependents_ready_to_be_evaluated(
            contract_valuation_id=contract_valuation_id,
            call_id=call_id,
            call_dependencies_repo=call_dependencies_repo,
            call_dependents_repo=call_dependents_repo,
            call_result_repo=call_result_repo,
            result_counters=result_counters,
        )

        # Check dependencies, and discard result when dependency has been fully used.
        if usage_counters is not None:
            call_dependencies = call_dependencies_repo[call_id]
            for dependency_id in call_dependencies.dependencies:
                dependency_result_id = make_call_result_id(contract_valuation_id, dependency_id)
                try:
                    usage_counters[dependency_result_id].pop()  # Pop one off the array (atomic decrement).
                except (KeyError, IndexError):
                    call_result = call_result_repo[dependency_result_id]
                    # Todo: Maybe do discard operations after lock has been released?
                    call_result.discard()
                    # Need to remove from the cache if we are to save memory.
                    try:
                        del(call_result_repo._cache[dependency_result_id])
                    except:
                        pass

        if call_result_lock is not None:
            # Make a list from the generator, if we are locking results.
            next_call_ids = list(ready_generator)
        else:
            # Otherwise put things directly on the queue.
            next_call_ids = []
            for next_call_id in ready_generator:
                call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, next_call_id))
                gevent.sleep(0)

    finally:
        # Unlock the results.
        if call_result_lock is not None:
            call_result_lock.release()

    # Queue the next calls (if there are any - see above).
    for next_call_id in next_call_ids:
        call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, next_call_id))


def find_dependents_ready_to_be_evaluated(contract_valuation_id, call_id, call_dependents_repo, call_dependencies_repo,
                                          call_result_repo, result_counters):
    # assert isinstance(contract_valuation_id, six.string_types), contract_valuation_id
    # assert isinstance(call_id, six.string_types), call_id
    # assert isinstance(call_dependents_repo, CallDependentsRepository)
    # assert isinstance(call_dependencies_repo, CallDependenciesRepository)
    # assert isinstance(call_result_repo, CallResultRepository)

    # Get dependents (if any exist).
    try:
        call_dependents = call_dependents_repo[call_id]

    # Don't worry if there are none.
    except KeyError:
        pass

    else:

        # Check if any dependents are ready to be evaluated.
        ready_dependents = []
        if result_counters is not None:
            for dependent_id in call_dependents.dependents:
                dependent_result_id = make_call_result_id(contract_valuation_id, dependent_id)
                try:
                    result_counters[dependent_result_id].pop()  # Pop one off the array (atomic decrement).
                except (KeyError, IndexError):
                    ready_dependents.append(dependent_id)
            return ready_dependents

        # assert isinstance(call_dependents, CallDependents)

        else:
            # Todo: Maybe speed this up by prepreparing the dependent-dependencies (so it's just one query).
            dependent_threads = []
            for dependent_id in call_dependents.dependents:
                # Single-threaded identification of dependents ready to be evaluated.
                add_dependent_if_ready(call_dependencies_repo,
                                       call_id,
                                       call_result_repo,
                                       contract_valuation_id,
                                       dependent_id,
                                       ready_dependents)
            [t.join() for t in dependent_threads]
            return ready_dependents


def add_dependent_if_ready(call_dependencies_repo, call_id, call_result_repo, contract_valuation_id,
                           dependent_id, ready_dependents):

    # Get the dependencies of this dependent.
    dependent_dependencies = call_dependencies_repo[dependent_id]
    # assert isinstance(dependent_dependencies, CallDependencies)

    for dependent_dependency_id in dependent_dependencies.dependencies:

        # Skip if this dependent dependency is the given call.
        if dependent_dependency_id == call_id:
            continue

        # Skip if a result is missing.
        if is_result_missing(contract_valuation_id, dependent_dependency_id, call_result_repo):
            break

    else:
        ready_dependents.append(dependent_id)


def is_result_missing(contract_valuation_id, dependent_dependency_id, call_result_repo):
    call_result_id = make_call_result_id(contract_valuation_id, dependent_dependency_id)
    return call_result_id not in call_result_repo


def compute_call_result(contract_valuation, call_requirement, market_simulation, perturbed_market_names,
                        call_dependencies_repo, call_result_repo, simulated_price_repo, compute_pool=None):
    """
    Parses, compiles and evaluates a call requirement.
    """
    # assert isinstance(contract_valuation, ContractValuation), contract_valuation
    assert isinstance(call_requirement, CallRequirement), call_requirement
    # assert isinstance(market_simulation, MarketSimulation), market_simulation
    # assert isinstance(call_dependencies_repo, CallDependenciesRepository), call_dependencies_repo
    # assert isinstance(call_result_repo, CallResultRepository)
    # assert isinstance(simulated_price_repo, SimulatedPriceRepository)

    # Todo: Put getting the dependency values in a separate thread, and perhaps make each call a separate thread.
    # Parse the DSL source into a DSL module object (composite tree of objects that provide the DSL semantics).
    if call_requirement._dsl_expr is not None:
        dsl_expr = call_requirement._dsl_expr
    else:
        dsl_module = dsl_parse(call_requirement.dsl_source)
        dsl_expr = dsl_module.body[0]

    # assert isinstance(dsl_module, Module), "Parsed stubbed expr string is not a module: %s" % dsl_module

    present_time = call_requirement.effective_present_time or market_simulation.observation_date

    simulated_value_dict = {}
    for fixing_date in set([present_time] + list_fixing_dates(dsl_expr)):
        for market_name in market_simulation.market_names:
            #
            price_id = make_simulated_price_id(contract_valuation.market_simulation_id, market_name, fixing_date)
            simulated_price = simulated_price_repo[price_id]
            # assert isinstance(simulated_price, SimulatedPrice)
            # assert isinstance(simulated_price.value, scipy.ndarray)
            simulated_value_dict[price_id] = simulated_price.value

    # Get all the call results depended on by this call.
    dependency_results = get_dependency_results(
        contract_valuation_id=contract_valuation.id,
        call_id=call_requirement.id,
        dependencies_repo=call_dependencies_repo,
        result_repo=call_result_repo,
    )

    first_market_name = market_simulation.market_names[0] if market_simulation.market_names else None


    # Compute the call result.
    if compute_pool is None:
        result_value, perturbed_values = evaluate_dsl_expr(dsl_expr, first_market_name, market_simulation.id,
                                         market_simulation.interest_rate, present_time, simulated_value_dict,
                                         perturbed_market_names, dependency_results)
    else:
        assert isinstance(compute_pool, Pool)
        async_result = compute_pool.apply_async(
            evaluate_dsl_expr,
            args=(dsl_expr, first_market_name, market_simulation.id, market_simulation.interest_rate,
                  present_time, simulated_value_dict, perturbed_market_names, dependency_results),
        )
        gevent.sleep(0.0001)
        result_value, perturbed_values = async_result.get()

    # Return the result.
    return result_value, perturbed_values


def evaluate_dsl_expr(dsl_expr, first_market_name, simulation_id, interest_rate, present_time, simulated_value_dict,
                      perturbed_market_names, dependency_results):

    evaluation_kwds = {
        'simulated_value_dict': simulated_value_dict,
        'simulation_id': simulation_id,
        'interest_rate': interest_rate,
        'present_time': present_time,
        'first_market_name': first_market_name,
    }

    result_value = None
    perturbed_values = {}

    for perturbed_market_name in [''] + perturbed_market_names:
        evaluation_kwds['perturbed_market_name'] = perturbed_market_name

        # Initialise namespace with the dependency values.
        dependency_values = {}
        for stub_id in dependency_results.keys():
            (dependency_result_value, dependency_perturbed_values) = dependency_results[stub_id]
            try:
                dependency_value = dependency_perturbed_values[perturbed_market_name]
            except KeyError:
                dependency_value = dependency_result_value
            dependency_values[stub_id] = dependency_value

        dsl_locals = DslNamespace(dependency_values)

        # Compile the parsed expr using the namespace to give something that can be evaluated.
        dsl_expr_reduced = dsl_expr.reduce(dsl_locals=dsl_locals, dsl_globals=DslNamespace())
        # assert isinstance(dsl_expr, DslExpression), dsl_expr

        expr_value = dsl_expr_reduced.evaluate(**evaluation_kwds)
        if perturbed_market_name == '':
            assert result_value is None
            result_value = expr_value
        else:
            perturbed_values[perturbed_market_name] = expr_value

    return result_value, perturbed_values


dsl_expr_pool = None


def get_compute_pool():
    # return None
    global dsl_expr_pool
    if dsl_expr_pool is None:
        dsl_expr_pool = Pool(processes=4)
    return dsl_expr_pool


def get_dependency_results(contract_valuation_id, call_id, dependencies_repo, result_repo):
    assert isinstance(result_repo, CallResultRepository), result_repo
    dependency_results = {}
    stub_dependencies = dependencies_repo[call_id]
    assert isinstance(stub_dependencies, CallDependencies), stub_dependencies
    for stub_id in stub_dependencies.dependencies:
        call_result_id = make_call_result_id(contract_valuation_id, stub_id)
        stub_result = result_repo[call_result_id]
        assert isinstance(stub_result, CallResult)
        dependency_results[stub_id] = (stub_result.result_value, stub_result.perturbed_values)
    return dependency_results
