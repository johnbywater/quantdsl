from multiprocessing.pool import Pool
from multiprocessing.sharedctypes import SynchronizedArray
from threading import Thread, Event

import eventlet
import scipy
import six
from multiprocessing import Value, Array, Process

from eventlet import tpool

from quantdsl.domain.model.call_result import register_call_result, make_call_result_id
from quantdsl.domain.model.contract_specification import make_simulated_price_id
from quantdsl.domain.services.dependency_graphs import get_dependency_values
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.semantics import DslNamespace, list_fixing_dates
# from quantdsl.semantics import numpy_from_sharedmem

drag_with_sharedmem = False


def generate_contract_valuation(contract_valuation_id, call_dependencies_repo, call_evaluation_queue, call_leafs_repo,
                                call_link_repo, call_requirement_repo, call_result_repo, contract_valuation_repo,
                                market_simulation_repo, simulated_price_repo, result_counters, call_dependents_repo):
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
        )
    else:
        evaluate_contract_in_parallel(
            contract_valuation_id=contract_valuation_id,
            contract_valuation_repo=contract_valuation_repo,
            call_leafs_repo=call_leafs_repo,
            call_evaluation_queue=call_evaluation_queue,
            call_link_repo=call_link_repo,
            result_counters=result_counters,
            call_dependencies_repo=call_dependencies_repo,
        )
        # Also run the green pool here.
        if isinstance(call_evaluation_queue, eventlet.Queue):
            pool = eventlet.GreenPool()
            # Keep looping if there are new calls to evaluate, or workers that may produce more.
            while True:
                item = call_evaluation_queue.get()
                dependency_graph_id, contract_valuation_id, call_id = item

                compute_pool = get_compute_pool()
                pool.spawn_n(evaluate_call_and_queue_next_calls,
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
                             call_result_lock=None,
                             compute_pool=compute_pool,
                             result_counters=result_counters,
                             )
                if call_id == dependency_graph_id:
                    # The last one.
                    pool.waitall()
                    break


def evaluate_contract_in_series(contract_valuation_id, contract_valuation_repo, market_simulation_repo,
                                simulated_price_repo, call_requirement_repo, call_dependencies_repo, call_link_repo,
                                call_result_repo):
    """
    Computes value of contract by following the series execution order of its call dependency graph
    in directly evaluating each call in turn until the whole graph has been evaluated.
    """

    # Get the contract valuation entity (it knows which call dependency graph and which market simualation to use).
    contract_valuation = contract_valuation_repo[contract_valuation_id]
    # assert isinstance(contract_valuation, ContractValuation), contract_valuation

    # Get the dependency graph ID.
    dependency_graph_id = contract_valuation.dependency_graph_id

    # Follow the execution order...
    for call_id in regenerate_execution_order(dependency_graph_id, call_link_repo):

        # Get the call requirement.
        call = call_requirement_repo[call_id]
        # assert isinstance(call, CallRequirement)

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
                                  call_evaluation_queue, result_counters, call_dependencies_repo):
    """
    Computes value of contract by putting the dependency graph leaves on an evaluation queue and expecting
    there is at least one worker loop evaluating the queued calls and putting satisfied dependents on the queue.
    """

    contract_valuation = contract_valuation_repo[contract_valuation_id]
    # assert isinstance(contract_valuation, ContractValuation), contract_valuation

    dependency_graph_id = contract_valuation.dependency_graph_id

    for call_id in regenerate_execution_order(dependency_graph_id, call_link_repo):
        call_dependencies = call_dependencies_repo[call_id]
        # assert isinstance(call_dependencies, CallDependencies)
        if result_counters is not None:
            count_dependencies = len(call_dependencies.dependencies)
            # Crude attempt to count down using atomic operations, so we get an exception when we can't pop off the last one.
            result_counters[call_id] = [None] * (count_dependencies - 1)

    call_leafs = call_leafs_repo[dependency_graph_id]
    # assert isinstance(call_leafs, CallLeafs)

    for call_id in call_leafs.leaf_ids:
        call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, call_id))


def loop_on_evaluation_queue(call_evaluation_queue, contract_valuation_repo, call_requirement_repo,
                             market_simulation_repo, call_dependencies_repo, call_result_repo, simulated_price_repo,
                             call_dependents_repo, call_result_lock, compute_pool=None, result_counters=None):
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
            compute_pool=compute_pool,
            result_counters=result_counters,
        )


def evaluate_call_and_queue_next_calls(contract_valuation_id, dependency_graph_id, call_id, call_evaluation_queue,
                                       contract_valuation_repo, call_requirement_repo, market_simulation_repo,
                                       call_dependencies_repo, call_result_repo, simulated_price_repo,
                                       call_dependents_repo, call_result_lock, compute_pool=None,
                                       result_counters=None):

    # Get the contract valuation.
    contract_valuation = contract_valuation_repo[contract_valuation_id]

    # Get the market simulation.
    market_simulation = market_simulation_repo[contract_valuation.market_simulation_id]
    # assert isinstance(market_simulation, MarketSimulation)

    call_requirement = call_requirement_repo[call_id]
    if isinstance(call_evaluation_queue, eventlet.Queue):
        result_value = tpool.execute(compute_call_result, contract_valuation,
                                           call_requirement,
                                           market_simulation,
                                           call_dependencies_repo,
                                           call_result_repo,
                                           simulated_price_repo,
                                           compute_pool,
                                           path_count=market_simulation.path_count)
    else:
        result_value = compute_call_result(contract_valuation,
                                           call_requirement,
                                           market_simulation,
                                           call_dependencies_repo,
                                           call_result_repo,
                                           simulated_price_repo,
                                           compute_pool,
                                           path_count=market_simulation.path_count)


    # # Lock the results.
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
        ready_generator = find_dependents_ready_to_be_evaluated(contract_valuation_id=contract_valuation_id, call_id=call_id,
                                                          call_dependencies_repo=call_dependencies_repo,
                                                          call_dependents_repo=call_dependents_repo,
                                                          call_result_repo=call_result_repo,
                                                          result_counters=result_counters, )
        if call_result_lock is not None:
            next_call_ids = list(ready_generator)
        else:
            next_call_ids = []
            for next_call_id in ready_generator:
                call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, next_call_id))

    finally:
        # Unlock the results.
        if call_result_lock is not None:
            call_result_lock.release()

    # Queue the next calls.
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
                try:
                    result_counters[dependent_id].pop()  # Pop one of the array (atomic decrement).
                except (KeyError, IndexError):
                    ready_dependents.append(dependent_id)
            return ready_dependents

        # assert isinstance(call_dependents, CallDependents)

        else:
            # Todo: Maybe speed this up by prepreparing the dependent-dependencies (so it's just one query).
            dependent_threads = []
            is_multithreaded = False
            for dependent_id in call_dependents.dependents:
                if is_multithreaded:
                    # Multi-threaded identification of dependents ready to be evaluated.
                    thread = Thread(target=add_dependent_if_ready,
                                    args=(call_dependencies_repo,
                                          call_id,
                                          call_result_repo,
                                          contract_valuation_id,
                                          dependent_id,
                                          ready_dependents))
                    thread.start()
                    dependent_threads.append(thread)
                else:
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

    is_multithreaded = False
    if is_multithreaded:

        # Multi-threaded checking of dependency results for this dependent.
        dependency_threads = []
        is_unsatisfied = Event()
        # Dependent is ready if each of its dependencies has a result.
        for dependent_dependency_id in dependent_dependencies.dependencies:

            # Skip if this dependent dependency is the given call.
            if dependent_dependency_id == call_id:
                continue

            # Look for any unsatisfied dependencies....
            thread = Thread(
                target=lambda *args: (is_result_missing(*args) and is_unsatisfied.set()),
                args=(contract_valuation_id, dependent_dependency_id, call_result_repo)
            )
            thread.start()
            dependency_threads.append(thread)

        [thread.join() for thread in dependency_threads]
        if not is_unsatisfied.is_set():
            # ....append all dependents that do not have a dependency without a result.
            ready_dependents.append(dependent_id)
    else:

        # Single threaded checking of dependency results for this dependent.
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


def compute_call_result(contract_valuation, call, market_simulation, call_dependencies_repo, call_result_repo,
                        simulated_price_repo, compute_pool=None, path_count=None):
    """
    Parses, compiles and evaluates a call requirement.
    """
    # assert isinstance(contract_valuation, ContractValuation), contract_valuation
    # assert isinstance(call, CallRequirement), call
    # assert isinstance(market_simulation, MarketSimulation), market_simulation
    # assert isinstance(call_dependencies_repo, CallDependenciesRepository), call_dependencies_repo
    # assert isinstance(call_result_repo, CallResultRepository)
    # assert isinstance(simulated_price_repo, SimulatedPriceRepository)

    # Todo: Put getting the dependency values in a separate thread, and perhaps make each call a separate thread.
    # Parse the DSL source into a DSL module object (composite tree of objects that provide the DSL semantics).
    if call._dsl_expr is not None:
        dsl_expr = call._dsl_expr
    else:
        dsl_module = dsl_parse(call.dsl_source)
        dsl_expr = dsl_module.body[0]

    # assert isinstance(dsl_module, Module), "Parsed stubbed expr string is not a module: %s" % dsl_module

    present_time = call.effective_present_time or market_simulation.observation_date

    simulated_value_dict = {}
    for fixing_date in set(list_fixing_dates(dsl_expr) + [present_time]):
        for market_name in market_simulation.market_names:
            price_id = make_simulated_price_id(contract_valuation.market_simulation_id, market_name, fixing_date)
            simulated_price = simulated_price_repo[price_id]
            # assert isinstance(simulated_price, SimulatedPrice)
            # assert isinstance(simulated_price.value, scipy.ndarray)
            if drag_with_sharedmem:
                price_value = sharedmem_from_numpy(simulated_price.value)
            else:
                price_value = simulated_price.value
            simulated_value_dict[price_id] = price_value

    # Get all the call results depended on by this call.
    dependency_values = get_dependency_values(contract_valuation.id, call.id, call_dependencies_repo, call_result_repo)

    for key in dependency_values.keys():
        call_result_value = dependency_values[key]
        if drag_with_sharedmem:
            dependency_values[key] = sharedmem_from_numpy(call_result_value)
        else:
            dependency_values[key] = call_result_value

    # Compute the call result.
    if compute_pool is None:
        result_value = evaluate_dsl_expr(dsl_expr, market_simulation.market_names, market_simulation.id,
                                         market_simulation.interest_rate, present_time, simulated_value_dict,
                                         None, None, **dependency_values)
    else:
        # We are multi-threading the call evaluation, but the computation can be dispatched to a subprocess
        # using shared memory to pass the data.
        assert isinstance(compute_pool, Pool)
        # result_value = evaluate_dsl_expr(dsl_expr, market_simulation, present_time, simulated_value_dict,
        #                                  None, None, **dependency_values)

        result = compute_pool.apply_async(
            evaluate_dsl_expr,
            args=(dsl_expr, market_simulation.market_names, market_simulation.id, market_simulation.interest_rate,
                  present_time, simulated_value_dict, None, path_count),
            kwds=dependency_values
        )
        result_value = result.get()
        # result_array = Array('d', scipy.zeros(path_count))
        # p = Process(
        #     target=evaluate_dsl_expr,
        #     args=(call.dsl_source, market_simulation, present_time, simulated_value_dict, result_array, path_count),
        #     kwargs=dependency_values
        # )
        # p.start()
        # p.join()
        # # result_value = numpy_from_sharedmem(result_array)
        # result_value = result_array

    # Return the result.
    return result_value


def sharedmem_from_numpy(simulated_price):
    if isinstance(simulated_price, scipy.ndarray):
        return Array('d', simulated_price)
    elif isinstance(simulated_price, six.integer_types + (float,)):
        return Value('d', simulated_price)
    elif isinstance(simulated_price, SynchronizedArray):
        return simulated_price
    else:
        raise NotImplementedError(type(simulated_price))


def evaluate_dsl_expr(dsl_expr, market_names, simulation_id, interest_rate, present_time, simulated_value_dict, result_array, path_count,
                      **dependency_values):

    # assert isinstance(result_array, (SynchronizedArray, type(None)))

    # Initialise namespace with the dependency values.
    dsl_locals = DslNamespace(dependency_values)

    # Compile the parsed expr using the namespace to give something that can be evaluated.
    dsl_expr = dsl_expr.reduce(dsl_locals=dsl_locals, dsl_globals=DslNamespace())
    # assert isinstance(dsl_expr, DslExpression), dsl_expr

    first_market_name = market_names[0] if market_names else None
    evaluation_kwds = {
        # 'simulated_price_repo': simulated_price_repo,
        'simulated_value_dict': simulated_value_dict,
        'simulation_id': simulation_id,
        'interest_rate': interest_rate,
        'present_time': present_time,
        'first_market_name': first_market_name,
    }
    result_value = dsl_expr.evaluate(**evaluation_kwds)

    if result_array is None:
        return result_value
    else:
        if isinstance(result_value, six.integer_types + (float,)):
            result_value = scipy.ones(path_count) * result_value
        result_array[:] = result_value
        return None


dsl_expr_pool = None


def get_compute_pool():
    # return None
    global dsl_expr_pool
    if dsl_expr_pool is None:
        dsl_expr_pool = Pool(processes=4)
    return dsl_expr_pool