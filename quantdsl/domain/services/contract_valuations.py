from multiprocessing.pool import Pool

from eventsourcing.domain.model.events import publish

from quantdsl.domain.model.call_dependencies import CallDependencies
from quantdsl.domain.model.call_dependents import CallDependents
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_result import register_call_result, make_call_result_id, CallResult, \
    CallResultRepository, ResultValueComputed
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.perturbation_dependencies import PerturbationDependencies
from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.infrastructure.event_sourced_repos.call_result_repo import CallResultRepo
from quantdsl.semantics import DslNamespace


def generate_contract_valuation(contract_valuation_id, call_dependencies_repo, call_evaluation_queue, call_leafs_repo,
                                call_link_repo, call_requirement_repo, call_result_repo, contract_valuation_repo,
                                market_simulation_repo, simulated_price_repo, result_counters,
                                call_dependents_repo, perturbation_dependencies_repo, simulated_price_dependencies_repo):
    if not call_evaluation_queue:
        evaluate_contract_in_series(
            contract_valuation_id=contract_valuation_id,
            contract_valuation_repo=contract_valuation_repo,
            market_simulation_repo=market_simulation_repo,
            simulated_price_repo=simulated_price_repo,
            call_requirement_repo=call_requirement_repo,
            call_dependencies_repo=call_dependencies_repo,
            call_dependents_repo=call_dependents_repo,
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
            call_link_repo=call_link_repo,
            result_counters=result_counters,
            call_dependencies_repo=call_dependencies_repo,
            call_dependents_repo=call_dependents_repo,
            perturbation_dependencies_repo=perturbation_dependencies_repo,
            simulated_price_dependencies_repo=simulated_price_dependencies_repo,
        )


def evaluate_contract_in_series(contract_valuation_id, contract_valuation_repo, market_simulation_repo,
                                simulated_price_repo, call_requirement_repo, call_dependencies_repo,
                                call_dependents_repo, call_link_repo,
                                call_result_repo, perturbation_dependencies_repo, simulated_price_dependencies_repo):
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
            perturbation_dependencies_repo=perturbation_dependencies_repo,
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

        # # Check for results that should be deleted.
        # # - dependency results should be deleted if there is a result for each dependent of the dependency
        # call_dependencies = call_dependencies_repo[call_id]
        # assert isinstance(call_dependencies, CallDependencies)
        # call_dependents = call_dependents_repo[call_id]
        # assert isinstance(call_dependents, CallDependents), (type(call_dependents), CallDependents)
        #
        # for dependency_id in call_dependencies.dependencies:
        #     for dependent_id in call_dependents.dependents[dependency_id]:
        #         if dependent_id != call_id and dependent_id not in call_result_repo:
        #             # Need to keep it.
        #             break
        #     else:
        #         del (call_result_repo[dependency_id])


def evaluate_contract_in_parallel(contract_valuation_id, contract_valuation_repo, call_leafs_repo, call_link_repo,
                                  call_evaluation_queue, result_counters, call_dependencies_repo,
                                  call_dependents_repo, perturbation_dependencies_repo,
                                  simulated_price_dependencies_repo):
    """
    Computes value of contract by putting the dependency graph leaves on an evaluation queue and expecting
    there is at least one worker loop evaluating the queued calls and putting satisfied dependents on the queue.
    """

    contract_valuation = contract_valuation_repo[contract_valuation_id]
    # assert isinstance(contract_valuation, ContractValuation), contract_valuation

    contract_specification_id = contract_valuation.contract_specification_id

    if result_counters is not None:
        for call_id in regenerate_execution_order(contract_specification_id, call_link_repo):
            call_dependencies = call_dependencies_repo[call_id]
            call_dependents = call_dependents_repo[call_id]
            assert isinstance(call_dependencies, CallDependencies)
            assert isinstance(call_dependents, CallDependents)
            count_dependencies = len(call_dependencies.dependencies)
            # Crude attempt to count down using atomic operations, so we get an exception when we can't pop off the last one.
            call_result_id = make_call_result_id(contract_valuation_id, call_id)
            result_counters[call_result_id] = [None] * (count_dependencies - 1)

    call_leafs = call_leafs_repo[contract_specification_id]
    # assert isinstance(call_leafs, CallLeafs)

    for call_id in call_leafs.leaf_ids:
        call_evaluation_queue.put((contract_specification_id, contract_valuation_id, call_id))


def loop_on_evaluation_queue(call_evaluation_queue, contract_valuation_repo, call_requirement_repo,
                             market_simulation_repo, call_dependencies_repo, call_result_repo, simulated_price_repo,
                             call_dependents_repo, perturbation_dependencies_repo, simulated_price_requirements_repo,
                             call_result_lock, compute_pool=None, result_counters=None):
    while True:
        item = call_evaluation_queue.get()
        try:
            contract_specification_id, contract_valuation_id, call_id = item

            evaluate_call_and_queue_next_calls(
                contract_valuation_id=contract_valuation_id,
                contract_specification_id=contract_specification_id,
                call_id=call_id,
                call_evaluation_queue=call_evaluation_queue,
                contract_valuation_repo=contract_valuation_repo,
                call_requirement_repo=call_requirement_repo,
                market_simulation_repo=market_simulation_repo,
                call_dependencies_repo=call_dependencies_repo,
                call_result_repo=call_result_repo,
                simulated_price_repo=simulated_price_repo,
                call_dependents_repo=call_dependents_repo,
                perturbation_dependencies_repo=perturbation_dependencies_repo,
                simulated_price_requirements_repo=simulated_price_requirements_repo,
                call_result_lock=call_result_lock,
                compute_pool=compute_pool,
                result_counters=result_counters,
            )
        finally:
            call_evaluation_queue.task_done()


def evaluate_call_and_queue_next_calls(contract_valuation_id, contract_specification_id, call_id, call_evaluation_queue,
                                       contract_valuation_repo, call_requirement_repo, market_simulation_repo,
                                       call_dependencies_repo, call_result_repo, simulated_price_repo,
                                       call_dependents_repo, perturbation_dependencies_repo,
                                       simulated_price_requirements_repo, call_result_lock,
                                       compute_pool=None, result_counters=None):

    # Get the contract valuation.
    contract_valuation = contract_valuation_repo[contract_valuation_id]
    assert isinstance(contract_valuation, ContractValuation)

    # Get the market simulation.
    market_simulation = market_simulation_repo[contract_valuation.market_simulation_id]

    call_requirement = call_requirement_repo[call_id]

    perturbation_dependencies = perturbation_dependencies_repo[call_id]
    assert isinstance(perturbation_dependencies, PerturbationDependencies)

    # Get the simulated price requirements for this call.
    simulation_requirements = simulated_price_requirements_repo[call_id].requirements

    result_value, perturbed_values = compute_call_result(
            contract_valuation=contract_valuation,
            call_requirement=call_requirement,
            market_simulation=market_simulation,
            perturbation_dependencies=perturbation_dependencies,
            call_dependencies_repo=call_dependencies_repo,
            call_result_repo=call_result_repo,
            simulated_price_repo=simulated_price_repo,
            perturbation_dependencies_repo=perturbation_dependencies_repo,
            simulation_requirements=simulation_requirements,
            compute_pool=compute_pool,
            )

    # Lock the results.
    # - avoids race conditions when checking, after a result
    #   has been written, if all results are now available, whilst
    #   others are writing results.
    # - could perhaps do this with optimistic concurrency control, so
    #   that result events can be collected by dependents
    #   and then evaluated when all are received - to be robust against
    #   concurrent operations causing concurrency exceptions, an accumulating
    #   operation would require to be retried only as many times as there are
    #   remaining dependents.
    if call_result_lock is not None:
        call_result_lock.acquire()

    try:
        # Register this result.
        # Todo: Retries on concurrency errors.
        register_call_result(
            call_id=call_id,
            result_value=result_value,
            perturbed_values=perturbed_values,
            contract_valuation_id=contract_valuation_id,
            contract_specification_id=contract_specification_id,
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

        if call_result_lock is not None:
            # Make a list from the generator, if we are locking results.
            next_call_ids = list(ready_generator)
        else:
            # Otherwise put things directly on the queue.
            next_call_ids = []
            for next_call_id in ready_generator:
                call_evaluation_queue.put((contract_specification_id, contract_valuation_id, next_call_id))

    finally:
        # Unlock the results.
        if call_result_lock is not None:
            call_result_lock.release()

    # Queue the next calls (if there are any - see above).
    for next_call_id in next_call_ids:
        call_evaluation_queue.put((contract_specification_id, contract_valuation_id, next_call_id))


def find_dependents_ready_to_be_evaluated(contract_valuation_id, call_id, call_dependents_repo, call_dependencies_repo,
                                          call_result_repo, result_counters):
    # assert isinstance(contract_valuation_id, six.string_types), contract_valuation_id
    # assert isinstance(call_id, six.string_types), call_id
    # assert isinstance(call_dependents_repo, CallDependentsRepository)
    # assert isinstance(call_dependencies_repo, CallDependenciesRepository)
    # assert isinstance(call_result_repo, CallResultRepository)
    assert result_counters is not None

    # Get dependents (if any exist).
    try:
        call_dependents = call_dependents_repo[call_id]
    except KeyError:
        # Don't worry if there are none.
        pass
    else:
        # Check if any dependents are ready to be evaluated.
        ready = []
        for dependent_id in call_dependents.dependents:
            dependent_result_id = make_call_result_id(contract_valuation_id, dependent_id)
            try:
                result_counters[dependent_result_id].pop()  # Pop one off the array (atomic decrement).
            except (KeyError, IndexError):
                ready.append(dependent_id)
        return ready


def compute_call_result(contract_valuation, call_requirement, market_simulation, perturbation_dependencies,
                        call_dependencies_repo, call_result_repo, simulated_price_repo,
                        perturbation_dependencies_repo, simulation_requirements,
                        compute_pool=None):
    """
    Parses, compiles and evaluates a call requirement.
    """
    # assert isinstance(contract_valuation, ContractValuation), contract_valuation
    assert isinstance(call_requirement, CallRequirement), call_requirement
    assert isinstance(market_simulation, MarketSimulation), market_simulation
    # assert isinstance(call_dependencies_repo, CallDependenciesRepository), call_dependencies_repo
    # assert isinstance(call_result_repo, CallResultRepository)
    # assert isinstance(simulated_price_dict, SimulatedPriceRepository)

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
    # market_simulation.dependencies

    #
    # all_fixing_dates = set([present_time] + list_fixing_dates(dsl_expr))
    # market_dependencies = perturbation_dependencies_repo[call_requirement.id]
    # assert isinstance(market_dependencies, PerturbationDependencies)
    # all_delivery_points = market_dependencies.dependencies
    # for fixing_date in all_fixing_dates:
    #     for delivery_point in all_delivery_points:
    #         market_name = delivery_point[0]
    #         delivery_date = delivery_point[1]
    #         simulation_id = contract_valuation.market_simulation_id
    #         price_id = make_simulated_price_id(simulation_id, market_name, fixing_date, delivery_date)
    #         simulated_price = simulated_price_dict[price_id]
    #         # assert isinstance(simulated_price, SimulatedPrice)
    #         # assert isinstance(simulated_price.value, scipy.ndarray)
    #         simulated_value_dict[price_id] = simulated_price.value

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

    # Compute the call result.
    if compute_pool is None:
        result_value, perturbed_values = evaluate_dsl_expr(dsl_expr, first_commodity_name, market_simulation.id,
                                                           market_simulation.interest_rate, present_time, simulated_value_dict,
                                                           perturbation_dependencies.dependencies, 
                                                           dependency_results, market_simulation.path_count,
                                                           market_simulation.perturbation_factor)
    else:
        assert isinstance(compute_pool, Pool)
        async_result = compute_pool.apply_async(
            evaluate_dsl_expr,
            args=(dsl_expr, first_commodity_name, market_simulation.id, market_simulation.interest_rate,
                  present_time, simulated_value_dict, perturbation_dependencies.dependencies, dependency_results,
                  market_simulation.path_count, market_simulation.perturbation_factor),
        )
        result_value, perturbed_values = async_result.get()

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


dsl_expr_pool = None


def get_compute_pool():
    # return None
    global dsl_expr_pool
    if dsl_expr_pool is None:
        dsl_expr_pool = Pool(processes=4)
    return dsl_expr_pool


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
