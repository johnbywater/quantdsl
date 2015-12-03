from quantdsl.domain.model.call_leafs import CallLeafs
from quantdsl.domain.model.call_link import CallLink
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_result import register_call_result
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.services.dependency_graphs import get_dependency_values
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.semantics import Module, DslNamespace, DslExpression


# Todo: As an alternative to evaluation in series, we could get the leaves and evaluation those,
# Todo: and then have a call result subscriber that calls a domain service to see if any of the
# Todo: dependents have all their dependency results available.

def evaluate_contract_in_series(contract_valuation_id, contract_valuation_repo, market_simulation_repo,
                                simulated_price_repo, call_requirement_repo, call_dependencies_repo, call_link_repo,
                                call_result_repo):

    contract_valuation = contract_valuation_repo[contract_valuation_id]
    assert isinstance(contract_valuation, ContractValuation), contract_valuation

    dependency_graph_id = contract_valuation.dependency_graph_id

    for call_id in regenerate_execution_order(dependency_graph_id, call_link_repo):

        # Get the call requirement.
        call = call_requirement_repo[call_id]
        assert isinstance(call, CallRequirement)

        # Evaluate the call requirement.
        evaluate_call_requirement(
            contract_valuation=contract_valuation,
            call=call,
            market_simulation_repo=market_simulation_repo,
            call_dependencies_repo=call_dependencies_repo,
            call_result_repo=call_result_repo,
            simulated_price_repo=simulated_price_repo,
        )


def evaluate_contract_in_parallel(contract_valuation_id, contract_valuation_repo, call_leafs_repo,
                                  call_link_repo, call_evaluation_queue):

    contract_valuation = contract_valuation_repo[contract_valuation_id]
    assert isinstance(contract_valuation, ContractValuation), contract_valuation

    dependency_graph_id = contract_valuation.dependency_graph_id

    # call_leafs = call_leafs_repo[dependency_graph_id]
    # assert isinstance(call_leafs, CallLeafs)
    #
    # for call_id in call_leafs.leaf_ids:
    #     call_evaluation_queue.put((contract_valuation_id, call_id))

    call_link = call_link_repo[dependency_graph_id]
    assert isinstance(call_link, CallLink)
    call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, call_link.call_id))


def evaluate_call_requirement(contract_valuation, call, market_simulation_repo, call_dependencies_repo, call_result_repo,
                              simulated_price_repo):

    assert isinstance(contract_valuation, ContractValuation), contract_valuation
    assert isinstance(call, CallRequirement), call

    market_simulation = market_simulation_repo[contract_valuation.market_simulation_id]
    assert isinstance(market_simulation, MarketSimulation), market_simulation

    dependency_values = get_dependency_values(call.id, call_dependencies_repo, call_result_repo)

    # Parse the DSL expr.
    stubbed_module = dsl_parse(call.dsl_source)
    assert isinstance(stubbed_module, Module), "Parsed stubbed expr string is not a module: %s" % stubbed_module

    # Initialise the call namespace with the dependency values.
    dsl_locals = DslNamespace(dependency_values)

    # Compile the parsed expr.
    dsl_expr = stubbed_module.body[0].reduce(dsl_locals=dsl_locals, dsl_globals=DslNamespace())
    assert isinstance(dsl_expr, DslExpression), dsl_expr

    # Evaluate the compiled DSL.
    first_market_name = market_simulation.market_names[0] if market_simulation.market_names else None
    evaluation_kwds = {
        'simulated_price_repo': simulated_price_repo,
        'simulation_id': market_simulation.id,
        'interest_rate': market_simulation.interest_rate,
        'present_time': call.effective_present_time or market_simulation.observation_date,
        'first_market_name': first_market_name,
    }
    result_value = dsl_expr.evaluate(**evaluation_kwds)

    # Register the result.
    register_call_result(
        call_id=call.id,
        result_value=result_value,
        contract_valuation_id=contract_valuation.id,
        dependency_graph_id=contract_valuation.dependency_graph_id,
    )
