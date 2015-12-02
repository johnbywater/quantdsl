from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_result import register_call_result
from quantdsl.domain.services.dependency_graphs import get_dependency_values
from quantdsl.domain.services.fixing_dates import regenerate_execution_order
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.semantics import Module, DslNamespace, DslExpression


# Todo: As an alternative to evaluation in series, we could get the leaves and evaluation those,
# Todo: and then have a call result subscriber that calls a domain service to see if any of the
# Todo: dependents have all their dependency results available.

def evaluate_contract_in_series(market_simulation, dependency_graph_id, simulated_price_repo, call_requirement_repo,
                                call_dependencies_repo, call_link_repo, call_result_repo):
    for call_id in regenerate_execution_order(dependency_graph_id, call_link_repo):
        call = call_requirement_repo[call_id]
        assert isinstance(call, CallRequirement)

        # Evaluate the call requirement.
        result_value = evaluate_call_requirement(call, call_dependencies_repo, call_id, call_result_repo,
                                                 market_simulation, simulated_price_repo)

        # - store the result
        register_call_result(call_id=call_id, result_value=result_value)


def evaluate_call_requirement(call, call_dependencies_repo, call_id, call_result_repo, market_simulation,
                              simulated_price_repo):
    dependency_values = get_dependency_values(call_id, call_dependencies_repo, call_result_repo)
    # - parse the expr
    stubbed_module = dsl_parse(call.dsl_source)
    assert isinstance(stubbed_module,
                      Module), "Parsed stubbed expr string is not a module: %s" % stubbed_module
    # - build a namespace from the dependency values
    dsl_locals = DslNamespace(dependency_values)
    # - compile the parsed expr
    dsl_expr = stubbed_module.body[0].reduce(dsl_locals=dsl_locals, dsl_globals=DslNamespace())
    assert isinstance(dsl_expr, DslExpression), dsl_expr
    # - evaluate the compiled expr
    first_market_name = market_simulation.market_names[0] if market_simulation.market_names else None
    evaluation_kwds = {
        'simulated_price_repo': simulated_price_repo,
        'simulation_id': market_simulation.id,
        'interest_rate': market_simulation.interest_rate,
        'present_time': call.effective_present_time or market_simulation.observation_date,
        'first_market_name': first_market_name,
    }
    return dsl_expr.evaluate(**evaluation_kwds)
