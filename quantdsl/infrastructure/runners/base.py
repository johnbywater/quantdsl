from abc import ABCMeta, abstractmethod

import six

from quantdsl.domain.model.call_specification import CallSpecification
from quantdsl.domain.model.dependency_graph import DependencyGraph
from quantdsl.exceptions import DslSyntaxError, DslSystemError
from quantdsl.semantics import Module, DslNamespace, DslExpression, compile_dsl_module, list_fixing_dates


class DependencyGraphRunner(six.with_metaclass(ABCMeta)):

    def __init__(self, dependency_graph):
        assert isinstance(dependency_graph, DependencyGraph)
        self.dependency_graph = dependency_graph

    def evaluate(self, **kwds):
        self.run(**kwds)
        try:
            return self.results_repo[self.dependency_graph.root_stub_id]
        except KeyError:
            raise DslSystemError("Result not found for root stub ID '{}'.".format(
                self.dependency_graph.root_stub_id
            ))

    @abstractmethod
    def run(self, **kwargs):
        self.run_kwds = kwargs
        self.call_count = 0
        self.results_repo = {}
        self.dependencies = {}

    def get_evaluation_kwds(self, dsl_source, effective_present_time):
        evaluation_kwds = self.run_kwds.copy()

        from quantdsl.services import list_fixing_dates
        from quantdsl.domain.services.parser import dsl_parse
        stubbed_module = dsl_parse(dsl_source)
        assert isinstance(stubbed_module, Module)
        fixing_dates = list_fixing_dates(stubbed_module)
        if effective_present_time is not None:
            fixing_dates.append(effective_present_time)

        # Rebuild the data structure (there was a problem, but I can't remember what it was.
        # Todo: Try without this block, perhaps the problem doesn't exist anymore.
        if 'all_market_prices' in evaluation_kwds:
            all_market_prices = evaluation_kwds.pop('all_market_prices')
            evaluation_kwds['all_market_prices'] = dict()
            for market_name in all_market_prices.keys():
                # if market_name not in market_names:
                #     continue
                market_prices = dict()
                for date in fixing_dates:
                    market_prices[date] = all_market_prices[market_name][date]
                evaluation_kwds['all_market_prices'][market_name] = market_prices
        return evaluation_kwds


def evaluate_call(call_spec, register_call_result):
    """
    Evaluates the stubbed expr identified by 'call_requirement_id'.
    """
    assert isinstance(call_spec, CallSpecification)

    evaluation_kwds = call_spec.evaluation_kwds.copy()

    # If this call has an effective present time value, use it as the 'present_time' in the evaluation_kwds.
    # This results from e.g. the Wait DSL element. Calls near the root of the expression might not have an
    # effective present time value, and the present time will be the observation time of the evaluation.

    # Evaluate the stubbed expr str.
    # - parse the expr
    try:
        # Todo: Rework this dependency. Figure out how to use alternative set of DSL classes when multiprocessing.
        from quantdsl.domain.services.parser import dsl_parse
        stubbed_module = dsl_parse(call_spec.dsl_expr_str)
    except DslSyntaxError:
        raise

    assert isinstance(stubbed_module, Module), "Parsed stubbed expr string is not a module: %s" % stubbed_module

    # - build a namespace from the dependency values
    dsl_locals = DslNamespace(call_spec.dependency_values)

    # - compile the parsed expr
    dsl_expr = stubbed_module.body[0].reduce(dsl_locals=dsl_locals, dsl_globals=DslNamespace())
    assert isinstance(dsl_expr, DslExpression), dsl_expr

    # - evaluate the compiled expr
    result_value = dsl_expr.evaluate(**evaluation_kwds)

    # - store the result
    register_call_result(call_id=call_spec.id, result_value=result_value)


def handle_result(call_requirement_id, result_value, results, dependents, dependencies, execution_queue):

    # Set the results.
    results[call_requirement_id] = result_value

    # Check if dependents are ready to be executed.
    for dependent_id in dependents[call_requirement_id]:
        if dependent_id in results:
            continue
        subscriber_required_ids = dependencies[dependent_id]
        # It's ready unless it requires a call that doesn't have a result yet.
        for required_id in subscriber_required_ids:
            # - don't need to see if this call has a result, that's why we're here!
            if required_id != call_requirement_id:
                # - check if the required call already has a result
                if required_id not in results:
                    break
        else:
            # All required results exist for the dependent call.
            execution_queue.put(dependent_id)

    # Check for results that should be deleted.
    # - dependency results should be deleted if there is a result for each dependent of the dependency
    for dependency_id in dependencies[call_requirement_id]:
        for dependent_id in dependents[dependency_id]:
            if dependent_id != call_requirement_id and dependent_id not in results:
                # Need to keep it.
                break
        else:
            del(results[dependency_id])
