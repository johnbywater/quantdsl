from abc import ABCMeta
import six

from quantdsl.dependency_graph import DependencyGraph
from quantdsl.exceptions import DslSyntaxError, DslSystemError
from quantdsl.semantics import Module, DslNamespace, DslExpression


class DependencyGraphRunner(six.with_metaclass(ABCMeta)):

    def __init__(self, dependency_graph):
        assert isinstance(dependency_graph, DependencyGraph)
        self.dependency_graph = dependency_graph

    def evaluate(self, **kwds):
        self.run(**kwds)
        try:
            return self.results_dict[self.dependency_graph.root_stub_id]
        except KeyError:
            errorData = (self.dependency_graph.root_stub_id, self.results_dict.keys())
            raise DslSystemError("root value not found", str(errorData))

    def run(self, **kwds):
        self.run_kwds = kwds
        self.call_count = 0
        self.results_dict = {}
        self.dependencies_by_stub = {}

    def get_dependency_values(self, call_requirement_id):
        dependency_values = {}
        dependency_stub_ids = self.dependencies_by_stub[call_requirement_id]
        for stub_id in dependency_stub_ids:
            try:
                stub_result = self.results_dict[stub_id]
            except KeyError:
                keys = self.results_dict.keys()
                raise KeyError("{} not in {} (really? {})".format(stub_id, keys, stub_id not in keys))
            else:
                dependency_values[stub_id] = stub_result
        return dependency_values

    def get_evaluation_kwds(self, stubbed_expr_str, effective_present_time):
        evaluation_kwds = self.run_kwds.copy()

        from quantdsl.services import dsl_parse, get_fixing_dates
        stubbed_module = dsl_parse(stubbed_expr_str)
        assert isinstance(stubbed_module, Module)
        # market_names = get_market_names(stubbed_module)
        fixing_dates = get_fixing_dates(stubbed_module)
        if effective_present_time is not None:
            fixing_dates.append(effective_present_time)

        # return evaluation_kwds
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


def evaluate_call(call_requirement_id, evaluation_kwds, dependency_values, result_queue, stubbed_expr_str, effective_present_time):
    """
    Evaluates the stubbed expr identified by 'call_requirement_id'.
    """

    # If necessary, overwrite the effective_present_time as the present_time in the evaluation_kwds.
    if effective_present_time:
        evaluation_kwds['present_time'] = effective_present_time

    # Evaluate the stubbed expr str.
    try:
        # Todo: Rework this dependency. Figure out how to use alternative set of DSL classes when multiprocessing.
        from quantdsl.services import dsl_parse
        stubbed_module = dsl_parse(stubbed_expr_str)
    except DslSyntaxError:
        raise

    assert isinstance(stubbed_module, Module), "Parsed stubbed expr string is not an module: %s" % stubbed_module

    dsl_namespace = DslNamespace()
    for stub_id, stub_result in dependency_values.items():
        dsl_namespace[stub_id] = stub_result

    simple_expr = stubbed_module.compile(dsl_locals=dsl_namespace, dsl_globals={})
    assert isinstance(simple_expr, DslExpression), "Reduced parsed stubbed expr string is not an " \
                                                   "expression: %s" % type(simple_expr)
    result_value = simple_expr.evaluate(**evaluation_kwds)
    result_queue.put((call_requirement_id, result_value))


def handle_result(call_requirement_id, result_value, results_dict, result_ids, notify_dict, dependency_dict,
                  execution_queue):

    # Set the results.
    results_dict[call_requirement_id] = result_value
    result_ids[call_requirement_id] = None

    # Check if subscribers are ready to be executed.
    for dependent_id in notify_dict[call_requirement_id]:
        if dependent_id in results_dict:
            continue
        subscriber_required_ids = dependency_dict[dependent_id]
        # It's ready unless it requires a call that doesn't have a result yet.
        for required_id in subscriber_required_ids:
            # - don't need to see if this call has a result, that's why we're here!
            if required_id != call_requirement_id:
                # - check if the required call already has a result
                if required_id not in results_dict:
                    break
        else:
            # All required results exist for the dependent call.
            execution_queue.put(dependent_id)

    # Check for results that should be deleted.
    # - dependency results should be deleted if there is a result for each dependent of the dependency
    for dependency_id in dependency_dict[call_requirement_id]:
        for dependent_id in notify_dict[dependency_id]:
            if dependent_id != call_requirement_id and dependent_id not in results_dict:
                # Need to keep it.
                break
        else:
            del(results_dict[dependency_id])
