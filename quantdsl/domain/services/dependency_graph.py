from quantdsl.domain.model.call_dependencies import CallDependencies, CallDependenciesRepository
from quantdsl.domain.model.call_dependents import CallDependentsRepository
from quantdsl.domain.model.call_result import CallResult, CallResultRepository
from quantdsl.domain.model.dependency_graph import DependencyGraph
from quantdsl.domain.model.leaf_calls import LeafCallsRepository, LeafCalls


def get_dependency_values(call_id, dependencies_repo, result_repo):
    assert isinstance(result_repo, CallResultRepository)
    dependency_values = {}
    stub_dependencies = dependencies_repo[call_id]
    assert isinstance(stub_dependencies, CallDependencies)
    for stub_id in stub_dependencies:
        try:
            stub_result = result_repo[stub_id]
        except KeyError:
            raise
        else:
            assert isinstance(stub_result, CallResult)
            dependency_values[stub_id] = stub_result.result_value
    return dependency_values


def generate_call_order(dependency_graph, leaf_calls_repo, call_dependents_repo, call_dependencies_repo):
    assert isinstance(dependency_graph, DependencyGraph), dependency_graph
    assert isinstance(leaf_calls_repo, LeafCallsRepository)
    assert isinstance(call_dependents_repo, CallDependentsRepository)
    assert isinstance(call_dependencies_repo, CallDependenciesRepository)
    # Topological sort, using Kahn's algorithm.
    leaf_calls = leaf_calls_repo[dependency_graph.id]
    assert isinstance(leaf_calls, LeafCalls)
    S = set(leaf_calls.call_ids)
    removed = set()
    not_removed = set()
    while S:
        n = S.pop()
        yield n
        try:
            call_dependents = call_dependents_repo[n]
        except KeyError:
            pass
        else:
            for m in call_dependents:
                removed.add((n, m))
                if (n, m) in not_removed:
                    not_removed.remove((n, m))
                for d in call_dependencies_repo[m]:
                    if (d, m) not in removed:
                        not_removed.add((d, m))
                        break
                else:
                    S.add(m)
    if not_removed:
        raise Exception("Circular dependencies: %s" % not_removed)
