from collections import defaultdict

from quantdsl.domain.model.call_dependencies import CallDependencies, CallDependenciesRepository
from quantdsl.domain.model.call_dependents import CallDependentsRepository
from quantdsl.domain.model.call_result import CallResult, CallResultRepository


def get_dependency_values(call_id, dependencies_repo, result_repo):
    assert isinstance(result_repo, CallResultRepository), result_repo
    dependency_values = {}
    stub_dependencies = dependencies_repo[call_id]
    assert isinstance(stub_dependencies, CallDependencies), stub_dependencies
    for stub_id in stub_dependencies:
        try:
            stub_result = result_repo[stub_id]
        except KeyError:
            raise
        else:
            assert isinstance(stub_result, CallResult), stub_result
            dependency_values[stub_id] = stub_result.result_value
    return dependency_values


def generate_execution_order(leaf_call_ids, call_dependents_repo, call_dependencies_repo):
    assert isinstance(call_dependents_repo, CallDependentsRepository)
    assert isinstance(call_dependencies_repo, CallDependenciesRepository)

    # Topological sort, using Kahn's algorithm.

    # Initialise set of nodes that have no outstanding dependencies with the leaf nodes.
    S = set(leaf_call_ids)
    removed_edges = defaultdict(set)
    while S:

        # Pick a node, n, that has zero outstanding dependencies.
        n = S.pop()

        # Yield node n.
        yield n

        # Get dependents, if any were registered.
        try:
            dependents = call_dependents_repo[n]
        except KeyError:
            continue

        # Visit the nodes that are dependent on n.
        for m in dependents:

            # Remove the edge n to m from the graph.
            removed_edges[m].add(n)

            # If there are zero edges to m that have not been removed, then we
            # can add m to the set of nodes with zero outstanding dependencies.
            for d in call_dependencies_repo[m]:
                if d not in removed_edges[m]:
                    break
            else:
                # Forget about removed edges to m.
                removed_edges.pop(m)

                # Add m to the set of nodes that have zero outstanding dependencies.
                S.add(m)
