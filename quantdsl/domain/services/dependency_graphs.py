from collections import defaultdict

from quantdsl.domain.model.call_dependencies import  CallDependenciesRepository, register_call_dependencies
from quantdsl.domain.model.call_dependents import CallDependentsRepository, register_call_dependents
from quantdsl.domain.model.call_leafs import register_call_leafs
from quantdsl.domain.model.call_link import register_call_link
from quantdsl.domain.model.call_requirement import register_call_requirement
from quantdsl.domain.model.contract_specification import ContractSpecification
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.semantics import Module, DslNamespace, extract_defs_and_exprs, DslExpression, generate_stubbed_calls


def generate_dependency_graph(contract_specification, call_dependencies_repo, call_dependents_repo, call_leafs_repo,
                              call_requirement_repo):

    assert isinstance(contract_specification, ContractSpecification)
    dsl_module = dsl_parse(dsl_source=contract_specification.specification)
    assert isinstance(dsl_module, Module)
    dsl_globals = DslNamespace()
    function_defs, expressions = extract_defs_and_exprs(dsl_module, dsl_globals)
    dsl_expr = expressions[0]
    assert isinstance(dsl_expr, DslExpression)
    dsl_locals = DslNamespace()

    leaf_ids = []
    all_dependents = defaultdict(list)

    # Generate stubbed call from the parsed DSL module object.
    for stub in generate_stubbed_calls(contract_specification.id, dsl_module, dsl_expr, dsl_globals, dsl_locals):
        # assert isinstance(stub, StubbedCall)

        # Register the call requirements.
        call_id = stub.call_id
        dsl_source = str(stub.dsl_expr)
        effective_present_time = stub.effective_present_time
        call_requirement = register_call_requirement(call_id, dsl_source, effective_present_time)

        # Hold onto the dsl_expr, helps in "single process" modes....
        call_requirement._dsl_expr = stub.dsl_expr
        # - put the entity directly in the cache, otherwise the entity will be regenerated when it is next accessed
        #   and the _dsl_expr will be lost.
        call_requirement_repo.add_cache(call_id, call_requirement)

        # Register the call requirements.
        dependencies = stub.requirements
        register_call_dependencies(call_id, dependencies)

        # Keep track of the leaves and the dependents.
        if len(dependencies) == 0:
            leaf_ids.append(call_id)
        else:
            for dependency_call_id in dependencies:
                all_dependents[dependency_call_id].append(call_id)

    # Register the call dependents.
    for call_id, dependents in all_dependents.items():
        register_call_dependents(call_id, dependents)
    register_call_dependents(contract_specification.id, [])

    # Generate and register the call order.
    link_id = contract_specification.id
    for call_id in generate_execution_order(leaf_ids, call_dependents_repo, call_dependencies_repo):
        register_call_link(link_id, call_id)
        link_id = call_id

    # Register the leaf ids.
    register_call_leafs(contract_specification.id, leaf_ids)


def generate_execution_order(leaf_call_ids, call_dependents_repo, call_dependencies_repo):
    """
    Topological sort, using Kahn's algorithm.
    """
    assert isinstance(call_dependents_repo, CallDependentsRepository)
    assert isinstance(call_dependencies_repo, CallDependenciesRepository)


    # Initialise set of nodes that have no outstanding requirements with the leaf nodes.
    S = set(leaf_call_ids)
    removed_edges = defaultdict(set)
    while S:

        # Pick a node, n, that has zero outstanding requirements.
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
            # can add m to the set of nodes with zero outstanding requirements.
            for d in call_dependencies_repo[m]:
                if d not in removed_edges[m]:
                    break
            else:
                # Forget about removed edges to m.
                removed_edges.pop(m)

                # Add m to the set of nodes that have zero outstanding requirements.
                S.add(m)

    # Todo: Restore the check for remaining (unremoved) edges. Hard to do from the domain model,
    # so perhaps get all nodes in memory and actually remove them from a
    # collection so that we can see if anything remains unremoved (indicates cyclical dependencies).
