from quantdsl.domain.model.call_link import CallLinkRepository, CallLink
from quantdsl.domain.model.call_requirement import CallRequirement, CallRequirementRepository
from quantdsl.domain.model.dependency_graph import DependencyGraph
from quantdsl.services import dsl_parse, find_fixing_times


def list_fixing_times(dependency_graph, call_requirement_repo, call_link_repo):
    assert isinstance(dependency_graph, DependencyGraph)
    assert isinstance(call_requirement_repo, CallRequirementRepository)
    assert isinstance(call_link_repo, CallLinkRepository)
    fixing_times = set()
    for call_id in regenerate_execution_order(dependency_graph, call_link_repo):

        # Get the stubbed expression.
        call_requirement = call_requirement_repo[call_id]
        assert isinstance(call_requirement, CallRequirement)
        dsl_expr = dsl_parse(call_requirement.dsl_source)

        # Find all the fixing times involved in this expression.
        for fixing_time in find_fixing_times(dsl_expr=dsl_expr):
            fixing_times.add(fixing_time)

    # Return a sorted list of fixing times.
    return sorted(list(fixing_times))


def regenerate_execution_order(dependency_graph, call_link_repo):
    assert isinstance(dependency_graph, DependencyGraph)
    assert isinstance(call_link_repo, CallLinkRepository)
    link_id = dependency_graph.id
    while True:
        call_link = call_link_repo[link_id]
        assert isinstance(call_link, CallLink)
        call_id = call_link.call_id
        yield call_id
        if call_id == dependency_graph.id:
            break
        link_id = call_id
