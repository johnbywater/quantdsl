from quantdsl.domain.model.call_link import CallLinkRepository
from quantdsl.domain.model.call_requirement import CallRequirement, CallRequirementRepository
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.semantics import find_fixing_dates


def list_fixing_dates(dependency_graph_id, call_requirement_repo, call_link_repo):
    assert isinstance(call_requirement_repo, CallRequirementRepository)
    assert isinstance(call_link_repo, CallLinkRepository)
    fixing_dates = set()
    for call_id in regenerate_execution_order(dependency_graph_id, call_link_repo):

        # Get the stubbed expression.
        call_requirement = call_requirement_repo[call_id]
        assert isinstance(call_requirement, CallRequirement)
        dsl_expr = dsl_parse(call_requirement.dsl_source)

        # Find all the fixing times involved in this expression.
        for fixing_date in find_fixing_dates(dsl_expr=dsl_expr):
            fixing_dates.add(fixing_date)

    # Return a sorted list of fixing times.
    return sorted(list(fixing_dates))
