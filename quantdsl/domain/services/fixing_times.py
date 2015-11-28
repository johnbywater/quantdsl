from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.infrastructure.event_sourced_repos.call_requirement_repo import CallRequirementRepo
from quantdsl.services import dsl_parse, find_fixing_times


def fixing_times_from_call_order(call_order, call_requirement_repo):
    fixing_times = set()
    assert isinstance(call_requirement_repo, CallRequirementRepo)
    for call_id in call_order:
        call_requirement = call_requirement_repo[call_id]
        assert isinstance(call_requirement, CallRequirement)
        dsl_expr = dsl_parse(call_requirement.dsl_source)
        for fixing_time in find_fixing_times(dsl_expr=dsl_expr):
            fixing_times.add(fixing_time)
    return sorted(list(fixing_times))