from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_requirement import CallRequirementRepository


class CallRequirementRepo(CallRequirementRepository, EventSourcedRepository):

    domain_class = CallRequirement