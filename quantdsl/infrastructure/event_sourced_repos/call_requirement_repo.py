from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.dependency_graph import Repository


class CallRequirementRepo(Repository, EventSourcedRepository):

    domain_class = CallRequirement