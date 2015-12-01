from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.call_dependents import CallDependents, CallDependentsRepository


class CallDependentsRepo(CallDependentsRepository, EventSourcedRepository):

    domain_class = CallDependents