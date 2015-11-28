from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.call_dependencies import CallDependencies, CallDependenciesRepository


class CallDependenciesRepo(CallDependenciesRepository, EventSourcedRepository):

    domain_class = CallDependencies