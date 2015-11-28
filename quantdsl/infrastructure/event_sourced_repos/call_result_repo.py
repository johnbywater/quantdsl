from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.call_result import CallResult, CallResultRepository


class CallResultRepo(CallResultRepository, EventSourcedRepository):

    domain_class = CallResult
