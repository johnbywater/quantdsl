from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository
from quantdsl.domain.model.call_link import CallLink, CallLinkRepository


class CallLinkRepo(CallLinkRepository, EventSourcedRepository):

    domain_class = CallLink
