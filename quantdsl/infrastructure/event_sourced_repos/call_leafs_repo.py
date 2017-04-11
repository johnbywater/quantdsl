from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.call_leafs import CallLeafs, CallLeafsRepository


class CallLeafsRepo(CallLeafsRepository, EventSourcedRepository):

    domain_class = CallLeafs