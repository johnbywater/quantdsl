from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository
from quantdsl.domain.model.leaf_calls import LeafCalls, LeafCallsRepository


class LeafCallsRepo(LeafCallsRepository, EventSourcedRepository):

    domain_class = LeafCalls
