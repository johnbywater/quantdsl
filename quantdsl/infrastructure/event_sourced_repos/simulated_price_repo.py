from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.simulated_price import SimulatedPrice, Repository


class SimulatedPriceRepo(Repository, EventSourcedRepository):

    domain_class = SimulatedPrice


