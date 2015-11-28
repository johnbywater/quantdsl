from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.simulated_price import SimulatedPrice, SimulatedPriceRepository


class SimulatedPriceRepo(SimulatedPriceRepository, EventSourcedRepository):

    domain_class = SimulatedPrice


