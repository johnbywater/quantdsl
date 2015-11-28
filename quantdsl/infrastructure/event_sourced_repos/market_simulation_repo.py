from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository
from quantdsl.domain.model.market_simulation import MarketSimulation, Repository


class MarketSimulationRepo(Repository, EventSourcedRepository):

    domain_class = MarketSimulation
