from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository
from quantdsl.domain.model.market_simulation import MarketSimulation, MarketSimulationRepository


class MarketSimulationRepo(MarketSimulationRepository, EventSourcedRepository):

    domain_class = MarketSimulation
