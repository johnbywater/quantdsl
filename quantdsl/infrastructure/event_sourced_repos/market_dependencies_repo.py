from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.market_dependencies import MarketDependencies, MarketDependenciesRepository


class MarketDependenciesRepo(MarketDependenciesRepository, EventSourcedRepository):

    domain_class = MarketDependencies