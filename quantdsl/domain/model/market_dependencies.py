from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class MarketDependencies(EventSourcedEntity):
    """
    A market dependency is a market that must be evaluated before this call can be evaluated.
    """

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, dependencies, **kwargs):
        super(MarketDependencies, self).__init__(**kwargs)
        self._dependencies = dependencies

    def __getitem__(self, item):
        return self._dependencies.__getitem__(item)

    @property
    def dependencies(self):
        return self._dependencies


def register_market_dependencies(call_requirement_id, dependencies):
    created_event = MarketDependencies.Created(entity_id=call_requirement_id, dependencies=dependencies)
    market_dependencies = MarketDependencies.mutator(event=created_event)
    publish(created_event)
    return market_dependencies


class MarketDependenciesRepository(EntityRepository):
    pass
