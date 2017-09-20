from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class CallDependencies(EventSourcedEntity):
    """
    A call dependency is a call that must be evaluated before this call can be evaluated.
    """

    class Created(EventSourcedEntity.Created):
        @property
        def dependencies(self):
            return self.__dict__['dependencies']

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, dependencies, **kwargs):
        super(CallDependencies, self).__init__(**kwargs)
        self._dependencies = dependencies

    def __getitem__(self, item):
        return self._dependencies.__getitem__(item)

    @property
    def dependencies(self):
        return self._dependencies


def register_call_dependencies(call_id, dependencies):
    created_event = CallDependencies.Created(entity_id=call_id, dependencies=dependencies)
    call_dependencies = CallDependencies.mutator(event=created_event)
    publish(created_event)
    return call_dependencies


class CallDependenciesRepository(EntityRepository):
    pass
