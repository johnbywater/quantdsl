from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class CallDependents(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, dependents, **kwargs):
        super(CallDependents, self).__init__(**kwargs)
        self._dependents = dependents

    def __getitem__(self, item):
        return self._dependents.__getitem__(item)

    @property
    def dependents(self):
        return self._dependents


def register_call_dependents(call_id, dependents):
    created_event = CallDependents.Created(entity_id=call_id, dependents=dependents)
    call_dependents = CallDependents.mutator(event=created_event)
    publish(created_event)
    return call_dependents


class CallDependentsRepository(EntityRepository):
    pass
