from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class LeafCalls(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, call_ids, **kwargs):
        super(LeafCalls, self).__init__(**kwargs)
        self._call_ids = call_ids

    def __getitem__(self, item):
        return self._call_ids.__getitem__(item)

    @property
    def call_ids(self):
        return self._call_ids


def register_leaf_calls(dependency_graph_id, call_ids):
    created_event = LeafCalls.Created(entity_id=dependency_graph_id, call_ids=call_ids)
    leaf_calls = LeafCalls.mutator(event=created_event)
    publish(created_event)
    return leaf_calls


class LeafCallsRepository(EntityRepository):
    pass
