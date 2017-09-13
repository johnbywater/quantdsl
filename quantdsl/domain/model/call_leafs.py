from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class CallLeafs(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, leaf_ids, **kwargs):
        super(CallLeafs, self).__init__(**kwargs)
        self._leaf_ids = leaf_ids

    @property
    def leaf_ids(self):
        return self._leaf_ids


def register_call_leafs(contract_specification_id, leaf_ids):
    created_event = CallLeafs.Created(entity_id=contract_specification_id, leaf_ids=leaf_ids)
    call_leafs = CallLeafs.mutator(event=created_event)
    publish(created_event)
    return call_leafs


class CallLeafsRepository(EntityRepository):
    pass
