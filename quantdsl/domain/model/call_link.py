from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class CallLink(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, call_id, **kwargs):
        super(CallLink, self).__init__(**kwargs)
        self._call_id = call_id

    @property
    def call_id(self):
        return self._call_id


def register_call_link(link_id, call_id):
    created_event = CallLink.Created(entity_id=link_id, call_id=call_id)
    call_link = CallLink.mutator(event=created_event)
    publish(created_event)
    return call_link


class CallLinkRepository(EntityRepository):
    pass
