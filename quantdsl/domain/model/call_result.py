from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class CallResult(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, result_value, **kwargs):
        super(CallResult, self).__init__(**kwargs)
        self._result_value = result_value

    @property
    def result_value(self):
        return self._result_value


def register_call_result(call_id, result_value):
    created_event = CallResult.Created(entity_id=call_id, result_value=result_value)
    call_result = CallResult.mutator(event=created_event)
    publish(created_event)
    return call_result


class CallResultRepository(EntityRepository):
    pass