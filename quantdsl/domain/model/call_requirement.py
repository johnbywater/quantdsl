from collections import namedtuple
from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish

StubbedCall = namedtuple('StubbedCall', ['call_id', 'dsl_expr', 'effective_present_time', 'requirements'])


class CallRequirement(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, dsl_source, effective_present_time, **kwargs):
        super(CallRequirement, self).__init__(**kwargs)
        self._dsl_source = dsl_source
        self._effective_present_time = effective_present_time
        self._dsl_expr = None

    @property
    def dsl_source(self):
        return self._dsl_source

    @property
    def effective_present_time(self):
        return self._effective_present_time


def register_call_requirement(call_id, dsl_source, effective_present_time):
    created_event = CallRequirement.Created(
        entity_id=call_id,
        dsl_source=dsl_source,
        effective_present_time=effective_present_time
    )
    call_requirement = CallRequirement.mutator(event=created_event)
    publish(created_event)
    return call_requirement


class CallRequirementRepository(EntityRepository):
    pass
