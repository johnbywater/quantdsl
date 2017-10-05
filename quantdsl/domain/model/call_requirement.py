from collections import namedtuple

import datetime
from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish

StubbedCall = namedtuple('StubbedCall', ['call_id', 'dsl_expr', 'effective_present_time', 'requirements'])


class CallRequirement(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, dsl_source, effective_present_time, contract_specification_id, cost, **kwargs):
        super(CallRequirement, self).__init__(**kwargs)
        self._dsl_source = dsl_source
        self._effective_present_time = effective_present_time
        self._contract_specification_id = contract_specification_id
        self._dsl_expr = None
        self._cost = cost

    @property
    def dsl_source(self):
        return self._dsl_source

    @property
    def effective_present_time(self):
        return self._effective_present_time

    @property
    def contract_specification_id(self):
        return self._contract_specification_id

    @property
    def cost(self):
        return self._cost


def register_call_requirement(call_id, dsl_source, effective_present_time, contract_specification_id, cost):
    assert isinstance(effective_present_time, (datetime.date, type(None))), effective_present_time
    created_event = CallRequirement.Created(
        entity_id=call_id,
        dsl_source=dsl_source,
        effective_present_time=effective_present_time,
        contract_specification_id=contract_specification_id,
        cost=cost,
    )
    call_requirement = CallRequirement.mutator(event=created_event)
    publish(created_event)
    return call_requirement


class CallRequirementRepository(EntityRepository):
    pass
