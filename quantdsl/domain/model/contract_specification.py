from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish
from quantdsl.domain.services.uuids import create_uuid4


class ContractSpecification(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, specification, **kwargs):
        super(ContractSpecification, self).__init__(**kwargs)
        self._specification = specification

    @property
    def specification(self):
        return self._specification


def register_contract_specification(specification):
    created_event = ContractSpecification.Created(entity_id=create_uuid4(), specification=specification)
    contract_specification = ContractSpecification.mutator(event=created_event)
    publish(created_event)
    return contract_specification


# Todo: Rename market_name to commodity_name?


class ContractSpecificationRepository(EntityRepository):
    pass
