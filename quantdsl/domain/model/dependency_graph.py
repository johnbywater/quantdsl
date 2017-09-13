from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish
from quantdsl.domain.services.uuids import create_uuid4


class DependencyGraph(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, contract_specification_id, **kwargs):
        super(DependencyGraph, self).__init__(**kwargs)
        self._contract_specification_id = contract_specification_id

    @property
    def contract_specification_id(self):
        return self._contract_specification_id


# def register_dependency_graph(contract_specification_id):
#     created_event = DependencyGraph.Created(entity_id=contract_specification_id, contract_specification_id=contract_specification_id)
#     contract_specification = DependencyGraph.mutator(event=created_event)
#     publish(created_event)
#     return contract_specification


class DependencyGraphRepository(EntityRepository):
    pass
