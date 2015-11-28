from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish
from quantdsl.domain.services.create_uuid4 import create_uuid4


class ContractValuation(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, dependency_graph_id, **kwargs):
        super(ContractValuation, self).__init__(**kwargs)
        self._dependency_graph_id = dependency_graph_id


def register_contract_valuation(dependency_graph_id):
    created_event = ContractValuation.Created(entity_id=create_uuid4(), dependency_graph_id=dependency_graph_id)
    contract_specification = ContractValuation.mutator(event=created_event)
    publish(created_event)
    return contract_specification


class Repository(EntityRepository):
    pass
