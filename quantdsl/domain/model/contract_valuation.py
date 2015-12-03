from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish
from quantdsl.domain.services.uuids import create_uuid4


class ContractValuation(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, market_simulation_id, dependency_graph_id, **kwargs):
        super(ContractValuation, self).__init__(**kwargs)
        self._market_simulation_id = market_simulation_id
        self._dependency_graph_id = dependency_graph_id

    @property
    def market_simulation_id(self):
        return self._market_simulation_id

    @property
    def dependency_graph_id(self):
        return self._dependency_graph_id


def start_contract_valuation(dependency_graph_id, market_simulation_id):
    created_event = ContractValuation.Created(entity_id=create_uuid4(),
                                              market_simulation_id=market_simulation_id,
                                              dependency_graph_id=dependency_graph_id)
    contract_specification = ContractValuation.mutator(event=created_event)
    publish(created_event)
    return contract_specification


class ContractValuationRepository(EntityRepository):
    pass