from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish
from quantdsl.domain.services.uuids import create_uuid4


class ContractValuation(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, market_simulation_id, dependency_graph_id, perturbed_market_name=None, **kwargs):
        super(ContractValuation, self).__init__(**kwargs)
        self._market_simulation_id = market_simulation_id
        self._dependency_graph_id = dependency_graph_id
        self._perturbed_market_name = perturbed_market_name

    @property
    def market_simulation_id(self):
        return self._market_simulation_id

    @property
    def dependency_graph_id(self):
        return self._dependency_graph_id

    @property
    def perturbed_market_name(self):
        return self._perturbed_market_name


def start_contract_valuation(dependency_graph_id, market_simulation_id, perturbed_market_name=None):
    contract_valuation_created = ContractValuation.Created(entity_id=create_uuid4(),
                                                           market_simulation_id=market_simulation_id,
                                                           dependency_graph_id=dependency_graph_id,
                                                           perturbed_market_name=perturbed_market_name)
    contract_valuation = ContractValuation.mutator(event=contract_valuation_created)
    publish(contract_valuation_created)
    return contract_valuation


class ContractValuationRepository(EntityRepository):
    pass
