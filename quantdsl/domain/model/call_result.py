from threading import Lock

from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class CallResult(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, result_value, contract_valuation_id, dependency_graph_id, **kwargs):
        super(CallResult, self).__init__(**kwargs)
        self._result_value = result_value
        self._contract_valuation_id = contract_valuation_id
        self._dependency_graph_id = dependency_graph_id

    @property
    def result_value(self):
        return self._result_value

    @property
    def contract_valuation_id(self):
        return self._contract_valuation_id

    @property
    def dependency_graph_id(self):
        return self._contract_valuation_id

    @property
    def scalar_result_value(self):
        try:
            return self._result_value.mean()
        except AttributeError:
            return self._result_value


def register_call_result(call_id, result_value, contract_valuation_id, dependency_graph_id):
    call_result_id = make_call_result_id(contract_valuation_id, call_id)
    created_event = CallResult.Created(entity_id=call_result_id,
                                       result_value=result_value,
                                       contract_valuation_id=contract_valuation_id,
                                       # Todo: Don't persist this, get the contract valuation object when needed.
                                       # Todo: Also save the list of fixing dates separately (if needs to be saved).
                                       dependency_graph_id=dependency_graph_id,
                                       )
    call_result = CallResult.mutator(event=created_event)

    publish(created_event)
    return call_result


def make_call_result_id(contract_valuation_id, call_id):
    assert call_id
    assert contract_valuation_id
    return contract_valuation_id + call_id


class CallResultRepository(EntityRepository):
    pass