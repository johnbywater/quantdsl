import scipy
from eventsourcing.domain.model.entity import EntityRepository, EventSourcedEntity
from eventsourcing.domain.model.events import publish


class CallResult(EventSourcedEntity):
    class Created(EventSourcedEntity.Created):
        @property
        def call_id(self):
            return self.__dict__['call_id']

        @property
        def contract_valuation_id(self):
            return self.__dict__['contract_valuation_id']

        @property
        def contract_specification_id(self):
            return self.__dict__['contract_specification_id']

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, result_value, perturbed_values, contract_valuation_id, call_id, contract_specification_id,
                 **kwargs):
        super(CallResult, self).__init__(**kwargs)
        self._result_value = result_value
        self._perturbed_values = perturbed_values
        self._contract_valuation_id = contract_valuation_id
        self._call_id = call_id
        self._contract_specification_id = contract_specification_id

    @property
    def result_value(self):
        return self._result_value

    @property
    def perturbed_values(self):
        return self._perturbed_values


def register_call_result(call_id, result_value, perturbed_values, contract_valuation_id, contract_specification_id):
    call_result_id = make_call_result_id(contract_valuation_id, call_id)
    created_event = CallResult.Created(entity_id=call_result_id,
                                       result_value=result_value,
                                       perturbed_values=perturbed_values,
                                       contract_valuation_id=contract_valuation_id,
                                       call_id=call_id,
                                       # Todo: Don't persist this, get the contract valuation object when needed.
                                       # Todo: Also save the list of fixing dates separately (if needs to be saved).
                                       contract_specification_id=contract_specification_id,
                                       )
    call_result = CallResult.mutator(event=created_event)

    publish(created_event)
    return call_result


def make_call_result_id(contract_valuation_id, call_id):
    assert contract_valuation_id, contract_valuation_id
    assert call_id, call_id
    return contract_valuation_id + call_id


class CallResultRepository(EntityRepository):
    pass


class ResultValueComputed(object):
    """Event published when a result value is computed.

    Used to track progress of computation more smoothly than
    can be achieved by listening for CallResult.Created events.

    (This is not a persisted domain event.)
    """
