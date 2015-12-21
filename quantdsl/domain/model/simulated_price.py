import datetime
import six
from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class SimulatedPrice(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, value, **kwargs):
        super(SimulatedPrice, self).__init__(**kwargs)
        self._value = value

    @property
    def value(self):
        return self._value


def register_simulated_price(market_simulation_id, market_name, fixing_date, price_value):
    simulated_price_id = make_simulated_price_id(market_simulation_id, market_name, fixing_date)
    created_event = SimulatedPrice.Created(entity_id=simulated_price_id, value=price_value)
    simulated_price = SimulatedPrice.mutator(event=created_event)
    publish(created_event)
    return simulated_price


def make_simulated_price_id(market_simulation_id, market_name, price_time):
    # assert isinstance(market_simulation_id, six.string_types), market_simulation_id
    # assert isinstance(market_name, six.string_types), market_name
    # assert isinstance(price_time, datetime.date), price_time
    return market_simulation_id + market_name + str(price_time)


class SimulatedPriceRepository(EntityRepository):
    pass
