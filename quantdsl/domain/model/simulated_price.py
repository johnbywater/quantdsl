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


def make_simulated_price_id(simulation_id, market_name, quoted_on, delivery_date=''):
    price_id = ("PriceId(simulation_id='{}' market_name='{}' delivery_date='{}' quoted_on='{}')"
                "".format(simulation_id, market_name, quoted_on, delivery_date))
    return price_id


class SimulatedPriceRepository(EntityRepository):
    pass


