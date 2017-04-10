import datetime
import six
from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish
from quantdsl.priceprocess.base import datetime_from_date


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


def register_simulated_price(market_simulation_id, market_name, fixing_date, delivery_date, price_value):
    simulated_price_id = make_simulated_price_id(market_simulation_id, market_name, fixing_date, delivery_date)
    created_event = SimulatedPrice.Created(entity_id=simulated_price_id, value=price_value)
    simulated_price = SimulatedPrice.mutator(event=created_event)
    publish(created_event)
    return simulated_price


def make_simulated_price_id(simulation_id, commodity_name, fixing_date, delivery_date):
    assert isinstance(commodity_name, six.string_types), commodity_name
    assert isinstance(fixing_date, (datetime.datetime, datetime.date)), (fixing_date, type(fixing_date))
    assert isinstance(delivery_date, (datetime.datetime, datetime.date)), (delivery_date, type(delivery_date))
    fixing_date = datetime_from_date(fixing_date)
    delivery_date = datetime_from_date(delivery_date)
    price_id = ("PriceId(simulation_id='{}' commodity_name='{}' fixing_date='{}', delivery_date='{}')"
                "".format(simulation_id, commodity_name, fixing_date, delivery_date))
    return price_id


class SimulatedPriceRepository(EntityRepository):
    pass


