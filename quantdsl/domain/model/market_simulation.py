from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish
from quantdsl.domain.services.uuids import create_uuid4


class MarketSimulation(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, market_calibration_id, market_names, fixing_dates, observation_date, path_count, **kwargs):
        super(MarketSimulation, self).__init__(**kwargs)
        self._market_calibration_id = market_calibration_id
        self._market_names = market_names
        self._fixing_dates = fixing_dates
        self._observation_date = observation_date
        self._path_count = path_count

    @property
    def market_calibration_id(self):
        return self._market_calibration_id

    @property
    def market_names(self):
        return self._market_names

    @property
    def fixing_dates(self):
        return self._fixing_dates

    @property
    def observation_date(self):
        return self._observation_date

    @property
    def path_count(self):
        return self._path_count


def register_market_simulation(market_calibration_id, market_names, fixing_dates, observation_date, path_count):
    created_event = MarketSimulation.Created(entity_id=create_uuid4(),
                                             market_calibration_id=market_calibration_id,
                                             market_names=market_names,
                                             fixing_dates=fixing_dates,
                                             observation_date=observation_date,
                                             path_count=path_count,
                                             )
    call_result = MarketSimulation.mutator(event=created_event)
    publish(created_event)
    return call_result


class MarketSimulationRepository(EntityRepository):
    pass
