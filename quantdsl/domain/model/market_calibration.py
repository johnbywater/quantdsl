from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish
from quantdsl.domain.services.create_uuid4 import create_uuid4


class MarketCalibration(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, calibration_params, **kwargs):
        super(MarketCalibration, self).__init__(**kwargs)
        self._calibration_params = calibration_params

    @property
    def calibration_params(self):
        return self._calibration_params


def register_market_calibration(calibration_params):
    created_event = MarketCalibration.Created(entity_id=create_uuid4(), calibration_params=calibration_params)
    call_result = MarketCalibration.mutator(event=created_event)
    publish(created_event)
    return call_result

def compute_market_calibration_params(price_process_name, historical_data):
    # Todo: Generate model params from historical price data.
    return {}


class Repository(EntityRepository):
    pass
