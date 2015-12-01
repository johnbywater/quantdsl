from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository
from quantdsl.domain.model.market_calibration import MarketCalibration, MarketCalibrationRepository


class MarketCalibrationRepo(MarketCalibrationRepository, EventSourcedRepository):

    domain_class = MarketCalibration
