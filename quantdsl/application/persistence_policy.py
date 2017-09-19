from eventsourcing.infrastructure.persistence_subscriber import PersistenceSubscriber

from quantdsl.domain.model.call_result import CallResult
from quantdsl.domain.model.simulated_price import SimulatedPrice


class PersistencePolicy(PersistenceSubscriber):
    @staticmethod
    def is_domain_event(event):
        return PersistenceSubscriber.is_domain_event(event) and \
            not isinstance(event, (
                CallResult.Created,
                CallResult.Discarded,
                SimulatedPrice.Created,
            ))