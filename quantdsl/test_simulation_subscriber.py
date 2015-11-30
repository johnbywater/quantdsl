from eventsourcing.domain.model.events import publish
from twisted.trial import unittest

from quantdsl.domain.model.market_simulation import MarketSimulation


class SimulationSubscriber(object):
    pass


class TestSimulationSubscriber(unittest.TestCase):

    def test_simulation_subscriber(self):

        s = SimulationSubscriber()
        market_simulation_created = MarketSimulation.Created()
        publish(market_simulation_created)
