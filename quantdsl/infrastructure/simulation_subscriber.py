from eventsourcing.domain.model.events import subscribe, unsubscribe

from quantdsl.domain.model.market_calibration import MarketCalibrationRepository
from quantdsl.domain.model.market_simulation import MarketSimulation, MarketSimulationRepository
from quantdsl.domain.services.simulated_prices import generate_simulated_prices


class SimulationSubscriber(object):

    def __init__(self, market_calibration_repo, market_simulation_repo):
        assert isinstance(market_calibration_repo, MarketCalibrationRepository)
        assert isinstance(market_simulation_repo, MarketSimulationRepository)
        self.market_calibration_repo = market_calibration_repo
        self.market_simulation_repo = market_simulation_repo
        subscribe(self.market_simulation_created, self.generate_simulated_prices)

    def close(self):
        unsubscribe(self.market_simulation_created, self.generate_simulated_prices)

    def market_simulation_created(self, event):
        return isinstance(event, MarketSimulation.Created)

    def generate_simulated_prices(self, event):
        # When a market simulation is created, generate and register all the simulated prices.
        assert isinstance(event, MarketSimulation.Created)
        market_simulation = self.market_simulation_repo[event.entity_id]
        market_calibration = self.market_calibration_repo[event.market_calibration_id]
        generate_simulated_prices(market_calibration, market_simulation)
