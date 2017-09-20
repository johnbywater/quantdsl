from eventsourcing.domain.model.events import subscribe, unsubscribe

from quantdsl.domain.model.market_calibration import MarketCalibrationRepository
from quantdsl.domain.model.market_simulation import MarketSimulation, MarketSimulationRepository
from quantdsl.domain.model.simulated_price import SimulatedPriceRepository
from quantdsl.domain.services.simulated_prices import generate_simulated_prices


class SimulationSubscriber(object):
    # When a market simulation is created, generate and register all the simulated prices.

    def __init__(self, market_calibration_repo, market_simulation_repo, simulated_price_repo):
        assert isinstance(market_calibration_repo, MarketCalibrationRepository)
        assert isinstance(market_simulation_repo, MarketSimulationRepository)
        # assert isinstance(simulated_price_repo, SimulatedPriceRepository)
        self.market_calibration_repo = market_calibration_repo
        self.market_simulation_repo = market_simulation_repo
        self.simulated_price_repo = simulated_price_repo
        subscribe(self.market_simulation_created, self.generate_simulated_prices_for_market_simulation)

    def close(self):
        unsubscribe(self.market_simulation_created, self.generate_simulated_prices_for_market_simulation)

    def market_simulation_created(self, event):
        return isinstance(event, MarketSimulation.Created)

    def generate_simulated_prices_for_market_simulation(self, event):
        assert isinstance(event, MarketSimulation.Created)
        market_simulation = self.market_simulation_repo[event.entity_id]
        market_calibration = self.market_calibration_repo[event.market_calibration_id]
        for simulated_price in generate_simulated_prices(market_simulation, market_calibration):
            self.simulated_price_repo[simulated_price.id] = simulated_price
