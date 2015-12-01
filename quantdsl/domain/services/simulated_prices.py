from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.simulated_price import register_simulated_price
from quantdsl.domain.services.price_processes import get_price_process
from quantdsl.priceprocess.base import PriceProcess


def generate_simulated_prices(market_calibration, market_simulation):
    if market_simulation.market_names and market_simulation.fixing_dates:
        for market_name, fixing_date, price_value in simulate_future_prices(market_simulation, market_calibration):
            register_simulated_price(market_simulation.id, market_name, fixing_date, price_value)


def simulate_future_prices(market_simulation, market_calibration):
    assert isinstance(market_simulation, MarketSimulation), market_simulation
    assert isinstance(market_calibration, MarketCalibration), market_calibration
    price_process = get_price_process(market_calibration.price_process_name)
    assert isinstance(price_process, PriceProcess), price_process
    return price_process.simulate_future_prices(
        market_names=market_simulation.market_names,
        fixing_dates=market_simulation.fixing_dates,
        observation_date=market_simulation.observation_date,
        path_count=market_simulation.path_count,
        calibration_params=market_calibration.calibration_params)
