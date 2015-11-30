from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.price_process import get_price_process
from quantdsl.priceprocess.base import PriceProcess


def simulate_future_prices(market_simulation, market_calibration):
    assert isinstance(market_calibration, MarketCalibration)
    assert isinstance(market_simulation, MarketSimulation)
    price_process = get_price_process(market_simulation.price_process_name)
    assert isinstance(price_process, PriceProcess), price_process
    return price_process.simulate_future_prices(
        market_names=market_simulation.market_names,
        fixing_dates=market_simulation.fixing_times,
        observation_time=market_simulation.observation_time,
        path_count=market_simulation.path_count,
        calibration_params=market_calibration.calibration_params)