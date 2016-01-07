import datetime

from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.simulated_price import make_simulated_price_id, SimulatedPrice
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME
from quantdsl.test_application import TestCase


class TestMarketSimulation(TestCase):

    NUMBER_MARKETS = 2
    NUMBER_DAYS = 5
    PATH_COUNT = 200

    def test_register_market_simulation(self):
        # Set up the market calibration.
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        calibration_params = {
            '#1-LAST-PRICE': 10,
            '#2-LAST-PRICE': 20,
            '#1-ACTUAL-HISTORICAL-VOLATILITY': 10,
            '#2-ACTUAL-HISTORICAL-VOLATILITY': 20,
            '#1-#2-CORRELATION': 0.5,
        }
        market_calibration = self.app.register_market_calibration(price_process_name, calibration_params)

        # Create a market simulation for a list of markets and fixing times.
        market_names = ['#%d' % (i+1) for i in range(self.NUMBER_MARKETS)]
        date_range = [datetime.date(2011, 1, 1) + datetime.timedelta(days=i) for i in range(self.NUMBER_DAYS)]
        fixing_dates = date_range[1:]
        observation_date = date_range[0]
        path_count = self.PATH_COUNT

        market_simulation = self.app.register_market_simulation(
            market_calibration_id=market_calibration.id,
            market_names=market_names,
            fixing_dates=fixing_dates,
            observation_date=observation_date,
            path_count=path_count,
            interest_rate=2.5,
        )

        assert isinstance(market_simulation, MarketSimulation)
        assert market_simulation.id
        market_simulation = self.app.market_simulation_repo[market_simulation.id]
        assert isinstance(market_simulation, MarketSimulation)
        self.assertEqual(market_simulation.market_calibration_id, market_calibration.id)
        self.assertEqual(market_simulation.market_names, ['#1', '#2'])
        self.assertEqual(market_simulation.fixing_dates, [datetime.date(2011, 1, i) for i in range(2, 6)])
        self.assertEqual(market_simulation.observation_date, datetime.date(2011, 1, 1))
        self.assertEqual(market_simulation.path_count, self.PATH_COUNT)

        # Check there are simulated prices for all markets at all fixing times.
        for market_name in market_names:
            for fixing_date in fixing_dates:
                simulated_price_id = make_simulated_price_id(market_simulation.id, market_name, fixing_date)
                simulated_price = self.app.simulated_price_repo[simulated_price_id]
                self.assertIsInstance(simulated_price, SimulatedPrice)
                self.assertTrue(simulated_price.value.mean())