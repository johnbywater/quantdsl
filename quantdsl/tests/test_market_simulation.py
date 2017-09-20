import datetime

from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.simulated_price import make_simulated_price_id, SimulatedPrice
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME
from quantdsl.tests.test_application import TestCase


class TestMarketSimulation(TestCase):

    NUMBER_MARKETS = 2
    NUMBER_DAYS = 5
    PATH_COUNT = 200

    def test_register_market_simulation(self):
        # Set up the market calibration.
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        calibration_params = {
            'market': ['#1', '#2'],
            'sigma': [0.1, 0.2],
            'curve': {
                '#1': [
                    ('2011-1-1', 10),
                ],
                '#2': [
                    ('2011-1-1', 20),
                ],
            },
            'rho': [
                [1.0, 0.5],
                [0.5, 1.0],
            ],
        }
        market_calibration = self.app.register_market_calibration(price_process_name, calibration_params)

        # Create a market simulation for a list of markets and fixing times.
        commodity_names = ['#%d' % (i+1) for i in range(self.NUMBER_MARKETS)]
        fixing_dates = [datetime.datetime(2011, 1, 1) + datetime.timedelta(days=i) for i in range(self.NUMBER_DAYS)]
        observation_date = fixing_dates[0]
        simulation_requirements = []
        for commodity_name in commodity_names:
            for fixing_date in fixing_dates:
                simulation_requirements.append((commodity_name, fixing_date, fixing_date))
        path_count = self.PATH_COUNT

        market_simulation = self.app.register_market_simulation(
            market_calibration_id=market_calibration.id,
            requirements=simulation_requirements,
            observation_date=observation_date,
            path_count=path_count,
            interest_rate=2.5,
        )

        assert isinstance(market_simulation, MarketSimulation)
        assert market_simulation.id
        market_simulation = self.app.market_simulation_repo[market_simulation.id]
        assert isinstance(market_simulation, MarketSimulation)
        self.assertEqual(market_simulation.market_calibration_id, market_calibration.id)
        # self.assertEqual(market_simulation.requirements[0], ['#1', '#2'])
        # self.assertEqual(market_simulation.fixing_dates, [datetime.date(2011, 1, i) for i in range(2, 6)])
        self.assertEqual(market_simulation.observation_date, datetime.datetime(2011, 1, 1))
        self.assertEqual(market_simulation.path_count, self.PATH_COUNT)

        # Check there are simulated prices for all the requirements.
        for requirement in simulation_requirements:
            commodity_name = requirement[0]
            fixing_date = requirement[1]
            delivery_date = requirement[2]
            simulated_price_id = make_simulated_price_id(market_simulation.id, commodity_name, fixing_date, delivery_date)
            simulated_price = self.app.simulated_price_repo[simulated_price_id]
            self.assertIsInstance(simulated_price, SimulatedPrice)
            self.assertTrue(simulated_price.value.mean())