from unittest.case import TestCase

from eventsourcing.domain.model.events import assert_event_handlers_empty

from quantdsl.interfaces.calcandplot import calc, calc_print_plot


class TestCalcAndPlot(TestCase):
    def setUp(self):
        assert_event_handlers_empty()

    def tearDown(self):
        assert_event_handlers_empty()

    def test(self):

        source_code = """from quantdsl.lib.storage2 import GasStorage
        
GasStorage(Date('2011-6-1'), Date('2011-9-1'), 'GAS', 0, 0, 50000, TimeDelta('1m'), 'monthly')
"""

        calc_print_plot(
            title="Gas Storage",
            source_code=source_code,
            observation_date='2011-1-1',
            interest_rate=2.5,
            path_count=20000,
            perturbation_factor=0.01,
            periodisation='monthly',
            price_process={
                'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
                'market': ['GAS'],
                'sigma': [0.5],
                'alpha': [0.1],
                'rho': [[1.0]],
                'curve': {
                    'GAS': (
                        ('2011-1-1', 13.5),
                        ('2011-2-1', 11.0),
                        ('2011-3-1', 10.0),
                        ('2011-4-1', 9.0),
                        ('2011-5-1', 7.5),
                        ('2011-6-1', 7.0),
                        ('2011-7-1', 6.5),
                        ('2011-8-1', 7.5),
                        ('2011-9-1', 8.5),
                        ('2011-10-1', 10.0),
                        ('2011-11-1', 11.5),
                        ('2011-12-1', 12.0),
                        ('2012-1-1', 13.5),
                        ('2012-2-1', 11.0),
                        ('2012-3-1', 10.0),
                        ('2012-4-1', 9.0),
                        ('2012-5-1', 7.5),
                        ('2012-6-1', 7.0),
                        ('2012-7-1', 6.5),
                        ('2012-8-1', 7.5),
                        ('2012-9-1', 8.5),
                        ('2012-10-1', 10.0),
                        ('2012-11-1', 11.5),
                        ('2012-12-1', 12.0)
                    )
                }
            },
            supress_plot=True,
        )
