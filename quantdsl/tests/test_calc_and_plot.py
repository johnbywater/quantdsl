from unittest.case import TestCase

from eventsourcing.domain.model.events import assert_event_handlers_empty

from quantdsl.exceptions import TimeoutError, CallLimitError
from quantdsl.interfaces.calcandplot import calc_print


class TestCalcPrint(TestCase):
    def setUp(self):
        assert_event_handlers_empty()

    def tearDown(self):
        assert_event_handlers_empty()

    def test_periodisation_monthly(self):

        source_code = """from quantdsl.lib.storage2 import GasStorage
        
GasStorage(Date('2011-6-1'), Date('2011-12-1'), 'GAS', 0, 0, 50000, TimeDelta('1m'), 1)
"""

        results = calc_print(
            source_code=source_code,
            observation_date='2011-1-1',
            interest_rate=2.5,
            periodisation='monthly',
            price_process={
                'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
                'market': ['GAS'],
                'sigma': [0.5],
                'alpha': [0.1],
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
        )

        self.assertAlmostEqual(results.fair_value.mean(), 6, places=0)
        self.assertEqual(len(results.periods), 6)

    def test_periodisation_alltime(self):
        source_code = """from quantdsl.lib.storage2 import GasStorage
        
GasStorage(Date('2011-6-1'), Date('2011-12-1'), 'GAS', 0, 0, 50000, TimeDelta('1m'), 1)
"""

        results = calc_print(
            source_code=source_code,
            observation_date='2011-1-1',
            interest_rate=2.5,
            periodisation='alltime',
            price_process={
                'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
                'market': ['GAS'],
                'sigma': [0.5],
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
        )

        self.assertAlmostEqual(results.fair_value.mean(), 6, places=0)
        self.assertEqual(len(results.periods), 1)

    def test_periodisation_none(self):
        source_code = """from quantdsl.lib.storage2 import GasStorage
        
GasStorage(Date('2011-6-1'), Date('2011-12-1'), 'GAS', 0, 0, 50000, TimeDelta('1m'), 1)
"""

        results = calc_print(
            source_code=source_code,
            observation_date='2011-1-1',
            interest_rate=2.5,
            price_process={
                'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
                'market': ['GAS'],
                'sigma': [0.5],
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
        )

        self.assertAlmostEqual(results.fair_value.mean(), 6, places=0)
        self.assertEqual(len(results.periods), 0)

    def test_timeout(self):
        source_code = """
from quantdsl.lib.storage2 import GasStorage        
GasStorage(Date('2011-1-1'), Date('2011-12-1'), 'GAS', 0, 0, 50000, TimeDelta('1m'), 1)
"""

        with self.assertRaises(SystemExit):
            calc_print(
                source_code=source_code,
                observation_date='2011-1-1',
                interest_rate=2.5,
                price_process={
                    'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
                    'market': ['GAS'],
                    'sigma': [0.5],
                    'curve': {
                        'GAS': (
                            ('2011-1-1', 13.5),
                        )
                    }
                },
                periodisation='monthly',
                timeout=.5,
            )

    def test_dependency_graph_size_limit(self):
        source_code = """
from quantdsl.lib.storage2 import GasStorage        
GasStorage(Date('2011-1-1'), Date('2011-4-1'), 'GAS', 0, 0, 50000, TimeDelta('1m'), 1)
"""

        with self.assertRaises(CallLimitError):
            calc_print(
                source_code=source_code,
                observation_date='2011-1-1',
                interest_rate=2.5,
                price_process={
                    'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
                    'market': ['GAS'],
                    'sigma': [0.5],
                    'curve': {
                        'GAS': (
                            ('2011-1-1', 13.5),
                        )
                    }
                },
                max_dependency_graph_size=1,
            )
