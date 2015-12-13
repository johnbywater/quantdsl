import unittest

from cassandra.cqlengine.management import drop_keyspace
from eventsourcing.application.with_cassandra import DEFAULT_CASSANDRA_KEYSPACE

from quantdsl.application.with_multiprocessing_and_cassandra import QuantDslApplicationWithMultiprocessingAndCassandra
from quantdsl.test_application_with_singlethread_and_pythonobjects import ApplicationTestCase


class TestQuantDslApplicationWithMultiprocessingAndCassandra(ApplicationTestCase):

    PATH_COUNT = 4000
    NUMBER_WORKERS = 4

    def tearDown(self):
        drop_keyspace(DEFAULT_CASSANDRA_KEYSPACE)   # Drop keyspace before closing the application.
        super(TestQuantDslApplicationWithMultiprocessingAndCassandra, self).tearDown()

    def setup_application(self):
        # self.app = get_app(num_workers=self.NUMBER_WORKERS)
        self.app = QuantDslApplicationWithMultiprocessingAndCassandra(num_workers=self.NUMBER_WORKERS)

    def test_generate_valuation_swing_option(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-01-05'), Market('NBP'), 3)
"""
        self.assert_contract_value(specification, 30.2051, expected_call_count=15)


if __name__ == '__main__':
    unittest.main()


