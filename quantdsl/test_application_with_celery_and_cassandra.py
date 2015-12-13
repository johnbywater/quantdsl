import unittest

import os
from cassandra.cqlengine.management import drop_keyspace
from eventsourcing.application.with_cassandra import DEFAULT_CASSANDRA_KEYSPACE

from quantdsl.test_application_with_pythonobjects_and_singlethread import ApplicationTestCase


class TestApplicationWithCassandraAndCelery(ApplicationTestCase):

    PATH_COUNT = 2000
    NUMBER_WORKERS = 4

    def tearDown(self):
        drop_keyspace(DEFAULT_CASSANDRA_KEYSPACE)   # Drop keyspace before closing the application.
        super(TestApplicationWithCassandraAndCelery, self).tearDown()
        # Todo: Stop celery workers.

    def setup_application(self):
        # Todo: Start celery workers.

        # export QUANTDSL_BACKEND='cassandra'
        # celery -A quantdsl.infrastructure.celery.tasks worker --loglevel=info -c4

        os.environ['QUANTDSL_BACKEND'] = 'cassandra'
        from quantdsl.infrastructure.celery.tasks import quantdsl_app
        self.app = quantdsl_app

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

Swing(Date('2011-01-01'), Date('2011-01-2'), Market('NBP'), 2)
# Swing(Date('2011-01-01'), Date('2011-02-12'), Market('NBP'), 20)
"""
        self.assert_contract_value(specification, 61.0401, expected_call_count=4)
        # self.assert_contract_value(specification, 61.0401, expected_call_count=694)

if __name__ == '__main__':
    unittest.main()