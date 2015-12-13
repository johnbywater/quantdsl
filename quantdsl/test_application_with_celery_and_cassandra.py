import unittest

import os

import sys
from subprocess import Popen

from cassandra.cqlengine import CQLEngineException
from cassandra.cqlengine.management import drop_keyspace
from eventsourcing.application.with_cassandra import DEFAULT_CASSANDRA_KEYSPACE
from eventsourcing.infrastructure.stored_events.cassandra_stored_events import create_cassandra_keyspace_and_tables

from quantdsl.application.with_cassandra import DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE
from quantdsl.test_application_with_singlethread_and_pythonobjects import ApplicationTestCase


class TestApplicationWithCassandraAndCelery(ApplicationTestCase):

    PATH_COUNT = 2000
    NUMBER_WORKERS = 4

    def tearDown(self):
        try:
            drop_keyspace(DEFAULT_CASSANDRA_KEYSPACE)   # Drop keyspace before closing the application.
        except CQLEngineException:
            pass
        super(TestApplicationWithCassandraAndCelery, self).tearDown()

        # Shutdown the celery worker.
        # - its usage as a context manager causes a wait for it to finish
        # after it has been terminated, and its stdin and stdout are closed
        with getattr(self, 'worker') as worker:
            if worker is not None:
                worker.terminate()

    def setup_application(self):
        os.environ['QUANTDSL_BACKEND'] = 'cassandra'

        from quantdsl.infrastructure.celery.tasks import get_app_for_celery_task
        self.app = get_app_for_celery_task()

        # Create Cassandra keyspace and tables.
        create_cassandra_keyspace_and_tables(DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE)

        # Check we've got a path to the 'celery' command line program (hopefully it's next to this python executable).
        celery_script_path = os.path.join(os.path.dirname(sys.executable), 'celery')
        self.assertTrue(os.path.exists(celery_script_path), celery_script_path)

        # Check the example task returns correct result (this assumes the celery worker IS running).
        # - invoke a celery worker process as a subprocess
        worker_cmd = [celery_script_path, 'worker', '-A', 'quantdsl.infrastructure.celery.tasks', '-P', 'eventlet', '-c', '1', '-l', 'info']
        self.worker = Popen(worker_cmd)


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
        self.assert_contract_value(specification, 30.2075, expected_call_count=15)

if __name__ == '__main__':
    unittest.main()