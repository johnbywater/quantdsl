import unittest
import os
import sys
from subprocess import Popen
from tempfile import NamedTemporaryFile

from cassandra.cqlengine.management import drop_keyspace
from eventsourcing.domain.model.events import assert_event_handlers_empty

from eventsourcing.infrastructure.stored_events.cassandra_stored_events import create_cassandra_keyspace_and_tables

from quantdsl.application.main import get_quantdsl_app
from quantdsl.infrastructure.celery.tasks import CeleryCallEvaluationQueueFacade, get_quant_dsl_app_for_celery_worker
from quantdsl.application.with_cassandra import DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE
from quantdsl.test_application import TestCase, ContractValuationTests


class TestApplicationWithCeleryAndSQLAlchemy(TestCase, ContractValuationTests):

    skip_assert_event_handers_empty = True  # Do it in setup/teardown class.

    @classmethod
    def setUpClass(cls):
        # Check it's a clean start.
        assert_event_handlers_empty()

        # Set up the application and the workers for the class, not each test, otherwise they drag.
        os.environ['QUANTDSL_BACKEND'] = 'sqlalchemy'
        cls.tmp_file = NamedTemporaryFile(suffix='quantdsl-test.db')

        os.environ['QUANTDSL_DB_URI'] = 'sqlite:///{}'.format(cls.tmp_file.name)
        cls._app = get_quant_dsl_app_for_celery_worker()

        # Check we've got a path to the 'celery' program.
        #  - expect it to be next to the current Python executable
        celery_script_path = os.path.join(os.path.dirname(sys.executable), 'celery')
        assert os.path.exists(celery_script_path), celery_script_path

        # Invoke a celery worker process as a subprocess - it is terminates at the end of this test case.
        worker_cmd = [celery_script_path, 'worker', '-A', 'quantdsl.infrastructure.celery.worker',
                      '-P', 'prefork',
                      '-c', str(1), '-l', 'info']
        cls.worker = Popen(worker_cmd)

    @classmethod
    def tearDownClass(cls):
        # Shutdown the celery worker.
        worker = getattr(cls, 'worker', None)
        if worker is not None:
            assert isinstance(worker, Popen)
            if hasattr(worker, '__exit__'):
                # Python3
                with worker:
                    worker.terminate()
            else:
                # Python2
                worker.terminate()
                worker.wait()
            cls.worker = None

        # Close the application.
        cls._app.close()

        # Remove the temporary file.
        if cls.tmp_file and cls.tmp_file.name:
            os.remove(cls.tmp_file.name)

        # Reset the environment.
        os.environ.pop('QUANTDSL_BACKEND')

        # Check everything is unsubscribed.
        assert_event_handlers_empty()

    def setup_application(self):
        # Make cls._app available as self.app, as expected by the test methods.
        self.app = self._app

    def tearDown(self):
        # Prevent the app being closed at the end of each test by super method.
        self.app = None
        super(TestApplicationWithCeleryAndSQLAlchemy, self).tearDown()


if __name__ == '__main__':
    unittest.main()
