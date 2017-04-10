import os
import sys
import unittest
from subprocess import Popen
from tempfile import NamedTemporaryFile

from multiprocessing import cpu_count

from quantdsl.infrastructure.celery.tasks import get_quant_dsl_app_for_celery_worker, \
    close_quant_dsl_app_for_celery_worker
from quantdsl.test_application import ContractValuationTests, TestCase


class TestApplicationWithCeleryAndSQLAlchemy(TestCase, ContractValuationTests):
    skip_assert_event_handers_empty = True

    def setup_application(self):
        # Just use the class app.
        self.app = self._app

    def tearDown(self):
        # Suppress closing the application in the super method.
        self.app = None
        super(TestApplicationWithCeleryAndSQLAlchemy, self).tearDown()

    @classmethod
    def setUpClass(cls):
        super(TestApplicationWithCeleryAndSQLAlchemy, cls).setUpClass()
        cls._setup_application()

    @classmethod
    def _setup_application(cls):
        cls.tmp_file = NamedTemporaryFile(suffix='quantdsl-test.db')
        os.environ['QUANTDSL_BACKEND'] = 'sqlalchemy'
        os.environ['QUANTDSL_DB_URI'] = 'sqlite:///{}'.format(cls.tmp_file.name)
        cls._app = get_quant_dsl_app_for_celery_worker()

        # Check we've got a path to the 'celery' program.
        #  - expect it to be next to the current Python executable
        celery_script_path = os.path.join(os.path.dirname(sys.executable), 'celery')
        assert os.path.exists(celery_script_path), celery_script_path

        # Invoke a celery worker process as a subprocess.
        cls.workers = []
        for i in range(1):
            worker_cmd = [celery_script_path, 'worker', '-A', 'quantdsl.infrastructure.celery.worker',
                          '-P', 'prefork',
                          '-c', str(cpu_count() - 1), '-l', 'info', '-n', 'worker-{}'.format(i)]
            worker = Popen(worker_cmd)
            cls.workers.append(worker)
        
    @classmethod
    def tearDownClass(cls):
        # Shutdown the celery worker.
        workers = getattr(cls, 'workers', [])
        for worker in workers:
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

        # Reset the environment.
        os.environ.pop('QUANTDSL_BACKEND')
        os.environ.pop('QUANTDSL_DB_URI')

        close_quant_dsl_app_for_celery_worker()
        cls._app = None

        super(TestApplicationWithCeleryAndSQLAlchemy, cls).tearDownClass()


if __name__ == '__main__':
    unittest.main()
