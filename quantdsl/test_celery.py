import os
from subprocess import Popen
import unittest
from celery.result import AsyncResult
import sys
from quantdsl.infrastructure.celery.tasks import add


class TestCeleryTasks(unittest.TestCase):

    def test_add(self):
        # Check the example task works directly.
        self.assertEqual(add(1, 2), 3)

        # Check the example task can be enqueued.
        self.assertIsInstance(add.delay(1, 2), AsyncResult)

        # Check the example task times out (this assumes the celery worker is NOT running).
        self.assertRaises(Exception, add.delay(1, 2).get, timeout=1)

        # Check we've got a path to the 'celery' command line program (hopefully it's next to this python executable).
        celery_script_path = os.path.join(os.path.dirname(sys.executable), 'celery')
        self.assertTrue(os.path.exists(celery_script_path), celery_script_path)

        # Check the example task returns correct result (this assumes the celery worker IS running).
        # - invoke a celery worker process as a subprocess
        worker_cmd = [celery_script_path, 'worker', '-A', 'quantdsl.infrastructure.celery.tasks', '-l', 'info']
        # - its usage as a context manager causes a wait for it to finish after it has been terminated
        with Popen(worker_cmd) as worker:
            try:
                # Check the example task works
                self.assertEqual(add.delay(1, 2).get(timeout=10), 3)
            finally:
                # Shutdown the celery worker.
                worker.terminate()

