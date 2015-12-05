from threading import Lock, Thread

from quantdsl.application.base import BaseQuantDslApplication
from quantdsl.application.with_pythonobjects import QuantDslApplicationWithPythonObjects
from quantdsl.test_application_with_pythonobjects_and_singlethread import TestContractValuation, do_evaluation

try:
    from queue import Queue
except ImportError:
    from Queue import Queue


class TestApplicationWithPythonObjectsAndMultithreading(TestContractValuation):

    NUMBER_WORKERS = 4

    def create_queue_and_lock(self):
        return

    def setup_application(self):
        self.app = QuantDslApplicationWithPythonObjects(
            call_evaluation_queue=Queue()
        )

        def do_singleprocess_evaluation(evaluation_queue, lock):
            app = self.app
            assert isinstance(app, BaseQuantDslApplication)
            app.do_evaluation(lock)

        call_result_lock = Lock()
        for _ in range(self.NUMBER_WORKERS):
            evaluation_queue_worker = Thread(
                target=do_singleprocess_evaluation,
                args=(self.app.call_evaluation_queue, call_result_lock)
            )
            evaluation_queue_worker.setDaemon(True)
            evaluation_queue_worker.start()

