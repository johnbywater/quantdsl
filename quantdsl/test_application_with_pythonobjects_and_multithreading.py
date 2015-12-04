from threading import Lock

from quantdsl.test_application_with_pythonobjects_and_singlethread import TestContractValuation

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

class TestApplicationWithPythonObjectsAndMultithreading(TestContractValuation):

    def create_queue_and_lock(self):
        return Queue(), Lock()


