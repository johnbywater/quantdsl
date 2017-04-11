import gevent

from quantdsl.application.with_greenthreads_and_python_objects import \
    QuantDslApplicationWithGreenThreadsAndPythonObjects
from quantdsl.test_application import TestCase, ContractValuationTests


class TestApplicationWithGreenThreadsAndPythonObjects(TestCase, ContractValuationTests):

    def setup_application(self):
        self.app = QuantDslApplicationWithGreenThreadsAndPythonObjects()

    def sleep(self, interval):
        gevent.sleep(interval)
