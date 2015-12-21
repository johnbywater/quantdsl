from quantdsl.application.with_greenthreads_and_python_objects import \
    QuantDslApplicationWithGreenThreadsAndPythonObjects
from quantdsl.test_application import ApplicationTestCase, ContractValuationTests


class TestApplicationWithGreenThreadsAndPythonObjects(ApplicationTestCase, ContractValuationTests):

    def setup_application(self):
        self.app = QuantDslApplicationWithGreenThreadsAndPythonObjects()
