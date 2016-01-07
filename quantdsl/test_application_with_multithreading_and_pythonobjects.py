from quantdsl.application.with_multithreading_and_python_objects import \
    QuantDslApplicationWithMultithreadingAndPythonObjects
from quantdsl.test_application import TestCase, ContractValuationTests


class TestApplicationWithPythonObjectsAndMultithreading(TestCase, ContractValuationTests):

    def setup_application(self):
        self.app = QuantDslApplicationWithMultithreadingAndPythonObjects(num_workers=self.NUMBER_WORKERS)
