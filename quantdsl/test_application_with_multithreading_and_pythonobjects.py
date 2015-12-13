from quantdsl.application.with_multithreading_and_python_objects import \
    QuantDslApplicationWithMultithreadingAndPythonObjects
from quantdsl.test_application_with_singlethread_and_pythonobjects import TestContractValuation



class TestApplicationWithPythonObjectsAndMultithreading(TestContractValuation):

    NUMBER_WORKERS = 4

    def setup_application(self):
        self.app = QuantDslApplicationWithMultithreadingAndPythonObjects(num_workers=self.NUMBER_WORKERS)


