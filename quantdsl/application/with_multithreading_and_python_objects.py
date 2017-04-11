from quantdsl.application.with_multithreading import QuantDslApplicationWithMultithreading
from quantdsl.application.with_pythonobjects import QuantDslApplicationWithPythonObjects


class QuantDslApplicationWithMultithreadingAndPythonObjects(QuantDslApplicationWithMultithreading,
                                                            QuantDslApplicationWithPythonObjects):

    pass