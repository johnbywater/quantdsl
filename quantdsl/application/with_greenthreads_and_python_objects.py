from quantdsl.application.with_greenthreads import QuantDslApplicationWithGreenThreads
from quantdsl.application.with_pythonobjects import QuantDslApplicationWithPythonObjects


class QuantDslApplicationWithGreenThreadsAndPythonObjects(QuantDslApplicationWithGreenThreads,
                                                          QuantDslApplicationWithPythonObjects):
    pass