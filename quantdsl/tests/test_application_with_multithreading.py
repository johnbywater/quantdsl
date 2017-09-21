from unittest.case import TestCase

from quantdsl.application.with_multithreading_and_python_objects import \
    QuantDslApplicationWithMultithreadingAndPythonObjects
from quantdsl.tests.test_application import ExperimentalTests, ExpressionTests, FunctionTests, LongerTests, \
    SingleTests, SpecialTests


class WithMultiThreading(TestCase):
    def setup_application(self, **kwargs):
        self.app = QuantDslApplicationWithMultithreadingAndPythonObjects(**kwargs)


class SingleTestsWithMultithreading(WithMultiThreading, SingleTests):
    pass


class ExperimentalTestsWithMultithreading(WithMultiThreading, ExperimentalTests):
    pass


class SpecialTestsWithMultithreading(WithMultiThreading, SpecialTests):
    pass


class ExpressionTestsWithMultithreading(WithMultiThreading, ExpressionTests):
    pass


class FunctionTestsWithMultithreading(WithMultiThreading, FunctionTests):
    pass


class LongerTestsWithMultithreading(WithMultiThreading, LongerTests):
    pass
