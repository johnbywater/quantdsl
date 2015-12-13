from quantdsl.application.with_cassandra import QuantDslApplicationWithCassandra
from quantdsl.test_application_with_singlethread_and_pythonobjects import TestContractValuation


class QuantDslApplicationWithSingleThreadAndCassandra(QuantDslApplicationWithCassandra):
    pass


class TestApplicationWithSingleThreadAndCassandra(TestContractValuation):

    NUMBER_WORKERS = 4

    def setup_application(self):
        self.app = QuantDslApplicationWithSingleThreadAndCassandra()


