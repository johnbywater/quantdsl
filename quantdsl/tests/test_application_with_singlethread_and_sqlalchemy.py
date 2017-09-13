from quantdsl.application.with_sqlalchemy import QuantDslApplicationWithSQLAlchemy
from quantdsl.tests.test_application import TestCase, ContractValuationTests


class TestQuantDslApplicationWithSQLAlchemy(TestCase, ContractValuationTests):

    def setup_application(self):
        self.app = QuantDslApplicationWithSQLAlchemy(db_uri='sqlite:///:memory:')
