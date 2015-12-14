import unittest
from tempfile import NamedTemporaryFile

import os

from quantdsl.application.with_multiprocessing_and_sqlalchemy import QuantDslApplicationWithMultiprocessingAndSQLAlchemy
from quantdsl.test_application import ApplicationTestCase, ContractValuationTests


class TestQuantDslApplicationWithMultiprocessingAndSQLAlchemy(ApplicationTestCase, ContractValuationTests):

    def setup_application(self):
        tempfile = NamedTemporaryFile()
        self.tempfile_name = tempfile.name
        tempfile.close()
        self.app = QuantDslApplicationWithMultiprocessingAndSQLAlchemy(
            num_workers=self.NUMBER_WORKERS,
            db_uri='sqlite:///'+self.tempfile_name
        )

    def tearDown(self):
        os.unlink(self.tempfile_name)
        super(TestQuantDslApplicationWithMultiprocessingAndSQLAlchemy, self).tearDown()


if __name__ == '__main__':
    unittest.main()


