import unittest

from cassandra.cqlengine.management import drop_keyspace
from eventsourcing.infrastructure.stored_events.cassandra_stored_events import create_cassandra_keyspace_and_tables

from quantdsl.application.with_cassandra import DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE
from quantdsl.application.with_multiprocessing_and_cassandra import QuantDslApplicationWithMultiprocessingAndCassandra
from quantdsl.test_application import ApplicationTestCase, ContractValuationTests


class TestQuantDslApplicationWithMultiprocessingAndCassandra(ApplicationTestCase, ContractValuationTests):

    def tearDown(self):
        drop_keyspace(DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE)   # Drop keyspace before closing the application.
        super(TestQuantDslApplicationWithMultiprocessingAndCassandra, self).tearDown()

    def setup_application(self):
        # self.app = get_app(num_workers=self.NUMBER_WORKERS)
        self.app = QuantDslApplicationWithMultiprocessingAndCassandra(num_workers=self.NUMBER_WORKERS)

        # Create Cassandra keyspace and tables.
        create_cassandra_keyspace_and_tables(DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE)


if __name__ == '__main__':
    unittest.main()


