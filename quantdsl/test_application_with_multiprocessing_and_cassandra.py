import unittest

from cassandra import OperationTimedOut
from cassandra.cqlengine.management import drop_keyspace
from eventsourcing.infrastructure.stored_events.cassandra_stored_events import create_cassandra_keyspace_and_tables

from quantdsl.application.with_cassandra import DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE
from quantdsl.application.with_multiprocessing_and_cassandra import QuantDslApplicationWithMultiprocessingAndCassandra
from quantdsl.test_application import TestCase, ContractValuationTests


class TestQuantDslApplicationWithMultiprocessingAndCassandra(TestCase, ContractValuationTests):

    def tearDown(self):
        # Drop keyspace before closing the application.
        drop_keyspace(DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE)
        super(TestQuantDslApplicationWithMultiprocessingAndCassandra, self).tearDown()

    def setup_application(self):
        self.app = QuantDslApplicationWithMultiprocessingAndCassandra(num_workers=self.NUMBER_WORKERS)
        try:
            # Create Cassandra keyspace and tables.
            create_cassandra_keyspace_and_tables(DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE)
        except OperationTimedOut:
            try:
                drop_keyspace(DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE)
            except:
                pass
            self.app.close()
            raise



if __name__ == '__main__':
    unittest.main()


