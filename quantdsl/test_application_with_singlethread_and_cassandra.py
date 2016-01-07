from cassandra.cqlengine.management import drop_keyspace
from eventsourcing.infrastructure.stored_events.cassandra_stored_events import create_cassandra_keyspace_and_tables

from quantdsl.application.with_cassandra import QuantDslApplicationWithCassandra, DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE
from quantdsl.test_application import TestCase, ContractValuationTests


class TestQuantDslApplicationWithCassandra(TestCase, ContractValuationTests):

    def setup_application(self):
        self.app = QuantDslApplicationWithCassandra()

        # Create Cassandra keyspace and tables.
        create_cassandra_keyspace_and_tables(DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE)

    def tearDown(self):
        drop_keyspace(DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE)   # Drop keyspace before closing the application.
        super(TestQuantDslApplicationWithCassandra, self).tearDown()
