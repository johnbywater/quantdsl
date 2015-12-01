from eventsourcing.application.with_cassandra import EventSourcingWithCassandra
from quantdsl.application.base import BaseQuantDslApplication


class QuantDslApplicationWithCassandra(EventSourcingWithCassandra, BaseQuantDslApplication):

    pass