from eventsourcing.application.with_sqlalchemy import EventSourcingWithSQLAlchemy
from quantdsl.application.base import QuantDslApplication


class QuantDslApplicationWithSQLAlchemy(EventSourcingWithSQLAlchemy, QuantDslApplication):

    pass