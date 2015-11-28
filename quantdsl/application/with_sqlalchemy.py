from eventsourcing.application.with_sqlalchemy import EventSourcingWithSQLAlchemy
from quantdsl.application.base import BaseQuantDslApplication


class QuantDslApplicationWithSQLAlchemy(EventSourcingWithSQLAlchemy, BaseQuantDslApplication):

    pass