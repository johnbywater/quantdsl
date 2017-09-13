from eventsourcing.application.with_pythonobjects import EventSourcingWithPythonObjects
from quantdsl.application.base import QuantDslApplication


class QuantDslApplicationWithPythonObjects(EventSourcingWithPythonObjects, QuantDslApplication):
    pass
