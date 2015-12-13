from filelock import FileLock

from quantdsl.application.base import BaseQuantDslApplication
from quantdsl.application.main import get_quantdsl_app
from quantdsl.infrastructure.celery.app import celery_app


class CeleryQueueFacade(object):

    def put(self, item):
        dependency_graph_id, contract_valuation_id, call_id = item
        celery_evaluate_call.delay(dependency_graph_id, contract_valuation_id, call_id)


queue_facade = CeleryQueueFacade()

file_lock = FileLock('/tmp/quantdsl-results-lock')

quantdsl_app = get_quantdsl_app(call_evaluation_queue=queue_facade)
assert isinstance(quantdsl_app, BaseQuantDslApplication)


@celery_app.task
def celery_evaluate_call(dependency_graph_id, contract_valuation_id, call_id):

    quantdsl_app.evaluate_call_and_queue_next_calls(
        contract_valuation_id=contract_valuation_id,
        dependency_graph_id=dependency_graph_id,
        call_id=call_id,
        lock=file_lock,
    )


@celery_app.task
def add(x, y):
    """
    An example task.
    """
    return x + y
