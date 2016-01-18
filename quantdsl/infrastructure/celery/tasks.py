import logging

from eventsourcing.infrastructure.stored_events.cassandra_stored_events import setup_cassandra_connection, \
    get_cassandra_setup_params
from filelock import FileLock

from quantdsl.application.base import QuantDslApplication
from quantdsl.application.main import get_quantdsl_app
from quantdsl.application.with_cassandra import DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE
from quantdsl.infrastructure.celery.app import celery_app


class CeleryCallEvaluationQueueFacade(object):

    def put(self, item):
        dependency_graph_id, contract_valuation_id, call_id = item
        try:
            celery_evaluate_call.delay(dependency_graph_id, contract_valuation_id, call_id)
            # result = celery_evaluate_call(dependency_graph_id, contract_valuation_id, call_id)
        except OSError as e:
            raise Exception("Celery call failed (is RabbitMQ running?): %s" % e)

_quantdsl_app_singleton = None


def get_quant_dsl_app_for_celery_worker():
    global _quantdsl_app_singleton
    if _quantdsl_app_singleton is None:
        _quantdsl_app_singleton = get_quantdsl_app(call_evaluation_queue=CeleryCallEvaluationQueueFacade())
        # setup_cassandra_connection(*get_cassandra_setup_params(default_keyspace=DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE))

    return _quantdsl_app_singleton


@celery_app.task
def celery_evaluate_call(dependency_graph_id, contract_valuation_id, call_id):

    quantdsl_app = get_quant_dsl_app_for_celery_worker()

    assert isinstance(quantdsl_app, QuantDslApplication)

    quantdsl_app.evaluate_call_and_queue_next_calls(
        contract_valuation_id=contract_valuation_id,
        dependency_graph_id=dependency_graph_id,
        call_id=call_id,
        lock=FileLock('/tmp/quantdsl-results-lock'),
    )


@celery_app.task
def add(x, y):
    """
    An example task.
    """
    return x + y
