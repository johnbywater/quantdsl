from multiprocessing import Manager
import unittest
from multiprocessing.pool import Pool

from quantdsl.application.with_cassandra import QuantDslApplicationWithCassandra
from quantdsl.domain.model.call_result import register_call_result
from quantdsl.domain.services.contract_valuations import evaluate_call_requirement, \
    find_dependents_ready_to_be_evaluated
from quantdsl.test_application_with_pythonobjects_and_singlethread import ApplicationTestCase


app = None


def get_app(call_evaluation_queue, call_result_queue):
    global app
    if app is None:
        app = QuantDslApplicationWithCassandra(
            call_evaluation_queue=call_evaluation_queue,
            # call_result_queue=call_result_queue,
        )
    return app


def do_evaluation(call_evaluation_queue, call_result_queue, lock):
    app = get_app(call_evaluation_queue, call_result_queue)
    while True:

        item = call_evaluation_queue.get()

        dependency_graph_id, contract_valuation_id, call_id = item

        # Compute the result.
        result_value = evaluate_call_requirement(
            app.contract_valuation_repo[contract_valuation_id],
            app.call_requirement_repo[call_id],
            app.market_simulation_repo,
            app.call_dependencies_repo,
            app.call_result_repo,
            app.simulated_price_repo
        )

        lock.acquire()
        try:
            register_call_result(
                call_id=call_id,
                result_value=result_value,
                contract_valuation_id=contract_valuation_id,
                dependency_graph_id=dependency_graph_id,
            )

            next_call_ids = find_dependents_ready_to_be_evaluated(
                call_id=call_id,
                call_dependencies_repo=app.call_dependencies_repo,
                call_dependents_repo=app.call_dependents_repo,
                call_result_repo=app.call_result_repo)

            for next_call_id in next_call_ids:
                call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, next_call_id))

        finally:
            lock.release()


def do_result(call_evaluation_queue, call_result_queue):
    app = get_app(call_evaluation_queue, call_result_queue)
    while True:
        item = call_result_queue.get()
        try:
            dependency_graph_id, contract_valuation_id, call_result_id = item
            next_call_ids = find_dependents_ready_to_be_evaluated(
                call_id=call_result_id,
                call_dependencies_repo=app.call_dependencies_repo,
                call_dependents_repo=app.call_dependents_repo,
                call_result_repo=app.call_result_repo)

            for next_call_id in next_call_ids:
                call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, next_call_id))

        finally:
            call_result_queue.task_done()


class TestApplicationWithCassandraAndMultiprocessing(ApplicationTestCase):

    workers = []
    PATH_COUNT = 20000

    def tearDown(self):
        super(TestApplicationWithCassandraAndMultiprocessing, self).tearDown()
        self.pool.terminate()

    def setup_queues_app_and_workers(self):

        num_evaluation_queue_workers = 12

        self.pool = Pool(processes=num_evaluation_queue_workers)
        self.manager = Manager()
        call_evaluation_queue = self.manager.Queue()
        call_result_queue = self.manager.Queue()
        call_result_lock = self.manager.Lock()

        for _ in range(num_evaluation_queue_workers):
            self.pool.apply_async(do_evaluation, (call_evaluation_queue, call_result_queue, call_result_lock))

        self.app = QuantDslApplicationWithCassandra(
            call_evaluation_queue=call_evaluation_queue,
            call_result_queue=call_result_queue,
        )


    def test_generate_valuation_swing_option(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-01-10'), Market('NBP'), 30)
"""
        self.assert_contract_value(specification, 30.2081, expected_call_count=None)

if __name__ == '__main__':
    unittest.main()