from multiprocessing import Manager
import unittest
from multiprocessing.pool import Pool

from quantdsl.application.base import BaseQuantDslApplication
from quantdsl.application.with_cassandra import QuantDslApplicationWithCassandra
from quantdsl.test_application_with_pythonobjects_and_singlethread import ApplicationTestCase

app = None


def get_app(call_evaluation_queue):
    global app
    if app is None:
        app = QuantDslApplicationWithCassandra(
            call_evaluation_queue=call_evaluation_queue,
        )
    return app


def do_multiprocess_evaluation(call_evaluation_queue, lock):
    app = get_app(call_evaluation_queue)
    assert isinstance(app, BaseQuantDslApplication)
    app.do_evaluation(lock)


class TestApplicationWithCassandraAndMultiprocessing(ApplicationTestCase):

    PATH_COUNT = 4000
    NUMBER_WORKERS = 4

    def tearDown(self):
        super(TestApplicationWithCassandraAndMultiprocessing, self).tearDown()
        self.pool.terminate()

    def setup_application(self):
        self.results = []
        self.pool = Pool(processes=self.NUMBER_WORKERS)
        self.manager = Manager()
        call_evaluation_queue = self.manager.Queue()
        call_result_lock = self.manager.Lock()

        for _ in range(self.NUMBER_WORKERS):
            self.pool.apply_async(do_multiprocess_evaluation, (call_evaluation_queue, call_result_lock))

        self.app = QuantDslApplicationWithCassandra(
            call_evaluation_queue=call_evaluation_queue,
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

Swing(Date('2011-01-01'), Date('2011-01-12'), Market('NBP'), 6)
"""
        self.assert_contract_value(specification, 61.0401, expected_call_count=64)

if __name__ == '__main__':
    unittest.main()