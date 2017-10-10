from threading import Event, Thread
from time import sleep

import six.moves.queue as queue

from quantdsl.application.base import QuantDslApplication
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.exceptions import TimeoutError, DslCompareArgsError, DslBinOpArgsError, DslIfTestExpressionError


class ServiceExit(Exception):
    pass


class QuantDslApplicationWithMultithreading(QuantDslApplication):
    def __init__(self, num_threads=4, *args, **kwargs):
        super(QuantDslApplicationWithMultithreading, self).__init__(call_evaluation_queue=queue.Queue(),
                                                                    *args, **kwargs)
        self.num_threads = num_threads
        self.has_thread_errored = Event()
        self.thread_exception = None
        self.threads = []

        # Start evaluation worker threads.
        for _ in range(self.num_threads):
            t = Thread(target=self.protected_loop_on_evaluation_queue)
            t.setDaemon(True)
            t.daemon = True
            t.start()
            self.threads.append(t)

    def protected_loop_on_evaluation_queue(self):
        try:
            self.loop_on_evaluation_queue()
        except Exception as e:
            if not self.has_thread_errored.is_set():
                self.thread_exception = e
                self.has_thread_errored.set()
            if not isinstance(e, (TimeoutError, DslCompareArgsError, DslBinOpArgsError, DslIfTestExpressionError)):
                raise

    def get_result(self, contract_valuation):
        assert isinstance(contract_valuation, ContractValuation)

        # Todo: Subscribe to call result, with handler that sets an event. Then wait for the
        # event with a timeout, in a while True loop, checking for interruptions and timeouts
        # like in Calculate.calculate().
        while True:
            try:
                return super(QuantDslApplicationWithMultithreading, self).get_result(contract_valuation)
            except KeyError:
                sleep(0.1)
                self.check_has_thread_errored()

    def check_has_thread_errored(self):
        if self.has_thread_errored.is_set():
            raise self.thread_exception
