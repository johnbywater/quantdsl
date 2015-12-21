from quantdsl.domain.services.contract_valuations import get_compute_pool

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from threading import Lock, Thread

from quantdsl.application.base import BaseQuantDslApplication


class QuantDslApplicationWithMultithreading(BaseQuantDslApplication):

    def __init__(self, num_workers, *args, **kwargs):
        result_counters = {}
        super(QuantDslApplicationWithMultithreading, self).__init__(call_evaluation_queue=Queue(),
                                                                    result_counters=result_counters, *args, **kwargs)

        # Create a thread lock.
        # call_result_lock = Lock()
        call_result_lock = None

        # compute_pool = get_compute_pool()
        compute_pool = None

        # Start evaluation worker threads.
        thread_target = lambda: self.loop_on_evaluation_queue(call_result_lock, compute_pool, result_counters)
        for _ in range(num_workers):
            t = Thread(target=thread_target)
            t.setDaemon(True)
            t.start()
