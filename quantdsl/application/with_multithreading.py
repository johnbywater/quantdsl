from quantdsl.domain.services.contract_valuations import get_compute_pool

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from threading import Thread

from quantdsl.application.base import QuantDslApplication


class QuantDslApplicationWithMultithreading(QuantDslApplication):

    def __init__(self, num_workers, *args, **kwargs):
        # Todo: Refactor the counter logic into a class, or find something that already does this. Perhaps use range.
        result_counters = {}
        usage_counters = {}
        super(QuantDslApplicationWithMultithreading, self).__init__(call_evaluation_queue=Queue(),
                                                                    result_counters=result_counters,
                                                                    usage_counters=usage_counters, *args, **kwargs)
        call_result_lock = None
        compute_pool = get_compute_pool()

        # Start evaluation worker threads.
        thread_target = lambda: self.loop_on_evaluation_queue(call_result_lock, compute_pool, result_counters, usage_counters)
        for _ in range(num_workers):
            t = Thread(target=thread_target)
            t.setDaemon(True)
            t.start()
