import multiprocessing
import gevent
# from gevent import monkey
# monkey.patch_all(thread=False, socket=False)
import gevent.pool
from gevent.queue import JoinableQueue

from quantdsl.application.base import QuantDslApplication


class QuantDslApplicationWithGreenThreads(QuantDslApplication):

    def __init__(self, num_workers=100, *args, **kwargs):
        result_counters = {}
        usage_counters = {}
        call_evaluation_queue = JoinableQueue()

        super(QuantDslApplicationWithGreenThreads, self).__init__(call_evaluation_queue=call_evaluation_queue,
                                                                  result_counters=result_counters,
                                                                  usage_counters=usage_counters, *args, **kwargs)

        call_result_lock = None
        self.compute_pool = multiprocessing.Pool(multiprocessing.cpu_count())

        # Start evaluation worker threads.
        self.workers = []
        self.pool = gevent.pool.Pool(num_workers)
        for _ in range(num_workers):
            self.pool.apply_async(
                    func=self.loop_on_evaluation_queue,
                    args=(call_result_lock, self.compute_pool, result_counters, usage_counters),
            )

    def close(self):
        super(QuantDslApplicationWithGreenThreads, self).close()
        try:
            self.compute_pool.terminate()
        finally:
            self.pool.kill(timeout=1)
