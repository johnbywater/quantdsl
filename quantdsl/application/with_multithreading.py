try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from threading import Lock, Thread

from quantdsl.application.base import BaseQuantDslApplication


class QuantDslApplicationWithMultithreading(BaseQuantDslApplication):

    def __init__(self, num_workers, *args, **kwargs):
        super(QuantDslApplicationWithMultithreading, self).__init__(call_evaluation_queue=Queue(), *args, **kwargs)

        # Create a thread lock.
        call_result_lock = Lock()

        # Start evaluation worker threads.
        thread_target = lambda: self.loop_on_evaluation_queue(call_result_lock)
        for _ in range(num_workers):
            t = Thread(target=thread_target)
            t.setDaemon(True)
            t.start()
