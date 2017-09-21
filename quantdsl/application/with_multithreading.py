from threading import Thread
import six.moves.queue as queue

from quantdsl.application.base import QuantDslApplication
from quantdsl.exceptions import TimeoutError


class QuantDslApplicationWithMultithreading(QuantDslApplication):

    def __init__(self, num_threads=4, *args, **kwargs):
        super(QuantDslApplicationWithMultithreading, self).__init__(call_evaluation_queue=queue.Queue(),
                                                                    *args, **kwargs)
        # Start evaluation worker threads.
        for _ in range(num_threads):
            t = Thread(target=self.protected_loop_on_evaluation_queue)
            t.setDaemon(True)
            t.start()

    def protected_loop_on_evaluation_queue(self):
        try:
            self.loop_on_evaluation_queue()
        except TimeoutError:
            # Don't need to see unhandled timeout exceptions from worker threads.
            pass