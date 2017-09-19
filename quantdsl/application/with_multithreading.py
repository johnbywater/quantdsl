from threading import Thread
import six.moves.queue as queue

from quantdsl.application.base import QuantDslApplication


class QuantDslApplicationWithMultithreading(QuantDslApplication):

    def __init__(self, num_threads=4, *args, **kwargs):
        super(QuantDslApplicationWithMultithreading, self).__init__(call_evaluation_queue=queue.Queue(),
                                                                    *args, **kwargs)
        # Start evaluation worker threads.
        for _ in range(num_threads):
            t = Thread(target=self.loop_on_evaluation_queue)
            t.setDaemon(True)
            t.start()
