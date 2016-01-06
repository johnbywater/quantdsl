from multiprocessing.pool import Pool
from multiprocessing import Manager

from quantdsl.application.base import QuantDslApplication


class QuantDslApplicationWithMultiprocessing(QuantDslApplication):

    def __init__(self, num_workers=None, call_evaluation_queue=None, **kwargs):
        if num_workers is not None:
            assert call_evaluation_queue is None
            # Parent.
            self.pool = Pool(processes=num_workers)
            self.manager = Manager()
            self.call_evaluation_queue = self.manager.Queue()

        else:
            # Child.
            self.pool = None
        super(QuantDslApplicationWithMultiprocessing, self).__init__(call_evaluation_queue=call_evaluation_queue, **kwargs)

        if self.pool:
            # Start worker pool.
            app_kwargs = self.get_subprocess_application_args()
            args = (self.manager.Lock(), self.__class__, app_kwargs)
            for i in range(num_workers):
                self.pool.apply_async(loop_on_evaluation_queue, args)

    def get_subprocess_application_args(self):
        app_kwargs = dict(
            call_evaluation_queue=self.call_evaluation_queue,
        )
        return app_kwargs

    def close(self):
        super(QuantDslApplicationWithMultiprocessing, self).close()
        if self.pool:
            self.pool.terminate()


def loop_on_evaluation_queue(call_result_lock, application_cls, app_kwargs):
    app = application_cls(**app_kwargs)
    app.loop_on_evaluation_queue(call_result_lock=call_result_lock)
