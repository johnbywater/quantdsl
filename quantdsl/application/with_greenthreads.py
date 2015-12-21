from eventlet import Queue

from quantdsl.application.base import BaseQuantDslApplication


class QuantDslApplicationWithGreenThreads(BaseQuantDslApplication):

    def __init__(self, *args, **kwargs):
        result_counters = {}
        super(QuantDslApplicationWithGreenThreads, self).__init__(call_evaluation_queue=Queue(),
                                                                  result_counters=result_counters, *args, **kwargs)

