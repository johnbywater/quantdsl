from threading import Thread
from time import sleep

from quantdsl.infrastructure.runners.base import DependencyGraphRunner, evaluate_call, handle_result


class MultiProcessingDependencyGraphRunner(DependencyGraphRunner):

    def __init__(self, dependency_graph, pool_size=None):
        super(MultiProcessingDependencyGraphRunner, self).__init__(dependency_graph)
        self.pool_size = pool_size

    def run(self, **kwds):
        super(MultiProcessingDependencyGraphRunner, self).run(**kwds)
        import multiprocessing
        self.evaluation_pool = multiprocessing.Pool(processes=self.pool_size)
        self.manager = multiprocessing.Manager()
        self.execution_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.results_dict = self.manager.dict()
        self.result_ids = self.manager.dict()
        self.calls_dict = self.manager.dict()
        self.calls_dict.update(self.dependency_graph.call_requirements)
        self.dependencies_by_stub = self.manager.dict()
        self.dependencies_by_stub.update(self.dependency_graph.dependencies_by_stub)
        self.dependents_by_stub = self.manager.dict()
        self.dependents_by_stub.update(self.dependency_graph.dependents_by_stub)
        self.errors = []
        # if 'all_market_prices' in self.run_kwds:
        #     all_market_prices = self.run_kwds.pop('all_market_prices')
        #     all_market_prices_dict = self.manager.dict()
        #     for market_name, market_prices in all_market_prices.items():
        #         all_market_prices_dict[market_name] = market_prices.copy()
        #         # market_prices_dict = self.manager.dict()
        #         # all_market_prices_dict[market_name] = market_prices_dict
        #         # for fixing_date, market_price in market_prices.items():
        #         #     market_prices_dict[fixing_date] = market_price
        #     self.run_kwds['all_market_prices'] = all_market_prices_dict

        evaluation_thread = Thread(target=self.evaluate_calls)
        evaluation_thread.daemon = True
        evaluation_thread.start()

        results_thread = Thread(target=self.handle_results)
        results_thread.daemon = True
        results_thread.start()

        # Put the leaves on the execution queue.
        for call_requirement_id in self.dependency_graph.leaf_ids:
            self.execution_queue.put(call_requirement_id)

        while results_thread.isAlive() and evaluation_thread.isAlive():
            sleep(1)
        if evaluation_thread.isAlive():
            self.execution_queue.put(None)
            evaluation_thread.join()
        if results_thread.isAlive():
            self.result_queue.put(None)
            results_thread.join()

        if self.errors:
            raise self.errors[0]

    def evaluate_calls(self):
        try:
            while True:
                call_requirement_id = self.execution_queue.get()
                if call_requirement_id is None:
                    break
                else:
                    dependency_values = self.get_dependency_values(call_requirement_id)
                    stubbed_expr_str, effective_present_time = self.calls_dict[call_requirement_id]

                    evaluation_kwds = self.get_evaluation_kwds(stubbed_expr_str, effective_present_time)

                    def target():
                        async_result = self.evaluation_pool.apply_async(evaluate_call, (
                            call_requirement_id,
                            evaluation_kwds,
                            dependency_values,
                            self.result_queue,
                            stubbed_expr_str,
                            effective_present_time,
                        ))
                        try:
                            async_result.get()
                        except Exception as error:
                            self.result_queue.put(None)
                            self.execution_queue.put(None)
                            self.errors.append(error)
                    thread = Thread(target=target)
                    thread.daemon = True
                    thread.start()

                    self.call_count += 1
        except:
            self.result_queue.put(None)
            raise

    def handle_results(self):
        try:
            while True:
                result = self.result_queue.get()
                if result is None:
                    break
                else:
                    (call_requirement_id, result_value) = result
                    handle_result(call_requirement_id, result_value, self.results_dict, self.result_ids,
                                  self.dependents_by_stub, self.dependencies_by_stub, self.execution_queue)

                if call_requirement_id == self.dependency_graph.root_stub_id:
                    break
        except:
            self.execution_queue.put(None)
            raise

