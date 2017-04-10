# from threading import Thread
# from time import sleep
# from quantdsl.domain.model.call_specification import CallSpecification
#
# from quantdsl.infrastructure.runners.base import DependencyGraphRunner, evaluate_call, handle_result
# from quantdsl.domain.services.dependency_graphs import get_dependency_values
# from quantdsl.domain.model.call_requirement import StubbedCall
#
#
# class MultiProcessingDependencyGraphRunner(DependencyGraphRunner):
#
#     def __init__(self, dependency_graph, pool_size=None):
#         super(MultiProcessingDependencyGraphRunner, self).__init__(dependency_graph)
#         self.pool_size = pool_size
#
#     def run(self, **kwargs):
#         super(MultiProcessingDependencyGraphRunner, self).run(**kwargs)
#         import multiprocessing
#         # Set up pool of evaluation workers.
#         self.evaluation_pool = multiprocessing.Pool(processes=self.pool_size)
#         # Set up a 'shared memory' dependency graph.
#         self.manager = multiprocessing.Manager()
#         self.execution_queue = self.manager.Queue()
#         self.result_queue = self.manager.Queue()
#         self.results_repo = self.manager.dict()
#         self.calls_dict = self.manager.dict()
#         self.calls_dict.update(self.dependency_graph.call_requirements)
#         self.requirements = self.manager.dict()
#         self.requirements.update(self.dependency_graph.requirements)
#         self.dependents = self.manager.dict()
#         self.dependents.update(self.dependency_graph.dependents)
#         self.errors = []
#         # if 'all_market_prices' in self.run_kwds:
#         #     all_market_prices = self.run_kwds.pop('all_market_prices')
#         #     all_market_prices_dict = self.manager.dict()
#         #     for market_name, market_prices in all_market_prices.items():
#         #         all_market_prices_dict[market_name] = market_prices.copy()
#         #         # market_prices_dict = self.manager.dict()
#         #         # all_market_prices_dict[market_name] = market_prices_dict
#         #         # for fixing_date, market_price in market_prices.items():
#         #         #     market_prices_dict[fixing_date] = market_price
#         #     self.run_kwds['all_market_prices'] = all_market_prices_dict
#
#         evaluation_thread = Thread(target=self.evaluate_calls)
#         evaluation_thread.daemon = True
#         evaluation_thread.start()
#
#         results_thread = Thread(target=self.handle_results)
#         results_thread.daemon = True
#         results_thread.start()
#
#         # Put the leaves on the execution queue.
#         for call_requirement_id in self.dependency_graph.leaf_ids:
#             self.execution_queue.put(call_requirement_id)
#
#         while results_thread.isAlive() and evaluation_thread.isAlive():
#             sleep(1)
#         if evaluation_thread.isAlive():
#             self.execution_queue.put(None)
#             evaluation_thread.join()
#         if results_thread.isAlive():
#             self.result_queue.put(None)
#             results_thread.join()
#
#         if self.errors:
#             raise self.errors[0]
#
#     def evaluate_calls(self):
#         try:
#             while True:
#                 call_requirement_id = self.execution_queue.get()
#                 if call_requirement_id is None:
#                     break
#                 else:
#                     call_requirement = self.calls_dict[call_requirement_id]
#                     assert isinstance(call_requirement, StubbedCall)
#                     dsl_source, effective_present_time = call_requirement
#                     evaluation_kwds = self.get_evaluation_kwds(dsl_source, effective_present_time)
#                     dependency_values = get_dependency_values(call_requirement_id, self.requirements, self.results_repo)
#
#                     call_spec = CallSpecification(
#                         id=call_requirement_id,
#                         dsl_expr_str=dsl_source,
#                         effective_present_time=effective_present_time,
#                         evaluation_kwds=evaluation_kwds,
#                         dependency_values=dependency_values,
#                     )
#
#                     def target():
#                         async_result = self.evaluation_pool.apply_async(evaluate_call, (
#                             call_spec,
#                             self.result_queue,
#                         ))
#                         try:
#                             async_result.get()
#                         except Exception as error:
#                             self.result_queue.put(None)
#                             self.execution_queue.put(None)
#                             self.errors.append(error)
#                     thread = Thread(target=target)
#                     thread.daemon = True
#                     thread.start()
#
#                     self.call_count += 1
#         except:
#             self.result_queue.put(None)
#             raise
#
#     def handle_results(self):
#         try:
#             while True:
#                 result = self.result_queue.get()
#                 if result is None:
#                     break
#                 else:
#                     (call_requirement_id, result_value) = result
#                     handle_result(call_requirement_id, result_value, self.results_repo, self.dependents,
#                                   self.requirements, self.execution_queue)
#
#                 if call_requirement_id == self.dependency_graph.root_stub_id:
#                     break
#         except Exception as error:
#             self.execution_queue.put(None)
#             self.errors.append(error)
#             raise
