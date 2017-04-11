# import six
#
# from quantdsl.infrastructure.runners.base import DependencyGraphRunner, evaluate_call, handle_result
# from quantdsl.domain.services.dependency_graphs import get_dependency_values
# from quantdsl.domain.model.call_specification import CallSpecification
#
#
# class SingleThreadedDependencyGraphRunner(DependencyGraphRunner):
#
#     def run(self, **kwargs):
#         super(SingleThreadedDependencyGraphRunner, self).run(**kwargs)
#         self.call_queue = six.moves.queue.Queue()
#         self.result_queue = six.moves.queue.Queue()
#         self.calls_dict = self.dependency_graph.call_requirements.copy()
#         self.requirements = self.dependency_graph.requirements.copy()
#         self.dependents = self.dependency_graph.dependents.copy()
#         # Put the leaves on the execution queue.
#         for call_requirement_id in self.dependency_graph.leaf_ids:
#             self.call_queue.put(call_requirement_id)
#         # Loop over the required call queue.
#         while not self.call_queue.empty():
#             # Get a waiting call requirement from the queue.
#             call_requirement_id = self.call_queue.get()
#
#             # Get the call attributes.
#             dsl_source, effective_present_time = self.calls_dict[call_requirement_id]
#             evaluation_kwds = self.get_evaluation_kwds(dsl_source, effective_present_time)
#             dependency_values = get_dependency_values(call_requirement_id, self.requirements, self.results_repo)
#
#             # Evaluate the call.
#             call_spec = CallSpecification(
#                 id=call_requirement_id,
#                 dsl_expr_str=dsl_source,
#                 effective_present_time=effective_present_time,
#                 evaluation_kwds=evaluation_kwds,
#                 dependency_values=dependency_values
#             )
#
#             evaluate_call(call_spec, self.result_queue)
#
#             while not self.result_queue.empty():
#                 call_requirement_id, result_value = self.result_queue.get()
#
#                 handle_result(call_requirement_id, result_value, self.results_repo, self.dependents,
#                               self.requirements, self.call_queue)
#
#             self.call_count += 1
