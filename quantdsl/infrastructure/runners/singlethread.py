import six
from quantdsl.infrastructure.runners.base import DependencyGraphRunner, evaluate_call, handle_result
from quantdsl.semantics import Module


class SingleThreadedDependencyGraphRunner(DependencyGraphRunner):

    def run(self, **kwds):
        super(SingleThreadedDependencyGraphRunner, self).run(**kwds)
        self.call_queue = six.moves.queue.Queue()
        self.result_queue = six.moves.queue.Queue()
        self.result_ids = {}
        self.calls_dict = self.dependency_graph.call_requirements.copy()
        self.dependencies_by_stub = self.dependency_graph.dependencies_by_stub.copy()
        self.dependents_by_stub = self.dependency_graph.dependents_by_stub.copy()
        # Put the leaves on the execution queue.
        for call_requirement_id in self.dependency_graph.leaf_ids:
            self.call_queue.put(call_requirement_id)
        # Loop over the required call queue.
        from quantdsl.services import dsl_parse
        while not self.call_queue.empty():
            call_requirement_id = self.call_queue.get()
            self.call_count += 1
            dependency_values = self.get_dependency_values(call_requirement_id)
            stubbed_expr_str, effective_present_time = self.calls_dict[call_requirement_id]
            stubbed_module = dsl_parse(stubbed_expr_str)

            assert isinstance(stubbed_module, Module)
            evaluation_kwds = self.get_evaluation_kwds(stubbed_expr_str, effective_present_time)
            evaluate_call(call_requirement_id, evaluation_kwds, dependency_values, self.result_queue,
                                    stubbed_expr_str, effective_present_time)
            while not self.result_queue.empty():
                call_requirement_id, result_value = self.result_queue.get()

                handle_result(call_requirement_id, result_value, self.results_dict, self.result_ids, self.dependents_by_stub,
                              self.dependencies_by_stub, self.call_queue)
