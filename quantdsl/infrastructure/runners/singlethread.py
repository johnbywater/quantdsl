import six
from quantdsl.infrastructure.runners.base import DependencyGraphRunner, evaluate_call, handle_result
from quantdsl.domain.model import CallSpecification


class SingleThreadedDependencyGraphRunner(DependencyGraphRunner):

    def run(self, **kwds):
        super(SingleThreadedDependencyGraphRunner, self).run(**kwds)
        self.call_queue = six.moves.queue.Queue()
        self.result_queue = six.moves.queue.Queue()
        self.calls_dict = self.dependency_graph.call_requirements.copy()
        self.dependencies = self.dependency_graph.dependencies.copy()
        self.dependents = self.dependency_graph.dependents.copy()
        # Put the leaves on the execution queue.
        for call_requirement_id in self.dependency_graph.leaf_ids:
            self.call_queue.put(call_requirement_id)
        # Loop over the required call queue.
        from quantdsl.services import dsl_parse
        while not self.call_queue.empty():
            # Get a waiting call requirement from the queue.
            call_requirement_id = self.call_queue.get()

            # Get the call attributes.
            dsl_source, effective_present_time = self.calls_dict[call_requirement_id]
            evaluation_kwds = self.get_evaluation_kwds(dsl_source, effective_present_time)
            dependency_values = self.get_dependency_values(call_requirement_id)

            # Evaluate the call.
            call_requirement = CallSpecification(
                id=call_requirement_id,
                dsl_expr_str=dsl_source,
                effective_present_time=effective_present_time,
                evaluation_kwds=evaluation_kwds,
                dependency_values=dependency_values
            )

            evaluate_call(call_requirement, self.result_queue)

            while not self.result_queue.empty():
                call_requirement_id, result_value = self.result_queue.get()

                handle_result(call_requirement_id, result_value, self.results_repo, self.dependents,
                              self.dependencies, self.call_queue)

            self.call_count += 1
