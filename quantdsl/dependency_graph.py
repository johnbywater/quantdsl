from quantdsl.exceptions import DslSystemError


class DependencyGraph(object):
    """
    Constructs dependency graph from stack of stubbed expressions.

    The calls are stored in a dict, keyed by ID. The call IDs are
    kept in a list to maintain order. The calls which don't depend
    on any other calls are identified as the 'leafIds'. The calls
    which are dependent on a call are known as 'dependencyIds'. And
    the calls which depend on a call are called 'notifyIds'.
    """
    def __init__(self, root_stub_id, stubbed_exprs_data, call_requirement_ids, call_requirements, dependencies_by_stub,
                 dependents_by_stub, leaf_ids):
        self.root_stub_id = root_stub_id
        self.stubbed_exprs_data = stubbed_exprs_data
        self.leaf_ids = leaf_ids
        self.call_requirement_ids = call_requirement_ids
        self.call_requirements = call_requirements
        self.dependencies_by_stub = dependencies_by_stub
        self.dependents_by_stub = dependents_by_stub

    def __len__(self):
        return len(self.stubbed_exprs_data)

    def has_instances(self, dslType):
        for _, stubbed_expr, _ in self.stubbed_exprs_data:
            if stubbed_expr.has_instances(dslType=dslType):
                return True
        return False

    def find_instances(self, dslType):
        instances = []
        for _, stubbed_expr, _ in self.stubbed_exprs_data:
            [instances.append(i) for i in stubbed_expr.find_instances(dslType=dslType)]
        return instances
