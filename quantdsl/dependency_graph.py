class DependencyGraph(object):
    """
    A dependency graph of stubbed expressions.
    """
    def __init__(self, root_stub_id, call_requirements, dependencies, dependents, leaf_ids):
        self.root_stub_id = root_stub_id
        self.leaf_ids = leaf_ids
        self.call_requirements = call_requirements
        self.dependencies = dependencies
        self.dependents = dependents

    def __len__(self):
        return len(self.call_requirements)
