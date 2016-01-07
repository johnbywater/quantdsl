from quantdsl.application.base import QuantDslApplication
from quantdsl.infrastructure.celery.tasks import celery_evaluate_call
from quantdsl.infrastructure.runners.base import DependencyGraphRunner

# We can either run the leaf nodes and run dependents that are ready to run,
# or we can reverse the result of topological sort, and run through a series.
# Running leaves first incurs the cost of checking for the existence of a result
# for each of the dependencies of each potentially runnable dependent. It allows for
# parallel evaluation of the nodes, which would allow all runnable nodes to be run
# at the same time. But if each node is executed on a different machine, each would
# require both the results it depends on and the simulation (e.g. market simulation)
# it depends on. That means the cost of checking for runnable nodes would be added
# to the cost of transferring the simulation and results data. If everything was on the
# same machine, there wouldn't be network transfers.
# Running in reverse topological order would avoid dependency checking, all data
# would be local, but dependencies could not be evaluated in parallel, because we
# wouldn't know when to queue the nodes, and queuing them too early (before dependencies
# evaluated) would fail.
# If the dependency graph is obtained before it is persisted, the size of the computation
# will be limited to the size of the dependency graph that will fit in memory. However
# if the dependency graph is persisted as it is generated - by refining the generation
# of the dependency graph to separate generating nodes from storing nodes - then computations
# will be limited to the size of the storage ("infinite").
# In the case where the dependency graph is persisted as it is generated, then evaluation
# of the nodes will incur the cost of retrieving the nodes.
# Similarly for the (e.g. market) simulation: if we are to avoid constraining the size of
# the simulation to the memory rather than the storage, the simulation will need to be
# stored, and the cost of retrieving the simulation from storage will be incurred.
# In other words, to avoid limiting compuation to the size of memory, both the dependency
# graph (and any simulation it uses) will need to be stored, and therefore retrieved.
# Hence, by comparison with evaluation of the dependency graph in series, parallel
# computation incurs only the extra cost of working out which nodes are ready to run,
# so long as we identify that the cost of storing and retrieving the nodes and the
# simulation are associated with avoiding constraining the computation by available memory.

# So the alternatives are:
# 1. either use reverse topological order OR discover runnable nodes after each node finishes
# 2. either put dependency graph nodes etc. in memory OR in a persistence mechanism
# 3. either run in local process with Python queues OR use a message queue and workers

# Because using message queues and workers depends on persisting the data, there are 3 choices:
# 1. in memory data + local processing (serialized or in parallel)
# 2. persisted data + local processing (serialized or in parallel)
# 3. persisted data + distributed processing (serialized or in parallel)
#
# If the data is in memory, the generation of the dependency graph and the simulation need
# to be done just before the evaluation. If the data is persisted, then generating the call
# dependency graph from the DSL expression could also be distributed.
#
# In case we want to avoid limiting the computation to the memory of a single machine, we
# will need to persist the call dependency graph, the simulation, and the results.

class DistributedDependencyGraphRunner(DependencyGraphRunner):

    def __init__(self, dependency_graph, app):
        super(DistributedDependencyGraphRunner, self).__init__(dependency_graph)
        assert isinstance(app, QuantDslApplication)
        self.app = app

    def run(self, **kwargs):
        super(DistributedDependencyGraphRunner, self).run(**kwargs)

        self.app.register_dependency_graph(self.dependency_graph)

        # Enqueue an execution job for each leaf of the dependency graph.
        for call_id in self.dependency_graph.leaf_ids:
            celery_evaluate_call(call_id, evaluation_kwds=kwargs)
