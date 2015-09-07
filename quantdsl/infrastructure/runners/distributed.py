from threading import Thread
from time import sleep
from quantdsl.domain.model import CallSpecification

from quantdsl.infrastructure.runners.base import DependencyGraphRunner, evaluate_call, handle_result
from quantdsl.semantics import CallRequirement


class DistributedDependencyGraphRunner(DependencyGraphRunner):

    def __init__(self, dependency_graph):
        super(DistributedDependencyGraphRunner, self).__init__(dependency_graph)

    def run(self, **kwds):
        super(DistributedDependencyGraphRunner, self).run(**kwds)
