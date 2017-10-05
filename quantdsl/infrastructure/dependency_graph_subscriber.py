from collections import defaultdict
from eventsourcing.domain.model.events import subscribe, unsubscribe
from quantdsl.domain.model.call_dependencies import CallDependenciesRepository
from quantdsl.domain.model.call_dependents import CallDependentsRepository
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.contract_specification import ContractSpecificationRepository, ContractSpecification
from quantdsl.domain.services.dependency_graphs import generate_dependency_graph
from quantdsl.exceptions import CallLimitError, RecursionDepthError


class DependencyGraphSubscriber(object):

    def __init__(self, contract_specification_repo, call_dependencies_repo, call_dependents_repo, call_leafs_repo,
                 call_requirement_repo, max_dependency_graph_size, dsl_classes):
        assert isinstance(contract_specification_repo, ContractSpecificationRepository)
        assert isinstance(call_dependencies_repo, CallDependenciesRepository)
        assert isinstance(call_dependents_repo, CallDependentsRepository)
        self.contract_specification_repo = contract_specification_repo
        self.call_dependencies_repo = call_dependencies_repo
        self.call_dependents_repo = call_dependents_repo
        self.call_leafs_repo = call_leafs_repo
        self.call_requirement_repo = call_requirement_repo
        subscribe(self.contract_specification_created, self.generate_dependency_graph)
        subscribe(self.call_requirement_created, self.limit_calls)
        self.total_calls = defaultdict(int)
        self.max_dependency_graph_size = max_dependency_graph_size
        self.dsl_classes = dsl_classes

    def close(self):
        unsubscribe(self.contract_specification_created, self.generate_dependency_graph)
        unsubscribe(self.call_requirement_created, self.limit_calls)

    def contract_specification_created(self, event):
        return isinstance(event, ContractSpecification.Created)

    def call_requirement_created(self, event):
        return isinstance(event, CallRequirement.Created)

    def limit_calls(self, event):
        if self.max_dependency_graph_size:
            contract_specification_id = event.contract_specification_id
            self.total_calls[contract_specification_id] += 1
            if self.total_calls[contract_specification_id] > self.max_dependency_graph_size:
                raise CallLimitError('maximum dependency graph size ({}) exceeded'.format(
                    self.max_dependency_graph_size))

    def generate_dependency_graph(self, event):
        assert isinstance(event, ContractSpecification.Created)
        contract_specification = self.contract_specification_repo[event.entity_id]
        try:
            generate_dependency_graph(contract_specification, self.call_dependencies_repo, self.call_dependents_repo,
                                      self.call_requirement_repo, dsl_classes=self.dsl_classes)
        except RuntimeError as e:
            if 'maximum recursion depth exceeded' in str(e):
                raise RecursionDepthError('maximum recursion depth exceeded')
            else:
                raise
