from eventsourcing.domain.model.events import subscribe, unsubscribe
from quantdsl.domain.model.call_dependencies import CallDependenciesRepository
from quantdsl.domain.model.call_dependents import CallDependentsRepository
from quantdsl.domain.model.contract_specification import ContractSpecificationRepository, ContractSpecification
from quantdsl.domain.services.dependency_graphs import generate_dependency_graph


class DependencyGraphSubscriber(object):

    def __init__(self, contract_specification_repo, call_dependencies_repo, call_dependents_repo, call_leafs_repo,
                 call_requirement_repo):
        assert isinstance(contract_specification_repo, ContractSpecificationRepository)
        assert isinstance(call_dependencies_repo, CallDependenciesRepository)
        assert isinstance(call_dependents_repo, CallDependentsRepository)
        self.contract_specification_repo = contract_specification_repo
        self.call_dependencies_repo = call_dependencies_repo
        self.call_dependents_repo = call_dependents_repo
        self.call_leafs_repo = call_leafs_repo
        self.call_requirement_repo = call_requirement_repo
        subscribe(self.contract_specification_created, self.generate_dependency_graph)

    def close(self):
        unsubscribe(self.contract_specification_created, self.generate_dependency_graph)

    def contract_specification_created(self, event):
        return isinstance(event, ContractSpecification.Created)

    def generate_dependency_graph(self, event):
        assert isinstance(event, ContractSpecification.Created)
        contract_specification = self.contract_specification_repo[event.entity_id]
        generate_dependency_graph(contract_specification, self.call_dependencies_repo, self.call_dependents_repo,
                                  self.call_leafs_repo, self.call_requirement_repo)