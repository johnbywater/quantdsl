from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository
from quantdsl.domain.model.contract_specification import ContractSpecification, Repository


class ContractSpecificationRepo(Repository, EventSourcedRepository):

    domain_class = ContractSpecification


