from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository
from quantdsl.domain.model.contract_specification import ContractSpecification, ContractSpecificationRepository


class ContractSpecificationRepo(ContractSpecificationRepository, EventSourcedRepository):

    domain_class = ContractSpecification


