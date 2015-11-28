from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.contract_specification import ContractSpecification
from quantdsl.domain.model.contract_valuation import ContractValuationRepository


class ContractValuationRepo(ContractValuationRepository, EventSourcedRepository):

    domain_class = ContractSpecification


