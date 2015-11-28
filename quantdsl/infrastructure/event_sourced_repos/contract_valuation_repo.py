from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.contract_specification import ContractSpecification
from quantdsl.domain.model.contract_valuation import Repository


class ContractValuationRepo(Repository, EventSourcedRepository):

    domain_class = ContractSpecification


