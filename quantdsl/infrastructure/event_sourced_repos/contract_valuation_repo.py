from eventsourcing.infrastructure.event_sourced_repo import EventSourcedRepository

from quantdsl.domain.model.contract_valuation import ContractValuationRepository, ContractValuation


class ContractValuationRepo(ContractValuationRepository, EventSourcedRepository):

    domain_class = ContractValuation


