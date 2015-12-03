from quantdsl.domain.model.contract_specification import ContractSpecification
from quantdsl.semantics import find_market_names
from quantdsl.domain.services.parser import dsl_parse


def list_market_names(contract_specification):
    assert isinstance(contract_specification, ContractSpecification)
    dsl_module = dsl_parse(contract_specification.specification)
    return list(find_market_names(dsl_expr=dsl_module))
