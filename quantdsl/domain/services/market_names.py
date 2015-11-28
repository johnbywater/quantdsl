from quantdsl.domain.model.contract_specification import ContractSpecification
from quantdsl.services import dsl_parse, find_market_names


def list_market_names(contract_specification):
    assert isinstance(contract_specification, ContractSpecification)
    dsl_module = dsl_parse(contract_specification.specification)
    return list(find_market_names(dsl_expr=dsl_module))
