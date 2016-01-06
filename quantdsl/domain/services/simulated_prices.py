from quantdsl.domain.model.call_link import CallLinkRepository
from quantdsl.domain.model.call_requirement import CallRequirementRepository, CallRequirement
from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.simulated_price import register_simulated_price
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.domain.services.price_processes import get_price_process
from quantdsl.priceprocess.base import PriceProcess
from quantdsl.semantics import find_fixing_dates, find_market_names


def generate_simulated_prices(market_simulation, market_calibration):
    for market_name, fixing_date, price_value in simulate_future_prices(market_simulation, market_calibration):
        register_simulated_price(market_simulation.id, market_name, fixing_date, price_value)


def simulate_future_prices(market_simulation, market_calibration):
    assert isinstance(market_simulation, MarketSimulation), market_simulation
    assert isinstance(market_calibration, MarketCalibration), market_calibration
    price_process = get_price_process(market_calibration.price_process_name)
    assert isinstance(price_process, PriceProcess), price_process
    return price_process.simulate_future_prices(
        market_names=market_simulation.market_names,
        fixing_dates=market_simulation.fixing_dates,
        observation_date=market_simulation.observation_date,
        path_count=market_simulation.path_count,
        calibration_params=market_calibration.calibration_params)


def list_market_names_and_fixing_dates(dependency_graph_id, call_requirement_repo, call_link_repo):
    assert isinstance(call_requirement_repo, CallRequirementRepository)
    assert isinstance(call_link_repo, CallLinkRepository)
    market_names = set()
    fixing_dates = set()
    for call_id in regenerate_execution_order(dependency_graph_id, call_link_repo):

        # Get the stubbed expression.
        call_requirement = call_requirement_repo[call_id]
        assert isinstance(call_requirement, CallRequirement)

        if call_requirement._dsl_expr is not None:
            dsl_expr = call_requirement._dsl_expr
        else:
            dsl_module = dsl_parse(call_requirement.dsl_source)
            dsl_expr = dsl_module.body[0]
            call_requirement._dsl_expr = dsl_expr

        # Find all the fixing times involved in this expression.
        for fixing_date in find_fixing_dates(dsl_expr=dsl_expr):
            fixing_dates.add(fixing_date)

        # Find all the market names involved in this expression.
        for market_name in find_market_names(dsl_expr=dsl_expr):
            market_names.add(market_name)

    # Return sorted lists.
    market_names = sorted(list(market_names))
    fixing_dates = sorted(list(fixing_dates))
    return market_names, fixing_dates