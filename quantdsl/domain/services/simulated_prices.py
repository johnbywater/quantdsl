from quantdsl.domain.model.call_link import CallLinkRepository
from quantdsl.domain.model.call_requirement import CallRequirementRepository, CallRequirement
from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.market_dependencies import register_market_dependencies
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


def list_market_names_and_fixing_dates(dependency_graph_id, call_requirement_repo, call_link_repo,
                                       call_dependencies_repo, market_dependencies_repo):
    assert isinstance(call_requirement_repo, CallRequirementRepository)
    assert isinstance(call_link_repo, CallLinkRepository)
    all_market_names = set()
    all_fixing_dates = set()

    cum_market_names_by_call_id = {}

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
            all_fixing_dates.add(fixing_date)

        # Find all the market names involved in this expression.
        expr_market_names = list(find_market_names(dsl_expr=dsl_expr))
        for market_name in expr_market_names:
            all_market_names.add(market_name)

        # Add the expression's market names to the market names of its dependencies.
        cum_market_names = set(expr_market_names)
        for dependency_id in call_dependencies_repo[call_id].dependencies:
            cum_market_names.update(cum_market_names_by_call_id[dependency_id])
        cum_market_names_by_call_id[call_id] = cum_market_names
        register_market_dependencies(call_id, list(cum_market_names))

    # Return sorted lists.
    all_market_names = sorted(list(all_market_names))
    all_fixing_dates = sorted(list(all_fixing_dates))
    return all_market_names, all_fixing_dates
