from quantdsl.domain.model.call_dependencies import CallDependencies
from quantdsl.domain.model.call_link import CallLinkRepository
from quantdsl.domain.model.call_requirement import CallRequirementRepository, CallRequirement
from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.perturbation_dependencies import register_perturbation_dependencies
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.domain.model.simulated_price import register_simulated_price
from quantdsl.domain.model.simulated_price_requirements import register_simulated_price_requirements
from quantdsl.domain.services.call_links import regenerate_execution_order
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.domain.services.price_processes import get_price_process
from quantdsl.priceprocess.base import PriceProcess
from quantdsl.semantics import DslObject


def generate_simulated_prices(market_simulation, market_calibration):
    for commodity_name, fixing_date, delivery_date, price_value in simulate_future_prices(market_simulation, market_calibration):
        yield register_simulated_price(market_simulation.id, commodity_name, fixing_date, delivery_date, price_value)


def simulate_future_prices(market_simulation, market_calibration):
    assert isinstance(market_simulation, MarketSimulation), market_simulation
    if not market_simulation.requirements:
        return []
    assert isinstance(market_calibration, MarketCalibration), market_calibration
    price_process = get_price_process(market_calibration.price_process_name)
    assert isinstance(price_process, PriceProcess), price_process
    return price_process.simulate_future_prices(
        observation_date=market_simulation.observation_date,
        requirements=market_simulation.requirements,
        path_count=market_simulation.path_count,
        calibration_params=market_calibration.calibration_params)


def identify_simulation_requirements(contract_specification_id, call_requirement_repo, call_link_repo,
                                     call_dependencies_repo, market_dependencies_repo, observation_date, requirements):
    assert isinstance(call_requirement_repo, CallRequirementRepository)
    assert isinstance(call_link_repo, CallLinkRepository)

    all_perturbation_dependencies = {}

    for call_id in regenerate_execution_order(contract_specification_id, call_link_repo):

        # Get the stubbed expression.
        call_requirement = call_requirement_repo[call_id]
        assert isinstance(call_requirement, CallRequirement)

        if call_requirement._dsl_expr is not None:
            dsl_expr = call_requirement._dsl_expr
        else:
            dsl_module = dsl_parse(call_requirement.dsl_source)
            dsl_expr = dsl_module.body[0]
            call_requirement._dsl_expr = dsl_expr
        assert isinstance(dsl_expr, DslObject), dsl_expr

        # Todo: Consolidate 'date' attributes to be a single element (rather than a possibly long sum expression).

        # Identify this call's requirements for simulated prices.
        simulation_requirements = set()
        present_time = call_requirement.effective_present_time or observation_date
        dsl_expr.identify_price_simulation_requirements(simulation_requirements, present_time=present_time)

        # Register the simulation requirements for each call (needed during evaluation).
        register_simulated_price_requirements(call_id, list(simulation_requirements))

        # Update the simulation requirements (needed for the market simulation).
        requirements.update(simulation_requirements)

        # Identify this call's perturbation dependencies.
        perturbation_dependencies = set()
        dsl_expr.identify_perturbation_dependencies(perturbation_dependencies, present_time=present_time)

        # Add the expression's perturbation dependencies to the perturbation dependencies of its call dependencies.
        call_dependencies = call_dependencies_repo[call_id]
        assert isinstance(call_dependencies, CallDependencies), call_dependencies
        for dependency_id in call_dependencies.dependencies:
            dependency_perturbation_dependencies = all_perturbation_dependencies[dependency_id]
            perturbation_dependencies.update(dependency_perturbation_dependencies)
        # Register the perturbation dependencies in the repo (needed when evaluating the call).
        register_perturbation_dependencies(call_id, list(perturbation_dependencies))

        # Save the perturbation dependencies for this call, so they are available for the dependent calls.
        all_perturbation_dependencies[call_id] = perturbation_dependencies

