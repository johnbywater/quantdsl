import datetime
from quantdsl.application.main import get_quantdsl_app
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_specification import CallSpecification
from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.infrastructure.celery.app import celery_app
from quantdsl.infrastructure.runners.base import evaluate_call, handle_result
from quantdsl.domain.services.dependency_graphs import get_dependency_values


@celery_app.task
def add(x, y):
    """
    An example task.
    """
    return x + y


# class MarketSimulations(object):
#     """
#     Provides access to market simulations.
#     """
#
#     def __init__(self, simulation_id):
#         self.simulation_id = simulation_id
#
#     def __len__(self):
#         # Todo: Make the validation of the simulation for the evaluation more explicit.
#         return 1
#
#     def __getitem__(self, key):
#         return MarketSimulation(simulation_id=self.simulation_id, name=key)
#
    # def values(self):
    #     return self.data.values()
    #

# class MarketSimulation(object):
#
#     def __init__(self, simulation_id, name):
#         self.simulation_id = simulation_id
#         self.name = name
#
#     def __getitem__(self, dt):
#         assert isinstance(dt, datetime.datetime)
#         app = get_quantdsl_app()
#         return app.simulated_price_repo[make_simulated_price_id(self.simulation_id, self.name, dt)]


@celery_app.task
def celery_evaluate_call(call_id, evaluation_kwds=None):
    if not evaluation_kwds:
        evaluation_kwds = {}

    quantdsl_app = get_quantdsl_app()

    if not 'interest_rate' in evaluation_kwds:
        evaluation_kwds['interest_rate'] = 10

    if not 'first_market_name' in evaluation_kwds:
        evaluation_kwds['first_market_name'] = '#1'

    call_requirement = quantdsl_app.call_requirement_repo[call_id]

    assert isinstance(call_requirement, CallRequirement)

    # if not 'all_market_prices' in evaluation_kwds:
    #     evaluation_kwds['all_market_prices'] = MarketSimulations(simulation_id=call_requirement.simulation_id)

    if not 'present_time' in evaluation_kwds:
        evaluation_kwds['present_time'] = call_requirement.effective_present_time

    if not 'present_time' in evaluation_kwds:
        evaluation_kwds['simulation_id'] = call_requirement.simulation_id

    dependency_values = get_dependency_values(call_id, quantdsl_app.call_dependencies_repo, quantdsl_app.call_result_repo)
    call_spec = CallSpecification(
        id=call_id,
        dsl_expr_str=call_requirement.dsl_source,
        effective_present_time=call_requirement.effective_present_time,
        evaluation_kwds=evaluation_kwds,
        dependency_values=dependency_values,
    )
    evaluate_call(call_spec, result_queue=ResultsQueueAdapter())


@celery_app.task
def celery_handle_result(call_id, result_value):
    quantdsl_app = get_quantdsl_app()
    results = ResultsDictAdapter(quantdsl_app.call_result_repo)
    dependents = quantdsl_app.call_dependents_repo
    dependencies = DependenciesDictAdapter(quantdsl_app.call_dependencies_repo)
    execution_queue = ExecutionQueuePutAdapter()
    handle_result(call_id, result_value, results, dependents, dependencies, execution_queue)


class ResultsQueueAdapter(object):

    def put(self, item):
        call_id = item[0]
        result_value = item[1]
        celery_handle_result(call_id=call_id, result_value=result_value)


class DependenciesDictAdapter(object):

    def __init__(self, call_dependencies_repo):
        self.repo = call_dependencies_repo

    def __contains__(self, item):
        try:
            self.__getitem__(item)
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, item):
        return self.repo[item]


class ResultsDictAdapter(object):

    def __init__(self, call_result_repo):
        self.repo = call_result_repo

    def __contains__(self, key):
        try:
            self.__getitem__(key)
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, key):
        return self.repo[key].result_value

    def __setitem__(self, key, value):
        get_quantdsl_app().register_call_result(call_id=key, result_value=value)

    def __delitem__(self, key):
        self.repo[key].discard()

    def items(self):
        raise Exception("Blah")


class ExecutionQueuePutAdapter(object):

    def put(self, call_id):
        celery_evaluate_call(call_id=call_id)
