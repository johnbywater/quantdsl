import datetime
import os
import sys
import unittest
from subprocess import Popen

import scipy
from pytz import utc

from quantdsl.domain.model.dependency_graph import DependencyGraph
from quantdsl.application.main import get_quantdsl_app
from quantdsl.domain.model.call_result import CallResult
from quantdsl.domain.model.simulated_price import register_simulated_price
from quantdsl.domain.services.uuids import create_uuid4
from quantdsl.infrastructure.celery.tasks import celery_evaluate_call, celery_handle_result
from quantdsl.infrastructure.runners.distributed import DistributedDependencyGraphRunner
from quantdsl.services import dsl_compile


class TestDistributedDependencyGraphRunner(unittest.TestCase):

    def _test_evaluate_call(self):
        # Check the example task works directly.
        # - set up the call requirement
        app = get_quantdsl_app()
        call_id = create_uuid4()
        app.register_call_requirement(call_id, '1 + 2', datetime.datetime.now())
        app.register_call_dependencies(call_id, [])
        app.register_call_dependents(call_id, [])

        celery_evaluate_call(call_id)
        call_result = app.call_result_repo[call_id]
        assert isinstance(call_result, CallResult)
        self.assertEqual(call_result.result_value, 3)

    def _test_handle_result(self):
        app = get_quantdsl_app()
        call_id = create_uuid4()
        app.register_call_requirement(call_id, '1 + 2', datetime.datetime.now())
        app.register_call_dependencies(call_id, [])
        app.register_call_dependents(call_id, [])
        celery_handle_result(call_id, 3)
        return

    def _test_distributed_dependency_graph_runner(self):
        # Setup the contract.
        #  - branching function calls
        dsl_source = """
def Swing(starts, ends, underlying, quantity):
    if (quantity == 0) or (starts >= ends):
        0
    else:
        Wait(starts, Choice(
            Swing(starts + TimeDelta('1d'), ends, underlying, quantity - 1) + Fixing(starts, underlying),
            Swing(starts + TimeDelta('1d'), ends, underlying, quantity)
        ))
Swing(Date('2011-01-01'), Date('2011-01-03'), 10, 50)
"""

        # Generate the dependency graph.
        dependency_graph = dsl_compile(dsl_source, is_parallel=True)
        assert isinstance(dependency_graph, DependencyGraph)

        # Remember the number of stubbed exprs - will check it after the evaluation.
        actual_len_stubbed_exprs = len(dependency_graph.call_requirements)

        kwds = {
            'interest_rate': 0,
            'present_time': datetime.datetime(2011, 1, 1, tzinfo=utc),
            'simulation_id': create_uuid4(),
        }
        app = get_quantdsl_app()
        market_simulation = app.register_market_simulation({})

        market_names = ['#1']
        for market_name in market_names:
            # NB Need enough days to cover the date range in the dsl_source.
            for i in range(0, 10):
                dt = datetime.datetime(2011, 1, 1, tzinfo=utc) + datetime.timedelta(1) * i
                value = scipy.array([10] * 2000)
                register_simulated_price(market_simulation.id, market_name, fixing_date=dt)

        # Check we've got a path to the 'celery' command line program (hopefully it's next to this python executable).
        celery_script_path = os.path.join(os.path.dirname(sys.executable), 'celery')
        self.assertTrue(os.path.exists(celery_script_path), celery_script_path)

        # Check the example task returns correct result (this assumes the celery worker IS running).
        # - invoke a celery worker process as a subprocess
        worker_cmd = [celery_script_path, 'worker', '-A', 'quantdsl.infrastructure.celery.tasks', '-l', 'info']
        # - its usage as a context manager causes a wait for it to finish after it has been terminated
        with Popen(worker_cmd) as worker:
            try:
                # Evaluate the dependency graph.
                runner = DistributedDependencyGraphRunner(dependency_graph, app=app)
                dsl_value = runner.evaluate(**kwds)

                # Get the mean of the value, if it has one.
                if isinstance(dsl_value, scipy.ndarray):
                    dsl_value = dsl_value.mean()

                # Check the value is expected.
                expected_value = 20
                self.assertEqual(dsl_value, expected_value)

                # Check the number of stubbed exprs is expected.
                expected_len_stubbed_exprs = 7
                self.assertEqual(actual_len_stubbed_exprs, expected_len_stubbed_exprs)

            finally:
                # Shutdown the celery worker.
                worker.terminate()
