from abc import ABCMeta
from threading import Thread
from time import sleep

import six

from quantdsl.exceptions import DslSyntaxError, DslSystemError
from quantdsl.semantics import Module, DslNamespace, DslExpression, Stub


class DependencyGraph(object):
    """
    Constructs dependency graph from stack of stubbed expressions.

    The calls are stored in a dict, keyed by ID. The call IDs are
    kept in a list to maintain order. The calls which don't depend
    on any other calls are identified as the 'leafIds'. The calls
    which are dependent on a call are known as 'dependencyIds'. And
    the calls which depend on a call are called 'notifyIds'.
    """
    def __init__(self, root_stub_id, stubbed_exprs_data):
        self.root_stub_id = root_stub_id
        assert isinstance(stubbed_exprs_data, list)
        assert len(stubbed_exprs_data), "Stubbed expressions is empty!"
        self.stubbed_exprs_data = stubbed_exprs_data
        self.leaf_ids = []
        self.call_requirement_ids = []
        self.call_requirements = {}
        self.dependency_ids = {}
        self.notify_ids = {self.root_stub_id: []}
        for stub_id, stubbed_expr, effective_present_time in self.stubbed_exprs_data:

            assert isinstance(stubbed_expr, DslExpression)

            # Finding stub instances reveals the dependency graph.
            required_stub_ids = [s.name for s in stubbed_expr.find_instances(Stub)]

            self.dependency_ids[stub_id] = required_stub_ids

            if len(required_stub_ids):
                # Each required stub needs to notify its dependents.
                for required_stub_id in required_stub_ids:
                    if required_stub_id not in self.notify_ids:
                        notify_ids = []
                        self.notify_ids[required_stub_id] = notify_ids
                    else:
                        notify_ids = self.notify_ids[required_stub_id]
                    if stub_id in notify_ids:
                        raise DslSystemError("Stub ID already in dependents of required stub. Probably wrong?")
                    notify_ids.append(stub_id)
            else:
                # Keep a track of the leaves of the dependency graph (stubbed exprs that don't depend on anything).
                self.leaf_ids.append(stub_id)

            # Stubbed expr has names that need to be replaced with results of other stubbed exprs.
            stubbedExprStr = str(stubbed_expr)

            self.call_requirements[stub_id] = (stubbedExprStr, effective_present_time)
            self.call_requirement_ids.append(stub_id)

        # Sanity check.
        assert self.root_stub_id in self.call_requirement_ids

    def __len__(self):
        return len(self.stubbed_exprs_data)

    def has_instances(self, dslType):
        for _, stubbedExpr, _ in self.stubbed_exprs_data:
            if stubbedExpr.has_instances(dslType=dslType):
                return True
        return False

    def find_instances(self, dslType):
        instances = []
        for _, stubbedExpr, _ in self.stubbed_exprs_data:
            [instances.append(i) for i in stubbedExpr.find_instances(dslType=dslType)]
        return instances

    def evaluate(self, dependency_graph_runner_class=None, pool_size=None, **kwds):
        # Make sure we've got a dependency graph runner.
        if dependency_graph_runner_class is None:
            dependency_graph_runner_class = SingleThreadedDependencyGraphRunner
        assert issubclass(dependency_graph_runner_class, DependencyGraphRunner)

        # Init and run the dependency graph runner.
        if pool_size:
            runner = dependency_graph_runner_class(self, pool_size=pool_size)
        else:
            runner = dependency_graph_runner_class(self)
        assert isinstance(runner, DependencyGraphRunner)
        runner.run(**kwds)
        self.runner = runner

        try:
            return self.runner.results_dict[self.root_stub_id]
        except KeyError:
            errorData = (self.root_stub_id, self.runner.results_dict.keys())
            raise DslSystemError("root value not found", str(errorData))


class DependencyGraphRunner(object):

    __metaclass__ = ABCMeta

    def __init__(self, dependency_graph):
        assert isinstance(dependency_graph, DependencyGraph)
        self.dependency_graph = dependency_graph

    def run(self, **kwds):
        self.run_kwds = kwds
        self.call_count = 0
        self.results_dict = {}
        self.dependency_dict = {}

    def get_dependency_values(self, call_requirement_id):
        dependency_values = {}
        dependency_stub_ids = self.dependency_dict[call_requirement_id]
        for stub_id in dependency_stub_ids:
            try:
                stub_result = self.results_dict[stub_id]
            except KeyError:
                keys = self.results_dict.keys()
                raise KeyError("{} not in {} (really? {})".format(stub_id, keys, stub_id not in keys))
            else:
                dependency_values[stub_id] = stub_result
        return dependency_values

    def get_evaluation_kwds(self, stubbed_expr_str, effective_present_time):
        evaluation_kwds = self.run_kwds.copy()

        from quantdsl.services import parse, get_market_names, get_fixing_dates
        stubbed_module = parse(stubbed_expr_str)
        assert isinstance(stubbed_module, Module)
        # market_names = get_market_names(stubbed_module)
        fixing_dates = get_fixing_dates(stubbed_module)
        if effective_present_time is not None:
            fixing_dates.append(effective_present_time)

        # return evaluation_kwds
        if 'all_market_prices' in evaluation_kwds:
            all_market_prices = evaluation_kwds.pop('all_market_prices')
            evaluation_kwds['all_market_prices'] = dict()
            for market_name in all_market_prices.keys():
                # if market_name not in market_names:
                #     continue
                market_prices = dict()
                for date in fixing_dates:
                    market_prices[date] = all_market_prices[market_name][date]
                evaluation_kwds['all_market_prices'][market_name] = market_prices
        return evaluation_kwds


class SingleThreadedDependencyGraphRunner(DependencyGraphRunner):

    def run(self, **kwds):
        super(SingleThreadedDependencyGraphRunner, self).run(**kwds)
        self.call_queue = six.moves.queue.Queue()
        self.result_queue = six.moves.queue.Queue()
        self.result_ids = {}
        self.calls_dict = self.dependency_graph.call_requirements.copy()
        self.dependency_dict = self.dependency_graph.dependency_ids.copy()
        self.notify_dict = self.dependency_graph.notify_ids.copy()
        # Put the leaves on the execution queue.
        for call_requirement_id in self.dependency_graph.leaf_ids:
            self.call_queue.put(call_requirement_id)
        # Loop over the required call queue.
        from quantdsl.services import parse, get_market_names, get_fixing_dates
        while not self.call_queue.empty():
            call_requirement_id = self.call_queue.get()
            self.call_count += 1
            dependency_values = self.get_dependency_values(call_requirement_id)
            stubbed_expr_str, effective_present_time = self.calls_dict[call_requirement_id]
            stubbed_module = parse(stubbed_expr_str)

            assert isinstance(stubbed_module, Module)
            evaluation_kwds = self.get_evaluation_kwds(stubbed_expr_str, effective_present_time)
            handle_call_requirement(call_requirement_id, evaluation_kwds, dependency_values, self.result_queue,
                                    stubbed_expr_str, effective_present_time)
            while not self.result_queue.empty():
                call_requirement_id, result_value = self.result_queue.get()

                handle_result(call_requirement_id, result_value, self.results_dict, self.result_ids, self.notify_dict,
                              self.dependency_dict, self.call_queue)


class MultiProcessingDependencyGraphRunner(DependencyGraphRunner):

    def __init__(self, dependency_graph, pool_size=None):
        super(MultiProcessingDependencyGraphRunner, self).__init__(dependency_graph)
        self.pool_size = pool_size

    def run(self, **kwds):
        super(MultiProcessingDependencyGraphRunner, self).run(**kwds)
        import multiprocessing
        self.evaluation_pool = multiprocessing.Pool(processes=self.pool_size)
        self.manager = multiprocessing.Manager()
        self.execution_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.results_dict = self.manager.dict()
        self.result_ids = self.manager.dict()
        self.calls_dict = self.manager.dict()
        self.calls_dict.update(self.dependency_graph.call_requirements)
        self.dependency_dict = self.manager.dict()
        self.dependency_dict.update(self.dependency_graph.dependency_ids)
        self.notify_dict = self.manager.dict()
        self.notify_dict.update(self.dependency_graph.notify_ids)
        self.errors = []
        # if 'all_market_prices' in self.run_kwds:
        #     all_market_prices = self.run_kwds.pop('all_market_prices')
        #     all_market_prices_dict = self.manager.dict()
        #     for market_name, market_prices in all_market_prices.items():
        #         all_market_prices_dict[market_name] = market_prices.copy()
        #         # market_prices_dict = self.manager.dict()
        #         # all_market_prices_dict[market_name] = market_prices_dict
        #         # for fixing_date, market_price in market_prices.items():
        #         #     market_prices_dict[fixing_date] = market_price
        #     self.run_kwds['all_market_prices'] = all_market_prices_dict

        results_thread = Thread(target=self.process_results)
        results_thread.daemon = True
        results_thread.start()

        stub_evaluations_thread = Thread(target=self.make_results)
        stub_evaluations_thread.daemon = True
        stub_evaluations_thread.start()

        # Put the leaves on the execution queue.
        for callRequirementId in self.dependency_graph.leaf_ids:
            self.execution_queue.put(callRequirementId)

        while results_thread.isAlive() and stub_evaluations_thread.isAlive():
            sleep(1)
        if stub_evaluations_thread.isAlive():
            self.execution_queue.put(None)
            stub_evaluations_thread.join()
        if results_thread.isAlive():
            self.result_queue.put(None)
            results_thread.join()

        if self.errors:
            raise self.errors[0]

    def make_results(self):
        try:
            while True:
                call_requirement_id = self.execution_queue.get()
                if call_requirement_id is None:
                    break
                else:
                    dependency_values = self.get_dependency_values(call_requirement_id)
                    stubbed_expr_str, effective_present_time = self.calls_dict[call_requirement_id]

                    evaluation_kwds = self.get_evaluation_kwds(stubbed_expr_str, effective_present_time)

                    def target():
                        async_result = self.evaluation_pool.apply_async(handle_call_requirement, (
                            call_requirement_id,
                            evaluation_kwds,
                            dependency_values,
                            self.result_queue,
                            stubbed_expr_str,
                            effective_present_time,
                        ))
                        try:
                            async_result.get()
                        except Exception as error:
                            self.result_queue.put(None)
                            self.execution_queue.put(None)
                            self.errors.append(error)
                    thread = Thread(target=target)
                    thread.daemon = True
                    thread.start()

                    self.call_count += 1
        except:
            self.result_queue.put(None)
            raise

    def process_results(self):
        try:
            while True:
                result = self.result_queue.get()
                if result is None:
                    break
                else:
                    (call_requirement_id, result_value) = result
                    handle_result(call_requirement_id, result_value, self.results_dict, self.result_ids,
                                  self.notify_dict, self.dependency_dict, self.execution_queue)

                if call_requirement_id == self.dependency_graph.root_stub_id:
                    break
        except:
            self.execution_queue.put(None)
            raise


def handle_call_requirement(call_requirement_id, evaluation_kwds, dependency_values, result_queue, stubbed_expr_str, effective_present_time):
    """
    Evaluates the stubbed expr identified by 'call_requirement_id'.
    """

    # If necessary, overwrite the effective_present_time as the present_time in the evaluation_kwds.
    if effective_present_time:
        evaluation_kwds['present_time'] = effective_present_time

    # Evaluate the stubbed expr str.
    try:
        # Todo: Rework this dependency. Figure out how to use alternative set of DSL classes when multiprocessing.
        from quantdsl.services import parse
        stubbed_module = parse(stubbed_expr_str)
    except DslSyntaxError:
        raise

    assert isinstance(stubbed_module, Module), "Parsed stubbed expr string is not an module: %s" % stubbed_module

    dsl_namespace = DslNamespace()
    for stub_id, stub_result in dependency_values.items():
        dsl_namespace[stub_id] = stub_result

    simple_expr = stubbed_module.compile(dslLocals=dsl_namespace, dslGlobals={})
    assert isinstance(simple_expr, DslExpression), "Reduced parsed stubbed expr string is not an " \
                                                   "expression: %s" % type(simple_expr)
    result_value = simple_expr.evaluate(**evaluation_kwds)
    result_queue.put((call_requirement_id, result_value))


def handle_result(call_requirement_id, result_value, results_dict, result_ids, notify_dict, dependency_dict,
                  execution_queue):

    # Set the results.
    results_dict[call_requirement_id] = result_value
    result_ids[call_requirement_id] = None

    # Check if subscribers are ready to be executed.
    for dependent_id in notify_dict[call_requirement_id]:
        if dependent_id in results_dict:
            continue
        subscriber_required_ids = dependency_dict[dependent_id]
        # It's ready unless it requires a call that doesn't have a result yet.
        is_subscriber_ready = True
        for required_id in subscriber_required_ids:
            if required_id == call_requirement_id:
                continue  # We know we're done.
            if required_id not in results_dict:
                is_subscriber_ready = False
                break
        if is_subscriber_ready:
            execution_queue.put(dependent_id)

    # Check for results that should be deleted.
    # - dependency results should be deleted if there is a result for each dependent of the dependency
    for dependency_id in dependency_dict[call_requirement_id]:
        for dependent_id in notify_dict[dependency_id]:
            if dependent_id != call_requirement_id and dependent_id not in results_dict:
                # Need to keep it.
                break
        else:
            results_dict.pop(dependency_id)
