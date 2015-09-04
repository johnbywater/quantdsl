import datetime
import math
import sys
import threading
import time

from six import print_
from quantdsl.semantics import DslNamespace, DslExpression, Market, Fixing, DslError, Module, StochasticObject
from quantdsl.priceprocess.base import PriceProcess
from quantdsl.syntax import DslParser
from quantdsl.runtime import DependencyGraph, MultiProcessingDependencyGraphRunner, SingleThreadedDependencyGraphRunner

## Application services.

DEFAULT_PRICE_PROCESS_NAME = 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess'

DEFAULT_PATH_COUNT = 20000


def dsl_eval(dsl_source, filename='<unknown>', is_parallel=None, dsl_classes=None, compile_kwds=None,
             evaluation_kwds=None, price_process_name=None, is_multiprocessing=False, pool_size=0, is_verbose=False,
             is_show_source=False, **extra_evaluation_kwds):
    """
    Returns the result of evaluating a compiled module (an expression, or a user defined function).

    An expression (with optional function defs) will evaluate to a simple value.

    A function def will evaluate to a DSL expression, will may then be evaluated (more than one
    function def without an expression is an error).
    """
    if price_process_name is None:
        price_process_name = DEFAULT_PRICE_PROCESS_NAME

    if evaluation_kwds is None:
        evaluation_kwds = DslNamespace()
    assert isinstance(evaluation_kwds, dict)
    evaluation_kwds.update(extra_evaluation_kwds)

    if is_show_source:
        print_("Reading DSL source:")
        print_()
        print_('"""')
        print_(dsl_source.strip())
        print_('"""')
        print_()

    if is_verbose:
        print_("Compiling DSL source, please wait...")
        print_()
    compileStartTime = datetime.datetime.now()

    # Compile the source into a primitive DSL expression.
    dsl_expr = dsl_compile(dsl_source, filename=filename, is_parallel=is_parallel, dsl_classes=dsl_classes,
                           compile_kwds=compile_kwds)
    compileTimeDelta = datetime.datetime.now() - compileStartTime
    assert isinstance(dsl_expr, (DslExpression, DependencyGraph)), type(dsl_expr)

    if is_verbose:
        if isinstance(dsl_expr, DependencyGraph):
            lenStubbedExprs = len(dsl_expr.stubbed_exprs_data)

            print_("Compiled DSL source into %d partial expressions (root ID: %s)." % (
                lenStubbedExprs, dsl_expr.root_stub_id))
            print_()

        print_("Duration of compilation: %s" % compileTimeDelta)
        print_()

        if isinstance(dsl_expr, DependencyGraph):
            if is_show_source:
                print_("Expression stack:")
                for stubbedExprData in dsl_expr.stubbed_exprs_data:
                    print_("  " + str(stubbedExprData[0]) + ": " + str(stubbedExprData[1]))
                print_()

    if dsl_expr.has_instances(dslType=StochasticObject) or dsl_expr.has_instances(dslType=Market):
        # evaluation_kwds must have 'observation_time'
        observation_time = evaluation_kwds['observation_time']
        assert isinstance(observation_time, datetime.datetime)

        if is_verbose:
            print_("Observation time: %s" % observation_time)
            print_()

        if 'present_time' in evaluation_kwds:
            msg = ("Don't set present_time here, set observation_time instead. "
                   "Adjust present_time with a Fixing or a Wait.")
            raise DslError(msg)

        # Initialise present_time as observation_time.
        evaluation_kwds['present_time'] = observation_time

    if dsl_expr.has_instances(dslType=Market):
        # evaluation_kwds must have 'path_count'
        if 'path_count' not in evaluation_kwds:
            evaluation_kwds['path_count'] = DEFAULT_PATH_COUNT
        path_count = evaluation_kwds['path_count']
        assert isinstance(path_count, int)

        # Check calibration for market dynamics.
        market_calibration = evaluation_kwds['market_calibration']
        assert isinstance(market_calibration, dict)

        # Construct the price simulations.
        if not 'all_market_prices' in evaluation_kwds:

            # Load the price process object.
            price_process_module_name, price_process_class_name = price_process_name.rsplit('.', 1)
            try:
                price_process_module = __import__(price_process_module_name, '', '', '*')
            except Exception as e:
                raise DslError("Can't import price process module '%s': %s" % (price_process_module_name, e))
            try:
                price_process_class = getattr(price_process_module, price_process_class_name)
            except Exception as e:
                raise DslError("Can't find price process class '%s' in module '%s': %s" % (price_process_class_name, price_process_module_name, e))

            assert issubclass(price_process_class, PriceProcess)

            price_process = price_process_class()

            if is_verbose:
                print_("Price process class: %s" % str(price_process_class).split("'")[1])
                print_()

            if is_verbose:
                print_("Path count: %d" % path_count)
                print_()

            if is_verbose:
                print_("Finding all Market names and Fixing dates...")
                print_()

            market_names = get_market_names(dsl_expr)
            fixing_dates = get_fixing_dates(dsl_expr)

            if is_verbose:
                print_("Simulating future prices for Market%s '%s' from observation time %s through fixing dates: %s." % (
                    '' if len(market_names) == 1 else 's',
                    ", ".join(market_names),
                    "'%04d-%02d-%02d'" % (observation_time.year, observation_time.month, observation_time.day),
                    # Todo: Only print first and last few, if there are loads.
                    ", ".join(["'%04d-%02d-%02d'" % (d.year, d.month, d.day) for d in fixing_dates[:8]]) + \
                    (", [...]" if len(fixing_dates) > 9 else '') + \
                    ((", '%04d-%02d-%02d'" % (fixing_dates[-1].year, fixing_dates[-1].month, fixing_dates[-1].day)) if len(fixing_dates) > 8 else '')
                ))
                print_()

            # Simulate the future prices.
            all_market_prices = price_process.simulateFuturePrices(market_names, fixing_dates, observation_time, path_count, market_calibration)

            # Add future price simulation to evaluation_kwds.
            evaluation_kwds['all_market_prices'] = all_market_prices

    # Initialise the evaluation timer variable (needed by showProgress thread).
    evalStartTime = None

    if isinstance(dsl_expr, DependencyGraph):
        if is_verbose:

            lenStubbedExprs = len(dsl_expr.stubbed_exprs_data)
            lenLeafIds = len(dsl_expr.leaf_ids)

            msg = "Evaluating %d expressions (%d %s) with " % (lenStubbedExprs, lenLeafIds, 'leaf' if lenLeafIds == 1 else 'leaves')
            if is_multiprocessing and pool_size:
                msg += "a multiprocessing pool of %s workers" % pool_size
            else:
                msg += "a single thread"
            msg += ", please wait..."

            print_(msg)
            print_()

            # Define showProgress() thread.
            def showProgress(stop):
                progress = 0
                movingRates = []
                while progress < 100 and not stop.is_set():
                    time.sleep(0.3)
                    if evalStartTime is None:
                        continue
                    # Avoid race condition.
                    if not hasattr(dsl_expr, 'runner') or not hasattr(dsl_expr.runner, 'resultIds'):
                        continue
                    if stop.is_set():
                        break

                    try:
                        lenResults = len(dsl_expr.runner.resultIds)
                    except IOError:
                         break
                    resultsTime = datetime.datetime.now()
                    movingRates.append((lenResults, resultsTime))
                    if len(movingRates) >= 15:
                        movingRates.pop(0)
                    if len(movingRates) > 1:
                        firstLenResults, firstTimeResults = movingRates[0]
                        lastLenResults, lastTimeResults = movingRates[-1]
                        lenDelta = lastLenResults - firstLenResults
                        resultsTimeDelta = lastTimeResults - firstTimeResults
                        timeDeltaSeconds = resultsTimeDelta.seconds + resultsTimeDelta.microseconds * 0.000001
                        rateStr = "%.2f expr/s" % (lenDelta / timeDeltaSeconds)
                    else:
                        rateStr = ''
                    progress = 100.0 * lenResults / lenStubbedExprs
                    sys.stdout.write("\rProgress: %01.2f%% (%s/%s) %s " % (progress, lenResults, lenStubbedExprs, rateStr))
                    sys.stdout.flush()
                sys.stdout.write("\r")
                sys.stdout.flush()
            stop = threading.Event()
            thread = threading.Thread(target=showProgress, args=(stop,))

            # Start showProgress() thread.
            thread.start()

    # Start timing the evaluation.
    evalStartTime = datetime.datetime.now()
    try:
        # Evaluate the primitive DSL expression.
        if isinstance(dsl_expr, DependencyGraph):
            if is_multiprocessing:
                dependency_graph_runner_class = MultiProcessingDependencyGraphRunner
            else:
                dependency_graph_runner_class = SingleThreadedDependencyGraphRunner
            value = dsl_expr.evaluate(dependency_graph_runner_class=dependency_graph_runner_class, pool_size=pool_size, **evaluation_kwds)
        else:
            value = dsl_expr.evaluate(**evaluation_kwds)
    except:
        if isinstance(dsl_expr, DependencyGraph):
            if is_verbose:
                if thread.isAlive():
                    # print "Thread is alive..."
                    stop.set()
                    # print "Waiting to join with thread..."
                    thread.join(timeout=1)
                    # print "Joined with thread..."
        raise

    # Stop timing the evaluation.
    evalTimeDelta = datetime.datetime.now() - evalStartTime

    if isinstance(dsl_expr, DependencyGraph):
        if is_verbose:
            # Join with showProgress thread.
            thread.join(timeout=3)

    if is_verbose:
        timeDeltaSeconds = evalTimeDelta.seconds + evalTimeDelta.microseconds * 0.000001
        if isinstance(dsl_expr, DependencyGraph):
            rateStr = "(%.2f expr/s)" % (lenStubbedExprs / timeDeltaSeconds)
        else:
            rateStr = ''
        print_("Duration of evaluation: %s    %s" % (evalTimeDelta, rateStr))
        print_()

    # Prepare the result.
    import scipy
    if isinstance(value, scipy.ndarray):
        mean = value.mean()
        stderr = value.std() / math.sqrt(path_count)
        return {
            'mean': mean,
            'stderr': stderr
        }
    else:
        return value


def get_fixing_dates(dsl_expr):
    # Find all unique fixing dates.
    fixing_dates = set()
    for dslFixing in dsl_expr.find_instances(dslType=Fixing):
        assert isinstance(dslFixing, Fixing)
        if dslFixing.date is not None:
            fixing_dates.add(dslFixing.date)
        else:
            pass
    fixing_dates = sorted(list(fixing_dates))
    return fixing_dates


def get_market_names(dsl_expr):
    # Find all unique market names.
    market_names = set()
    for dsl_market in dsl_expr.find_instances(dslType=Market):
        assert isinstance(dsl_market, Market)
        market_names.add(dsl_market.name)

    return market_names


def dsl_compile(dsl_source, filename='<unknown>', is_parallel=None, dsl_classes=None, compile_kwds=None, **extraCompileKwds):
    """
    Returns a DSL expression, created according to the given DSL source module.

    That is, if the source module contains a function def and an expression which
    calls that function, then the expression's function call will be evaluated
    and the resulting DSL expression will be substituted for the function call
    in the module's expression, so that calls to user defined functions are eliminated
    and a single DSL expression is obtained.

    If the source module contains a function def, but no expression, the module is compiled
    into a function def object. Calling .apply() on a function def object will return a DSL
    expression object, which can be evaluated by calling its .evaluate() method.
    """
    if compile_kwds is None:
        compile_kwds = DslNamespace()
    assert isinstance(compile_kwds, dict)
    compile_kwds.update(extraCompileKwds)

    # Parse the source into a DSL module object.
    dslModule = parse(dsl_source, filename=filename, dsl_classes=dsl_classes)

    assert isinstance(dslModule, Module)

    # Compile the module into a single (primitive) expression.
    if is_parallel:
        dependencyGraphClass = DependencyGraph
    else:
        dependencyGraphClass = None
    return dslModule.compile(DslNamespace(), compile_kwds, dependencyGraphClass=dependencyGraphClass)


def parse(dsl_source, filename='<unknown>', dsl_classes=None):
    """
    Returns a DSL module, created according to the given DSL source module.
    """
    if dsl_classes is None:
        from quantdsl.semantics import defaultDslClasses
        dsl_classes = defaultDslClasses.copy()

    return DslParser().parse(dsl_source, filename=filename, dsl_classes=dsl_classes)
