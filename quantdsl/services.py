import datetime
import math
import sys
import threading
import time

from six import print_

from quantdsl.domain.model.dependency_graph import DependencyGraph
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.domain.services.price_processes import get_price_process
from quantdsl.semantics import DslNamespace, DslExpression, Market, DslError, StochasticObject, \
    compile_dsl_module, list_fixing_dates, find_delivery_points

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
    # assert isinstance(evaluation_kwds, dict)
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
    compile_start_time = datetime.datetime.now()

    # Compile the source into a primitive DSL expression, with optional dependency graph.
    dsl_expr = dsl_compile(dsl_source, filename=filename, is_parallel=is_parallel, dsl_classes=dsl_classes,
                           compile_kwds=compile_kwds)

    # Measure the compile_dsl_module time.
    compile_time_delta = datetime.datetime.now() - compile_start_time

    # Check the result of the compilation.
    # Todo: This feels unnecessary?
    if is_parallel:
        assert isinstance(dsl_expr, DependencyGraph), type(dsl_expr)
    else:
        assert isinstance(dsl_expr, DslExpression), type(dsl_expr)

    if is_verbose:
        if isinstance(dsl_expr, DependencyGraph):

            print_("Compiled DSL source into %d partial expressions (root ID: %s)." % (
                len(dsl_expr.stubbed_calls), dsl_expr.root_stub_id))
            print_()

        print_("Duration of compilation: %s" % compile_time_delta)
        print_()

        if isinstance(dsl_expr, DependencyGraph):
            if is_show_source:
                print_("Expression stack:")
                for stubbed_exprData in dsl_expr.stubbed_calls:
                    print_("  " + str(stubbed_exprData[0]) + ": " + str(stubbed_exprData[1]))
                print_()

    # If the expression has any stochastic elements, the evaluation kwds must have an 'observation_date' (datetime).
    if dsl_expr.has_instances(dsl_type=StochasticObject):
        observation_date = evaluation_kwds['observation_date']
        assert isinstance(observation_date, datetime.date)

        if is_verbose:
            print_("Observation time: %s" % observation_date)
            print_()

        # Avoid any confusion with the internal 'present_time' variable.
        if 'present_time' in evaluation_kwds:
            msg = ("Don't set present_time here, set observation_date instead. "
                   "Hint: Adjust effective present time with Fixing or Wait elements.")
            raise DslError(msg)

        # Initialise present_time as observation_date.
        evaluation_kwds['present_time'] = observation_date

        # If the expression has any Market elements, a market simulation is required
        if dsl_expr.has_instances(dsl_type=Market):

            # If a market simulation is required, evaluation kwds must have 'path_count' (integer).
            if 'path_count' not in evaluation_kwds:
                evaluation_kwds['path_count'] = DEFAULT_PATH_COUNT
            path_count = evaluation_kwds['path_count']
            assert isinstance(path_count, int)

            # If a market simulation is required, evaluation_kwds must have 'market_calibration' (integer).
            market_calibration = evaluation_kwds['market_calibration']
            assert isinstance(market_calibration, dict)

            # If a market simulation is required, generate the simulated prices using the price process.
            if not 'all_market_prices' in evaluation_kwds:

                if is_verbose:
                    print_("Price process: %s" % price_process_name)
                    print_()

                price_process = get_price_process(price_process_name)

                if is_verbose:
                    print_("Path count: %d" % path_count)
                    print_()

                if is_verbose:
                    print_("Finding all Market names and Fixing dates...")
                    print_()

                # Extract market names from the expression.
                # Todo: Avoid doing this on the dependency graph, when all the Market elements must be in the original.
                market_names = find_delivery_points(dsl_expr)

                # Extract fixing dates from the expression.
                # Todo: Perhaps collect the fixing dates?
                fixing_dates = list_fixing_dates(dsl_expr)

                if is_verbose:
                    print_("Simulating future prices for Market%s '%s' from observation time %s through fixing dates: %s." % (
                        '' if len(market_names) == 1 else 's',
                        ", ".join(market_names),
                        "'%04d-%02d-%02d'" % (observation_date.year, observation_date.month, observation_date.day),
                        # Todo: Only print first and last few, if there are loads.
                        ", ".join(["'%04d-%02d-%02d'" % (d.year, d.month, d.day) for d in fixing_dates[:8]]) + \
                        (", [...]" if len(fixing_dates) > 9 else '') + \
                        ((", '%04d-%02d-%02d'" % (fixing_dates[-1].year, fixing_dates[-1].month, fixing_dates[-1].day)) if len(fixing_dates) > 8 else '')
                    ))
                    print_()

                # Simulate the future prices.
                all_market_prices = price_process.simulate_future_prices(market_names, fixing_dates, observation_date, path_count, market_calibration)

                # Add future price simulation to evaluation_kwds.
                evaluation_kwds['all_market_prices'] = all_market_prices

    # Initialise the evaluation timer variable (needed by showProgress thread).
    evalStartTime = None

    if is_parallel:
        if is_verbose:

            len_stubbed_exprs = len(dsl_expr.stubbed_calls)
            lenLeafIds = len(dsl_expr.leaf_ids)

            msg = "Evaluating %d expressions (%d %s) with " % (len_stubbed_exprs, lenLeafIds, 'leaf' if lenLeafIds == 1 else 'leaves')
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
                    progress = 100.0 * lenResults / len_stubbed_exprs
                    sys.stdout.write("\rProgress: %01.2f%% (%s/%s) %s " % (progress, lenResults, len_stubbed_exprs, rateStr))
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
        if is_parallel:
            if is_multiprocessing:
                dependency_graph_runner_class = MultiProcessingDependencyGraphRunner
            else:
                dependency_graph_runner_class = SingleThreadedDependencyGraphRunner
            value = dsl_expr.evaluate(dependency_graph_runner_class=dependency_graph_runner_class, pool_size=pool_size, **evaluation_kwds)
        else:
            value = dsl_expr.evaluate(**evaluation_kwds)
    except:
        if is_parallel:
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
        if is_parallel:
            len_stubbed_exprs = len(dsl_expr.stubbed_calls)
            rateStr = "(%.2f expr/s)" % (len_stubbed_exprs / timeDeltaSeconds)
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
    # assert isinstance(compile_kwds, dict)
    compile_kwds.update(extraCompileKwds)

    # Parse the source into a DSL module object.
    dsl_module = dsl_parse(dsl_source, filename=filename, dsl_classes=dsl_classes)

    # assert isinstance(dsl_module, Module)

    # Compile the module into either a dependency graph
    # if 'is_parallel' is True, otherwise a single primitive expression.
    return compile_dsl_module(dsl_module, DslNamespace(), compile_kwds, is_dependency_graph=is_parallel)


