import datetime
import math
import sys
import threading
import time
from quantdsl.semantics import DslNamespace, DslExpression, Market, Fixing, DslError, Module, StochasticObject
from quantdsl.priceprocess.base import PriceProcess
from quantdsl.syntax import DslParser
from quantdsl.runtime import DependencyGraph, MultiProcessingDependencyGraphRunner, SingleThreadedDependencyGraphRunner

## Application services.

DEFAULT_PRICE_PROCESS_NAME = 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess'

DEFAULT_PATH_COUNT = 20000

def eval(dslSource, filename='<unknown>', isParallel=None, dslClasses=None, compileKwds=None, evaluationKwds=None,
         priceProcessName=None, isMultiprocessing=False, poolSize=0, isVerbose=False, isShowSource=False, **extraEvaluationKwds):
    """
    Returns the result of evaluating a compiled module (an expression, or a user defined function).

    An expression (with optional function defs) will evaluate to a simple value.

    A function def will evaluate to a DSL expression, will may then be evaluated (more than one
    function def without an expression is an error).
    """
    if priceProcessName is None:
        priceProcessName = DEFAULT_PRICE_PROCESS_NAME

    if evaluationKwds is None:
        evaluationKwds = DslNamespace()
    assert isinstance(evaluationKwds, dict)
    evaluationKwds.update(extraEvaluationKwds)

    if isShowSource:
        print "Reading DSL source:"
        print
        print '"""'
        print dslSource.strip()
        print '"""'
        print

    if isVerbose:
        print "Compiling DSL source, please wait..."
        print
    compileStartTime = datetime.datetime.now()

    # Compile the source into a primitive DSL expression.
    dslExpr = compile(dslSource, filename=filename, isParallel=isParallel, dslClasses=dslClasses, compileKwds=compileKwds)
    compileTimeDelta = datetime.datetime.now() - compileStartTime
    assert isinstance(dslExpr, (DslExpression, DependencyGraph)), type(dslExpr)

    if isVerbose:
        if isinstance(dslExpr, DependencyGraph):
            lenStubbedExprs = len(dslExpr.stubbedExprsData)

            print "Compiled DSL source into %d partial expressions (root ID: %s)." % (lenStubbedExprs, dslExpr.rootStubId)
            print

        print "Duration of compilation: %s" % compileTimeDelta
        print

        if isinstance(dslExpr, DependencyGraph):
            if isShowSource:
                print "Expression stack:"
                for stubbedExprData in dslExpr.stubbedExprsData:
                    print "  " + str(stubbedExprData[0]) + ": " + str(stubbedExprData[1])
                print

    if dslExpr.hasInstances(dslType=StochasticObject):
        # evaluationKwds must have 'observationTime'
        observationTime = evaluationKwds['observationTime']
        assert isinstance(observationTime, datetime.datetime)

        if 'presentTime' in evaluationKwds:
            msg = "Don't set presentTime here, set observationTime instead. Adjust presentTime with a Fixing or a Wait."
            raise DslError(msg)

        # Initialise presentTime as observationTime.
        evaluationKwds['presentTime'] = observationTime

    if dslExpr.hasInstances(dslType=Market):
        # evaluationKwds must have 'pathCount'
        if 'pathCount' not in evaluationKwds:
            evaluationKwds['pathCount'] = DEFAULT_PATH_COUNT
        pathCount = evaluationKwds['pathCount']
        assert isinstance(pathCount, int)

        # Check calibration for market dynamics.
        marketCalibration = evaluationKwds['marketCalibration']
        assert isinstance(marketCalibration, dict)

        # Construct the price simulations.
        if not 'allMarketPrices' in evaluationKwds:

            # Load the price process object.
            priceProcessModuleName, priceProcessClassName = priceProcessName.rsplit('.', 1)
            try:
                priceProcessModule = __import__(priceProcessModuleName, '', '', '*')
            except Exception, e:
                raise DslError("Can't import price process module '%s': %s" % (priceProcessModuleName, e))
            try:
                priceProcessClass = getattr(priceProcessModule, priceProcessClassName)
            except Exception, e:
                raise DslError("Can't find price process class '%s' in module '%s': %s" % (priceProcessClassName, priceProcessModuleName, e))

            assert issubclass(priceProcessClass, PriceProcess)

            priceProcess = priceProcessClass()

            if isVerbose:
                print "Price process class: %s" % str(priceProcessClass).split("'")[1]
                print

            if isVerbose:
                print "Path count: %d" % pathCount
                print

            if isVerbose:
                print "Finding all Market names and Fixing dates..."
                print

            # Find all unique market names.
            marketNames = set()
            for dslMarket in dslExpr.findInstances(dslType=Market):
                assert isinstance(dslMarket, Market)
                marketNames.add(dslMarket.name)

            # Find all unique fixing dates.
            fixingDates = set()
            for dslFixing in dslExpr.findInstances(dslType=Fixing):
                assert isinstance(dslFixing, Fixing)
                fixingDates.add(dslFixing.date)
            fixingDates = sorted(list(fixingDates))

            if isVerbose:
                print "Simulating future prices for Market%s '%s' from observation time %s through fixing dates: %s." % (
                    '' if len(marketNames) == 1 else 's',
                    ", ".join(marketNames),
                    "'%04d-%02d-%02d'" % (observationTime.year, observationTime.month, observationTime.day),
                    # Todo: Only print first and last few, if there are loads.
                    ", ".join(["'%04d-%02d-%02d'" % (d.year, d.month, d.day) for d in fixingDates[:8]]) + \
                    (", [...]" if len(fixingDates) > 9 else '') + \
                    ((", '%04d-%02d-%02d'" % (fixingDates[-1].year, fixingDates[-1].month, fixingDates[-1].day)) if len(fixingDates) > 8 else '')
                )
                print

            # Simulate the future prices.
            allMarketPrices = priceProcess.simulateFuturePrices(marketNames, fixingDates, observationTime, pathCount, marketCalibration)

            # Add future price simulation to evaluationKwds.
            evaluationKwds['allMarketPrices'] = allMarketPrices

    # Initialise the evaluation timer variable (needed by showProgress thread).
    evalStartTime = None

    if isinstance(dslExpr, DependencyGraph):
        if isVerbose:

            lenStubbedExprs = len(dslExpr.stubbedExprsData)
            lenLeafIds = len(dslExpr.leafIds)

            msg = "Evaluating %d expressions (%d %s) with " % (lenStubbedExprs, lenLeafIds, 'leaf' if lenLeafIds == 1 else 'leaves')
            if isMultiprocessing and poolSize:
                msg += "a multiprocessing pool of %s workers" % poolSize
            else:
                msg += "a single thread"
            msg += ", please wait..."

            print msg
            print

            # Define showProgress() thread.
            def showProgress(stop):
                progress = 0
                movingRates = []
                while progress < 100 and not stop.is_set():
                    time.sleep(0.3)
                    if evalStartTime is None:
                        continue
                    # Avoid race condition.
                    if not hasattr(dslExpr, 'runner') or not hasattr(dslExpr.runner, 'resultIds'):
                        continue
                    if stop.is_set():
                        break

                    try:
                        lenResults = len(dslExpr.runner.resultIds)
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
        if isinstance(dslExpr, DependencyGraph):
            if isMultiprocessing:
                dependencyGraphRunnerClass = MultiProcessingDependencyGraphRunner
            else:
                dependencyGraphRunnerClass = SingleThreadedDependencyGraphRunner
            value = dslExpr.evaluate(dependencyGraphRunnerClass=dependencyGraphRunnerClass, poolSize=poolSize, **evaluationKwds)
        else:
            value = dslExpr.evaluate(**evaluationKwds)
    except:
        if isinstance(dslExpr, DependencyGraph):
            if isVerbose:
                if thread.isAlive():
                    # print "Thread is alive..."
                    stop.set()
                    # print "Waiting to join with thread..."
                    thread.join(timeout=1)
                    # print "Joined with thread..."
        raise

    # Stop timing the evaluation.
    evalTimeDelta = datetime.datetime.now() - evalStartTime

    if isinstance(dslExpr, DependencyGraph):
        if isVerbose:
            # Join with showProgress thread.
            thread.join(timeout=3)

    if isVerbose:
        timeDeltaSeconds = evalTimeDelta.seconds + evalTimeDelta.microseconds * 0.000001
        if isinstance(dslExpr, DependencyGraph):
            rateStr = "(%.2f expr/s)" % (lenStubbedExprs / timeDeltaSeconds)
        else:
            rateStr = ''
        print "Duration of evaluation: %s    %s" % (evalTimeDelta, rateStr)
        print

    # Prepare the result.
    import scipy
    if isinstance(value, scipy.ndarray):
        mean = value.mean()
        stderr = value.std() / math.sqrt(pathCount)
        return {
            'mean': mean,
            'stderr': stderr
        }
    else:
        return value


def compile(dslSource, filename='<unknown>', isParallel=None, dslClasses=None, compileKwds=None, **extraCompileKwds):
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
    if compileKwds is None:
        compileKwds = DslNamespace()
    assert isinstance(compileKwds, dict)
    compileKwds.update(extraCompileKwds)

    # Parse the source into a DSL module object.
    dslModule = parse(dslSource, filename=filename, dslClasses=dslClasses)

    assert isinstance(dslModule, Module)

    # Compile the module into a single (primitive) expression.
    if isParallel:
        dependencyGraphClass = DependencyGraph
    else:
        dependencyGraphClass = None
    return dslModule.compile(DslNamespace(), compileKwds, dependencyGraphClass=dependencyGraphClass)


def parse(dslSource, filename='<unknown>', dslClasses=None):
    """
    Returns a DSL module, created according to the given DSL source module.
    """
    if dslClasses is None:
        from quantdsl.semantics import defaultDslClasses
        dslClasses = defaultDslClasses.copy()

    return DslParser().parse(dslSource, filename=filename, dslClasses=dslClasses)
