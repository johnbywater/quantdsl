from quantdsl.defaults import DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE, DEFAULT_PATH_COUNT, DEFAULT_PERTURBATION_FACTOR, \
    DEFAULT_PRICE_PROCESS_NAME, DEFAULT_INTEREST_RATE

__version__ = '1.4.0'


def calc(source_code, observation_date=None, interest_rate=DEFAULT_INTEREST_RATE, path_count=DEFAULT_PATH_COUNT,
         perturbation_factor=DEFAULT_PERTURBATION_FACTOR, price_process=None, periodisation=None,
         dsl_classes=None, max_dependency_graph_size=DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE,
         timeout=None, is_double_sided_deltas=True, verbose=False):
    import quantdsl.calculate

    return quantdsl.calculate.calc(source_code,
                                   observation_date=observation_date,
                                   interest_rate=interest_rate,
                                   path_count=path_count,
                                   perturbation_factor=perturbation_factor,
                                   price_process=price_process,
                                   periodisation=periodisation,
                                   dsl_classes=dsl_classes,
                                   max_dependency_graph_size=max_dependency_graph_size,
                                   timeout=timeout,
                                   is_double_sided_deltas=is_double_sided_deltas,
                                   verbose=verbose
                                   )






# Todo: Write test for import module that doesn't exist (currently fails with AttributeError cos spec is None).
# Todo: Support things like "third Wednesday of contract month" e.g. for settlement of futures.
# Todo: Multiprocessing of repeat simulations and valuations, with combination of results, to increase accuracy.
# Todo: Something to show estimated memory usage.
# Todo: Finish off support for yearly periodisation, and also hourly.
# Todo: Command line support, so QuantDSL can be written and executed without knowing Python.
# Todo: Price process as DSL (included in module directly or by import, module named as arg to calc()).
# Todo: Price process as DSL, either calibration params, or args for calibration with e.g. data from quandl).
# Todo: More price processes (jumps, heston).
# Todo: Better report object: separate out the delta hedging.
# Todo: Better interface objects: separate out the print() statements.
# Todo: Better deltas (dx sometimes uses average of prices in month, when not all prices may be used in expression,
# so identify which are involved and just use those).
# Todo: Tidy up how args are passed into evaluate(), it seems correct, but also a bit ad hoc.
# Todo: Support names in expressions being resolved by evaluation args (e.g. like 'observation_date' but more general).
# Todo: StockMarket element.
# Todo: Make price process create calibration params from market observations, as well as consume the calibration
# parameters.
# Todo: Change all names from lower camel case to underscore separated style.
# Todo: Develop multi-factor PriceProcess model (e.g. Schwartz-Smith)?
# Todo: Separate more clearly the syntax parsing (the Parser methods) from the semantic model the DSL objects.
# Todo: Separate more clearly a general function language implementation, which could be extended with any set of
# primitive elements.
# Todo: Use function arg annotation to declare types of DSL function args (will only work with Python 3).
# Todo: Develop closures, function defs within function defs may help to reduce call argument complexity.
# Todo: Think about other possibility of supporting another syntax? Perhaps there is a better syntax than the Python
#  based syntax?
# Todo: Develop natural language "skin" for Quant DSL expressions (something like how Gherkin syntax maps to
# functions?)?
# Todo: Support list comprehensions, for things like a strip of options?
# Todo: Develop a GUI that shows the graph being evaluated, allowing results to be examined, allows models to be
# developed. Look at the "language workbench" ideas from Martin Fowler (environment which shows example results,
# with editable code reachable from the results, and processing built-in)?
# Todo: Better stats available on number of call requirements, number of leaves in dependency graph, depth of graph?
# Todo: Prediction of cost of evaluating an expression, cost of network data requests, could calibrate by running
# sample stubbed expressions (perhaps complicated for LongstaffSchwartz cos the LeastSqaures routine is run
# different numbers of times).
# Todo: Support plotting.
# Todo: Clean up the str, repr, pprint stuff?
# Todo: Raise Quant DSL-specific type mismatch errors at run time (ie e.g. handle situation where datetime and
# string can't be added).
# Todo: Anyway, identify when type mismatches will occur - can't multiply a date by a number, can't add a date to a
# date or to a number, can't add a number to a timedelta. Etc?
# Todo: (Long one) Go through all ways of writing broken DSL source code, and make sure there are sensible errors.
# Todo: Figure out behaviour for observation_date > any fixing date, currently leads to a complex numbers (square
# root of negative time delta).
# Todo: Think/talk about regressing on correlated brownian motions, rather than uncorrelated ones - is there
# actually a difference? If no difference, there is no need to keep the uncorrelated Brownian motions.
# Todo: Review the test coverage of the code.
# Todo: Review the separation of concerns between the various test cases.
# Todo: Move these todos to an issue tracker.

# Note on how to install matplotlib in virtualenv:
# http://www.stevenmaude.co.uk/2013/09/installing-matplotlib-in-virtualenv.html
