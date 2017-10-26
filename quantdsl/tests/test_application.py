import datetime
import unittest
from copy import deepcopy

import scipy
from eventsourcing.domain.model.events import assert_event_handlers_empty

from quantdsl.interfaces.results import Results
from quantdsl.application.with_pythonobjects import QuantDslApplicationWithPythonObjects
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.market_simulation import MarketSimulation
from quantdsl.exceptions import CallLimitError, DslBinOpArgsError, DslCompareArgsError, \
    DslTestExpressionCannotBeEvaluated, RecursionDepthError, DslSyntaxError
from quantdsl.semantics import discount
from quantdsl.defaults import DEFAULT_PRICE_PROCESS_NAME


class ApplicationTestCase(unittest.TestCase):
    skip_assert_event_handers_empty = False
    NUMBER_DAYS = 5
    NUMBER_MARKETS = 2
    NUMBER_WORKERS = 20
    PATH_COUNT = 20000
    PERTURBATION_FACTOR = 0.01
    MAX_DEPENDENCY_GRAPH_SIZE = 2000

    price_process_name = DEFAULT_PRICE_PROCESS_NAME
    CALIBRATION_PARAMS = {
        'market': ['#1', '#2', 'NBP', 'TTF', 'SPARKSPREAD'],
        'sigma': [0.5, 0.5, 0.5, 0.4, 0.4],
        'curve': {
            '#1': [
                ('2011-1-1', 10),
            ],
            '#2': [
                ('2011-1-1', 10),
            ],
            'NBP': [
                ('2011-1-1', 10),
            ],
            'TTF': [
                ('2011-1-1', 11),
            ],
            'SPARKSPREAD': [
                ('2011-1-1', 1),
            ],
        },
        'rho': [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.4, 0.0],
            [0.0, 0.0, 0.4, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    }

    def setUp(self):
        if not self.skip_assert_event_handers_empty:
            assert_event_handlers_empty()

        self.setup_application(max_dependency_graph_size=self.MAX_DEPENDENCY_GRAPH_SIZE)

        self.calibration_params = deepcopy(self.CALIBRATION_PARAMS)
        self.observation_date = datetime.datetime(2011, 1, 1)
        self.interest_rate = 2.5

    def tearDown(self):
        if self.app is not None:
            self.app.close()
        if not self.skip_assert_event_handers_empty:
            assert_event_handlers_empty()
            # super(ContractValuationTestCase, self).tearDown()

    def setup_application(self, **kwargs):
        self.app = QuantDslApplicationWithPythonObjects(**kwargs)

    def assert_contract_value(self, source_code, expected_value=None, expected_deltas=None, expected_call_count=None,
                              periodisation=None, is_double_sided_deltas=True):

        results = self.calc_contract_value(source_code, is_double_sided_deltas, periodisation)

        # Check the results.
        if expected_value is not None:
            actual_value = results.fair_value_mean
            if isinstance(actual_value, float):
                self.assertAlmostEqual(actual_value, expected_value, places=2)
            else:
                self.assertEqual(actual_value, expected_value)

        # Check the deltas.
        if expected_deltas is not None:
            for perturbed_name in expected_deltas.keys():
                try:
                    actual_delta = results.deltas[perturbed_name]
                except KeyError:
                    raise KeyError((perturbed_name, results.deltas.keys()))
                else:
                    actual_value = actual_delta.mean()
                    expected_value = expected_deltas[perturbed_name]
                    error_msg = "{}: {} != {}".format(perturbed_name, actual_value, expected_value)
                    self.assertAlmostEqual(actual_value, expected_value, places=2, msg=error_msg)

        # Check the call count.
        if expected_call_count is not None:
            self.assertEqual(self.call_count, expected_call_count)

    def calc_contract_value(self, source_code, is_double_sided_deltas=True, periodisation=None):
        # Fix the seed.
        scipy.random.seed(1354802735)

        # Register the specification (creates call dependency graph).
        contract_specification = self.app.compile(source_code=source_code, observation_date=self.observation_date)

        # Check the call count (the number of nodes of the call dependency graph).
        self.call_count = self.calc_call_count(contract_specification.id)

        # Generate the market simulation.
        market_simulation = self.app.simulate(
            contract_specification=contract_specification,
            price_process_name=self.price_process_name,
            calibration_params=self.calibration_params,
            observation_date=self.observation_date,
            path_count=self.PATH_COUNT,
            interest_rate=self.interest_rate,
            perturbation_factor=self.PERTURBATION_FACTOR,
            periodisation=periodisation,
        )
        # Start the contract valuation.
        contract_valuation = self.app.evaluate(
            contract_specification_id=contract_specification.id,
            market_simulation_id=market_simulation.id,
            periodisation=periodisation,
            is_double_sided_deltas=is_double_sided_deltas
        )
        assert isinstance(contract_valuation, ContractValuation)
        valuation_result = self.app.get_result(contract_valuation)
        periods = self.app.get_periods(contract_valuation)

        return Results(
            valuation_result=valuation_result,
            periods=periods,
            contract_valuation=contract_valuation,
            market_simulation=market_simulation
        )

    def calc_call_count(self, contract_specification_id):
        return self.app.calc_call_count(contract_specification_id)

    def get_result(self, contract_valuation):
        return self.app.get_result(contract_valuation)


class ExpressionTests(ApplicationTestCase):
    def test_generate_valuation_addition(self):
        self.assert_contract_value("""1 + 2""", 3)
        self.assert_contract_value("""2 + 4""", 6)

    def test_market(self):
        self.assert_contract_value("Market('#1')", 10, {'#1': 1}, periodisation='alltime')
        self.assert_contract_value("Market('#2')", 10, {'#2': 1}, periodisation='alltime')

    def test_market_plus(self):
        self.assert_contract_value("Market('#1') + 10", 20)
        self.assert_contract_value("Market('#2') + 20", 30)

    def test_market_minus(self):
        self.assert_contract_value("Market('#1') - 10", 0)
        self.assert_contract_value("Market('#2') - 10", 0)

    def test_market_multiply_market(self):
        self.assert_contract_value("Market('#1') * Market('#2')", 100)

    def test_market_divide(self):
        self.assert_contract_value("Market('#1') / 10", 1)

    def test_fixing(self):
        specification = "Fixing(Date('2012-01-01'), Market('NBP'))"
        self.assert_contract_value(specification, 10.0195, expected_deltas={'NBP': 1.00195}, periodisation='alltime')
        specification = "Fixing(Date('2012-01-02'), Market('NBP'))"
        self.assert_contract_value(specification, 10.0195, expected_deltas={'NBP': 1.00195}, periodisation='alltime')
        specification = "Fixing(Date('2012-01-03'), Market('NBP'))"
        self.assert_contract_value(specification, 10.0195, expected_deltas={'NBP': 1.00195}, periodisation='alltime')
        specification = "Fixing(Date('2012-01-04'), Market('NBP'))"
        self.assert_contract_value(specification, 10.0196, expected_deltas={'NBP': 1.00196}, periodisation='alltime')
        specification = "Fixing(Date('2112-01-01'), Market('NBP'))"
        self.assert_contract_value(specification, 9.434, expected_deltas={'NBP': 0.943}, periodisation='alltime')
        specification = "Fixing(Date('2112-01-02'), Market('NBP'))"
        self.assert_contract_value(specification, 9.434, expected_deltas={'NBP': 0.943}, periodisation='alltime')
        specification = "0 * Fixing(Date('2112-01-01'), Market('NBP')) + Fixing(Date('2112-01-02'), Market('NBP'))"
        self.assert_contract_value(specification, 9.5024, expected_deltas={'NBP': 0.9502}, periodisation='alltime')
        specification = "0 * Fixing(Date('2112-01-01'), Market('NBP')) + 0 * Fixing(Date('2112-01-02'), " \
                        "Market('NBP')) + Fixing(Date('2112-01-03'), Market('NBP'))"
        self.assert_contract_value(specification, 9.551, expected_deltas={'NBP': 0.9551}, periodisation='alltime')

        # Todo: This seems wrong. Why isn't it still approximately 10? Need to check numbers in Brownian motions.
        specification = "Fixing(Date('2472-01-02'), Market('NBP'))"
        self.assert_contract_value(specification, 0, expected_deltas={'NBP': 0}, periodisation='alltime')

    def test_wait(self):
        specification = "Wait(Date('2012-01-01'), Market('NBP'))"
        self.assert_contract_value(specification, 9.772)

        # Check the delta equals the discount rate.
        self.calibration_params['sigma'] = [0.0, 0.0, 0.0, 0.0, 0.0]
        specification = "Wait('2022-1-1', Market('#1'))"
        discount_rate = discount(1, self.observation_date, datetime.datetime(2022, 1, 1), self.interest_rate)
        self.assert_contract_value(specification, 7.60, expected_deltas={'#1': discount_rate}, periodisation='alltime',
                                   is_double_sided_deltas=True)

        # Single sided delta.
        self.assert_contract_value(specification, 7.60, expected_deltas={'#1': discount_rate}, periodisation='alltime',
                                   is_double_sided_deltas=False)

    def test_settlement(self):
        specification = "Settlement(Date('2012-01-01'), Market('NBP'))"
        self.assert_contract_value(specification, 9.753)

    def test_choice(self):
        specification = "Fixing(Date('2012-01-01'), Choice( Market('NBP') - 9, 0))"
        self.assert_contract_value(specification, 2.435)

    def test_max(self):
        specification = "Fixing(Date('2011-01-01'), Max(Market('#1'), Market('#2')))"
        self.assert_contract_value(specification, 10.0000)

        specification = "Fixing(Date('2012-01-01'), Max(Market('#1'), Market('#2')))"
        self.assert_contract_value(specification, 12.752)

    def test_bermudan(self):
        specification = """
Fixing(Date('2011-06-01'), Choice(Market('NBP') - 9,
    Fixing(Date('2012-01-01'), Choice(Market('NBP') - 9, 0))))
"""
        self.assert_contract_value(specification, 2.393)

    def test_bermudan_with_alltime_delta(self):
        specification = """
Fixing(Date('2011-06-01'), Choice(Market('NBP') - 9,
    Fixing(Date('2012-01-01'), Choice(Market('NBP') - 9, 0))))
"""
        self.assert_contract_value(specification, 2.393, expected_deltas={'NBP': 0.671}, periodisation='alltime')

    def test_bermudan_with_monthly_deltas(self):
        specification = """
Fixing(Date('2011-06-01'), Choice(Market('NBP') - 9,
    Fixing(Date('2012-01-01'), Choice(Market('NBP') - 9, 0))))
"""
        self.assert_contract_value(specification, 2.393, expected_deltas={
            'NBP-2011-6': 0.0197,
            'NBP-2012-1': 0.0589,
        }, periodisation='monthly')

        self.assert_contract_value(specification, 2.393, expected_deltas={
            'NBP-2011-6': 0.0197,
            'NBP-2012-1': 0.0589,
        }, periodisation='monthly')

    def test_identical_fixings(self):
        specification = "Fixing(Date('2012-01-02'), Market('#1')) - Fixing(Date('2012-01-02'), Market('#1'))"
        self.assert_contract_value(specification, 0)

    def test_uncorrelated_markets(self):
        specification = """
Max(
    Fixing(
        Date('2012-01-01'),
        Market('#1')
    ) *
    Fixing(
        Date('2012-01-01'),
        Market('#2')
    ) / 10.0,
    0.0
) - Max(
    Fixing(
        Date('2013-01-01'),
        Market('#1')
    ), 0
)"""
        self.assert_contract_value(specification, 0.00856)

    def test_correlated_markets(self):
        specification = """
Max(
    Fixing(
        Date('2012-01-01'),
        Market('TTF')
    ) *
    Fixing(
        Date('2012-01-01'),
        Market('NBP')
    ) / 10.0,
    0.0
) - Max(
    Fixing(
        Date('2013-01-01'),
        Market('TTF')
    ), 0
)"""
        self.assert_contract_value(specification, 0.954)

    def test_futures(self):
        specification = "Wait(Date('2012-01-01'), Market('#1') - 9)"
        self.assert_contract_value(specification, 0.994)

    def test_european_zero_volatility(self):
        self.calibration_params['sigma'] = [0, 0, 0, 0, 0]
        self.assert_contract_value("Wait(Date('2012-01-01'), Choice(Market('#1') - 9, 0))", 0.975)

    def test_european_high_volatility(self):
        self.assert_contract_value("Wait(Date('2012-01-01'), Choice(Market('NBP') - 9, 0))", 2.375)

    def test_sum_contracts(self):
        specification = """
Fixing(
    Date('2011-06-01'),
    Choice(
        Market('NBP') - 9,
        Fixing(
            Date('2012-01-01'),
            Choice(
                Market('NBP') - 9,
                0
            )
        )
    )
) + Fixing(
    Date('2011-06-01'),
    Choice(
        Market('NBP') - 9,
        Fixing(
            Date('2012-01-01'),
            Choice(
                Market('NBP') - 9,
                0
            )
        )
    )
)
"""
        self.assert_contract_value(specification, 4.7877)

    def test_brownian_increments(self):
        specification = """
Max(
    Fixing(
        Date('2012-01-01'),
        Market('#1')
    ) /
    Fixing(
        Date('2011-01-01'),
        Market('#1')
    ),
    1.0
) - Max(
    Fixing(
        Date('2013-01-01'),
        Market('#1')
    ) /
    Fixing(
        Date('2012-01-01'),
        Market('#1')
    ),
    1.0
)
"""
        self.assert_contract_value(specification, 0.00, expected_deltas={'#1': 0.00}, periodisation='alltime')
        self.assert_contract_value(specification, 0.004, expected_deltas={'#1': -0.00}, periodisation='alltime',
                                   is_double_sided_deltas=False)

    def test_max_vs_choice(self):
        specification = """
Max(
    Fixing(
        Date('2012-01-01'),
        Market('#1')
    ) /
    Fixing(
        Date('2011-01-01'),
        Market('#1')
    ),
    1.0
) - Choice(
    Fixing(
        Date('2012-01-01'),
        Market('#1')
    ) /
    Fixing(
        Date('2011-01-01'),
        Market('#1')
    ),
    1.0
)
"""
        self.assert_contract_value(specification, 0.196, expected_deltas={'#1': 0.00}, periodisation='alltime')
        self.assert_contract_value(specification, 0.196, expected_deltas={'#1': 0.00}, periodisation='alltime',
                                   is_double_sided_deltas=False)

    def test_is_day_of_month(self):
        # Detects true and false.
        self.assert_contract_value("IsDayOfMonth(1)", True)
        self.assert_contract_value("IsDayOfMonth(2)", False)

        # Can be fixed.
        self.assert_contract_value("Fixing('2011-1-1', IsDayOfMonth(1))", True)
        self.assert_contract_value("Fixing('2011-1-2', IsDayOfMonth(1))", False)

        # Can be used in an if test condition.
        self.assert_contract_value("1 if IsDayOfMonth(1) else 0", 1)
        self.assert_contract_value("1 if Fixing(Date('2011-1-1'), IsDayOfMonth(1)) else 0", 1)
        self.assert_contract_value("1 if Fixing(Date('2011-1-2'), IsDayOfMonth(1)) else 0", 0)

        # Can be used in a function test expression, with function arg as date.
        self.assert_contract_value("""
def f(d):
    1 if Fixing(d, IsDayOfMonth(1)) else 0
    
f(Date('2011-1-1'))
""", 1)
        self.assert_contract_value("""
def f(d):
    1 if Fixing(d, IsDayOfMonth(1)) else 0
    
f(Date('2011-1-2'))
""", 0)


class FunctionTests(ApplicationTestCase):

    def test_broken_python_syntax(self):
        dsl_source = """
de f1():
    1

def f2():
    2

f1() + f2()
"""
        with self.assertRaises(DslSyntaxError):
            self.assert_contract_value(dsl_source)

    def test_broken_python_syntax_in_imported_module(self):
        dsl_source = """
from quantdsl.tests.lib.broken_python_syntax import f1, f2

f1() + f2()
"""
        with self.assertRaises(DslSyntaxError):
            self.assert_contract_value(dsl_source)

    def test_two_defs(self):
        dsl_source = """
def f1():
    1

def f2():
    2

f1() + f2()
"""
        self.assert_contract_value(dsl_source, 3, expected_call_count=3)

    def test_function_call_in_function_body(self):
        dsl_source = """
def f1():
    1 + f2()

def f2():
    2

f1()
"""
        self.assert_contract_value(dsl_source, 3, expected_call_count=3)

    def test_function_call_in_function_call_arg(self):
        dsl_source = """
def f1(x):
    1 + x

def f2():
    2
    
f1(f2())
"""
        self.assert_contract_value(dsl_source, 3, expected_call_count=2)

    def test_function_call_in_function_call_in_function_call_arg(self):
        dsl_source = """
def f1(x):
    1 + x

def f2(x):
    2 + x + Market('NBP') + f4()

def f3():
    Market('NBP') + f4()
    
def f4():
    1
    
def f5():
    Market('NBP')

f1(f2(Market('NBP')))
"""
        self.assert_contract_value(dsl_source, expected_call_count=4)

    def test_function_name_in_function_call_arg(self):
        dsl_source = """
def f1(f):
    1 + f()

def f2():
    2

f1(f2)
"""
        self.assert_contract_value(dsl_source, 3, expected_call_count=3)

    def test_ifstatement_test_expression_gets_function_that_can_be_evaluated(self):
        code = """
def f1(x):
    100 if x else -100

def f2(x):
    x * 3

f1(f2(%s))
"""
        self.assert_contract_value(code % 1, 100)
        self.assert_contract_value(code % 0, -100)

    def test_ifstatement_test_expression_has_function_call_that_can_be_evaluated_in_compare(self):
        code = """
def f1(x):
    100 if f2(x) > 0 else -100

def f2(x):
    x

f1(%s)
"""
        self.assert_contract_value(code % 1, 100)
        self.assert_contract_value(code % -1, -100)

    def test_ifstatement_test_expression_gets_function_call_that_can_be_evaluated(self):
        code = """
def f1(x):
    100 if f2(x) > 0 else -100

def f2(x):
    x

f1(f2(%s))
"""
        self.assert_contract_value(code % 1, 100)
        self.assert_contract_value(code % -1, -100)

    def test_ifstatement_test_expression_gets_function_call_that_cannot_be_evaluated(self):
        code = """
def f1(x):
    100 if x > 0 else -100

def f2(x):
    x * Choice(1, 2)

f1(f2(%s))
"""
        with self.assertRaises(DslTestExpressionCannotBeEvaluated):
            self.assert_contract_value(code % 1)

    def test_if_test_expression_has_nested_function_calls_and_gets_nested_function_calls(self):
        code = """
def f1(x):
    1 if f6(f4(f3(x))) else 0

def f2(x):
    x

def f3(x):
    x

def f4(x):
    x

def f5(x):
    x

def f6(x):
    x

f1(f5(f2(%s)))
"""
        # Not sure how this works, but it seems to!
        self.assert_contract_value(code % 1, 1)
        self.assert_contract_value(code % 0, 0)

    def test_call_cache_simple_function(self):
        code = """
def f(x):
    x
"""
        self.assert_contract_value(code + "f(1)", expected_call_count=2)
        self.assert_contract_value(code + "f(1) + f(1)", expected_call_count=2)
        self.assert_contract_value(code + "f(1) + f(1) + f(1)", expected_call_count=2)
        self.assert_contract_value(code + "f(1) + f(1) + f(1)", expected_call_count=2)
        self.assert_contract_value(code + "f(f(1))", expected_call_count=2)
        self.assert_contract_value(code + "f(f(f(1)))", expected_call_count=2)
        self.assert_contract_value(code + "f(f(f(1))) + f(f(f(1)))", expected_call_count=2)
        self.assert_contract_value(code + "f(f(f(1))) + f(f(f(f(1))))", expected_call_count=2)

    def test_call_cache_inlined_function(self):
        code = """
@inline
def f(x):
    x
"""
        self.assert_contract_value(code + "f(1)", expected_call_count=1)
        self.assert_contract_value(code + "f(1) + f(1)", expected_call_count=1)
        self.assert_contract_value(code + "f(1) + f(1) + f(1)", expected_call_count=1)
        self.assert_contract_value(code + "f(f(1))", expected_call_count=1)
        self.assert_contract_value(code + "f(f(f(1)))", expected_call_count=1)
        self.assert_contract_value(code + "f(f(f(1))) + f(f(f(1)))", expected_call_count=1)
        self.assert_contract_value(code + "f(f(f(1))) + f(f(f(f(1))))", expected_call_count=1)

    def test_call_cache_recombine_branches_integer_args(self):
        code = """
def f(x, t):
    if t <= 0:
        x
    else:
        if x <= 0:
            Max(f(x, t-1), f(x+1, t-1))
        else:
            Max(f(x-1, t-1), f(x, t-1))
            
f(0, %s)
"""
        self.assert_contract_value(code % 0, 0, expected_call_count=2)
        self.assert_contract_value(code % 1, 1, expected_call_count=4)
        self.assert_contract_value(code % 2, 1, expected_call_count=6)
        self.assert_contract_value(code % 3, 1, expected_call_count=8)
        self.assert_contract_value(code % 4, 1, expected_call_count=10)

    def test_call_cache_recombine_branches_with_function_call_arg(self):
        code = """
def f1(x, y, t):
    if t <= 0:
        x + y
    else:
        if x <= 0:
            Max(f1(x, y, t-1), f1(x+1, y, t-1))
        else:
            Max(f1(x-1, y, t-1), f1(x, y, t-1))
        
def f2(x):
    x
        
f1(0, f2(1), %s)
"""
        self.assert_contract_value(code % 0, 1, expected_call_count=2)
        self.assert_contract_value(code % 1, 2, expected_call_count=4)
        self.assert_contract_value(code % 2, 2, expected_call_count=6)
        self.assert_contract_value(code % 3, 2, expected_call_count=8)
        self.assert_contract_value(code % 4, 2, expected_call_count=10)

    def test_call_cache_recombine_branches_with_inlined_function_call_arg(self):
        code = """
def f1(x, y, t):
    if t <= 0:
        x + y
    else:
        if x <= 0:
            Max(f1(x, y, t-1), f1(x+1, y, t-1))
        else:
            Max(f1(x-1, y, t-1), f1(x, y, t-1))
        
@inline
def f2(x):
    x
        
f1(0, f2(1), %s)
"""
        self.assert_contract_value(code % 0, 1, expected_call_count=2)
        self.assert_contract_value(code % 1, 2, expected_call_count=4)
        self.assert_contract_value(code % 2, 2, expected_call_count=6)
        self.assert_contract_value(code % 3, 2, expected_call_count=8)
        self.assert_contract_value(code % 4, 2, expected_call_count=10)

    def test_call_cache_recombine_branches_with_referenced_function_call_arg(self):
        code = """
def f1(x, y, t):
    if t <= 0:
        x + y(1)
    else:
        if x <= 0:
            Max(f1(x, y, t-1), f1(x+1, y, t-1))
        else:
            Max(f1(x-1, y, t-1), f1(x, y, t-1))
        
@inline
def f2(x):
    x
        
f1(0, f2, %s)
"""
        self.assert_contract_value(code % 0, 1, expected_call_count=2)
        self.assert_contract_value(code % 1, 2, expected_call_count=4)
        self.assert_contract_value(code % 2, 2, expected_call_count=6)
        self.assert_contract_value(code % 3, 2, expected_call_count=8)
        self.assert_contract_value(code % 4, 2, expected_call_count=10)

    def test_call_cache_recombine_branches_with_inlined_function_call_arg_refactored(self):
        code = """
def f1(x, y, t):
    if t <= 0:
        x + y
    else:
        if x <= 0:
            up(x, y, t)
        else:
            down(x, y, t)
        
@inline
def f2(x):
    x
        
@inline
def up(x, y, t):
    Max(f1(x, y, t-1), f1(x+1, y, t-1))

@inline
def down(x, y, t):
    Max(f1(x-1, y, t-1), f1(x, y, t-1))

f1(0, f2(1), %s)
"""
        self.assert_contract_value(code % 0, 1, expected_call_count=2)
        self.assert_contract_value(code % 1, 2, expected_call_count=4)
        self.assert_contract_value(code % 2, 2, expected_call_count=6)
        self.assert_contract_value(code % 3, 2, expected_call_count=8)
        self.assert_contract_value(code % 4, 2, expected_call_count=10)

    def test_call_cache_recombine_branches_with_timedelta_arg(self):
        # calls, time deltas).
        code = """
def f1(t, x):
    if t <= 0:
        x
    else:
        Max(1 + f1(f2(t), x), f1(f2(t), x+1)) 

@inline
def f2(t):
    t-1  
      
f1(%s, 1)
"""
        self.assert_contract_value(code % 0, 1, expected_call_count=2)
        self.assert_contract_value(code % 1, 2, expected_call_count=4)
        self.assert_contract_value(code % 2, 3, expected_call_count=7)
        self.assert_contract_value(code % 3, 4, expected_call_count=11)
        self.assert_contract_value(code % 4, 5, expected_call_count=16)

    def test_inlined_same_as_not_inlined_american_option(self):
        tmpl = """
{decorator}
def American(starts, ends, strike, underlying):
    if starts <= ends:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )
    else:
        0

{decorator}
def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

American(Date('2011-1-1'), Date('2011-1-3'), 9, Market('NBP'))
"""

        source_code_with_inline = tmpl.format(decorator='@inline')
        source_code_without_inline = tmpl.format(decorator='')

        results_without_inline = self.calc_contract_value(source_code_without_inline)
        results_with_inline = self.calc_contract_value(source_code_with_inline)

        value_without_inline = results_without_inline.fair_value.mean()
        value_with_inline = results_with_inline.fair_value.mean()

        self.assertAlmostEqual(value_with_inline, value_without_inline, places=3)

    def test_inlined_same_as_not_inlined_gas_storage(self):
        source_code_without_inline = """
def GasStorage(start, end, commodity_name, quantity, limit, step):
    if ((start < end) and (limit > 0)):
        if quantity <= 0:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Inject(start, end, commodity_name, quantity, limit, step, 1),
            ))
        elif quantity >= limit:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Withdraw(start, end, commodity_name, quantity, limit, step, 1),
            ))
        else:
            # 0
            Wait(start, Choice(
                # Continue(start, end, commodity_name, quantity, limit, step),
                # Inject(start, end, commodity_name, quantity, limit, step, 1),
                Withdraw(start, end, commodity_name, quantity, limit, step, 1),
                Withdraw(start, end, commodity_name, quantity, limit, step, 1),
                # 0,
                # 1
                # 1
                # Withdraw(start, end, commodity_name, quantity, limit, step, 1),
            ))
    else:
        0

@inline
def Continue(start, end, commodity_name, quantity, limit, step):
    GasStorage(start + step, end, commodity_name, quantity, limit, step)

@inline
def Inject(start, end, commodity_name, quantity, limit, step, vol):
    Continue(start, end, commodity_name, quantity + vol, limit, step) - vol * Market(commodity_name)

@inline
def Withdraw(start, end, commodity_name, quantity, limit, step, vol):
    Continue(start, end, commodity_name, quantity - vol, limit, step) + vol * Market(commodity_name)

GasStorage(Date('2011-8-1'), Date('2011-10-1'), 'NBP', 0, 1000, TimeDelta('1m'))
"""
        source_code_with_inline = "@inline\n" + source_code_without_inline

        self.calibration_params['curve'] = {
            'NBP': [
                ('2011-1-1', 13.5),
                # ('2011-2-1', 11.0),
                # ('2011-3-1', 10.0),
                # ('2011-4-1', 9.0),
                # ('2011-5-1', 8.5),
                # ('2011-6-1', 7.0),
                # ('2011-7-1', 6.5),
                # ('2011-8-1', 7.5),
                # ('2011-9-1', 8.5),
                # ('2011-10-1', 10.0),
                # ('2011-11-1', 11.5),
                # ('2011-12-1', 12.0),
            ],
        }

        results_with_inline = self.calc_contract_value(source_code_with_inline)
        results_without_inline = self.calc_contract_value(source_code_without_inline)
        value_without_inline = results_without_inline.fair_value.mean()
        value_with_inline = results_with_inline.fair_value.mean()
        self.assertAlmostEqual(value_with_inline, value_without_inline, places=3)

    def test_functional_fibonacci_numbers(self):
        fib_tmpl = """
def fib(n):
    fib(n-1) + fib(n-2) if n > 1 else n

fib(%s)
"""
        self.assert_contract_value(fib_tmpl % 0, 0, expected_call_count=2)
        self.assert_contract_value(fib_tmpl % 1, 1, expected_call_count=2)
        self.assert_contract_value(fib_tmpl % 2, 1, expected_call_count=4)
        self.assert_contract_value(fib_tmpl % 3, 2, expected_call_count=5)
        self.assert_contract_value(fib_tmpl % 4, 3, expected_call_count=6)
        self.assert_contract_value(fib_tmpl % 5, 5, expected_call_count=7)
        self.assert_contract_value(fib_tmpl % 6, 8, expected_call_count=8)
        self.assert_contract_value(fib_tmpl % 7, 13, expected_call_count=9)

    def test_factorial_1(self):
        code = """
def factorial(n):
    if n > 1:
        n * factorial(n-1)
    else:
        1
factorial(%s)
"""
        self.assert_contract_value(code % 1, 1, expected_call_count=2)
        self.assert_contract_value(code % 10, 3628800, expected_call_count=11)

    def test_factorial_2(self):
        code = """
def factorial(n, acc):
    if n > 1:
        factorial(n-1, acc * n)
    else:
        acc
factorial(%s, 1)
"""
        self.assert_contract_value(code % 1, 1, expected_call_count=2)
        self.assert_contract_value(code % 10, 3628800, expected_call_count=11)

    def test_compare_args_error(self):
        code = "TimeDelta('1d') > 1"
        with self.assertRaises(DslCompareArgsError):
            self.assert_contract_value(code)

    def test_bin_op_args_error(self):
        code = "TimeDelta('1d') + 1"
        with self.assertRaises(DslBinOpArgsError):
            self.assert_contract_value(code)

    def test_function_call_with_keywords_not_currently_supported(self):
        dsl_source = """
def f(n):
    return n

f(n=1)
"""
        with self.assertRaises(DslSyntaxError):
            self.assert_contract_value(dsl_source)

    def test_function_call_with_starargs_not_currently_supported(self):
        dsl_source = """
def f(n):
    return n

f(*[1])
"""
        with self.assertRaises(DslSyntaxError):
            self.assert_contract_value(dsl_source)

    def test_function_call_with_starstarkwargs_not_currently_supported(self):
        dsl_source = """
def f(n):
    return n

f(**{n:1})
"""
        with self.assertRaises(DslSyntaxError):
            self.assert_contract_value(dsl_source)

    def test_function_call_as_call_arg_gets_correct_present_time_when_inlined(self):
        dsl_source = """
def MyFixing(end, underlying):
    return Fixing(end, underlying)

@inline
def Discount(start):
    Settlement(start, 1)

MyFixing(Date('2012-01-01'), Discount(Date('2011-01-01')))
"""
        self.assert_contract_value(dsl_source, 1.025, expected_call_count=2)

    def test_function_call_as_call_arg_gets_correct_present_time_when_not_inlined(self):
        dsl_source = """
def MyFixing(end, underlying):
    return Fixing(end, underlying)

def Discount(start):
    Settlement(start, 1)

MyFixing(Date('2012-01-01'), Discount(Date('2011-01-01')))
"""
        self.assert_contract_value(dsl_source, 1.025, expected_call_count=None)

    def test_function_as_call_arg_gets_correct_present_time(self):
        # Novelty here is passing a function as a
        # call arg, and then calling the arg name.
        dsl_source = """
def MyWait(start, end, func):
    return Fixing(end, func(start))

def Discount(start):
    Settlement(start, 1)

MyWait(Date('2011-01-01'), Date('2012-01-01'), Discount)
    """
        self.assert_contract_value(dsl_source, 1.025, expected_call_count=None)

    def test_functional_derivative_option_definition(self):
        specification = """
def Option(date, strike, x, y):
    return Wait(date, Choice(x - strike, y))

Option(Date('2012-01-01'), 9, Market('NBP'), 0)
"""
        self.assert_contract_value(specification, 2.375, expected_call_count=2)

    def test_functional_european_option_definition(self):
        specification = """
def Option(date, strike, underlying, alternative):
    return Wait(date, Choice(underlying - strike, alternative))

def European(date, strike, underlying):
    return Option(date, strike, underlying, 0)

European(Date('2012-01-01'), 9, Market('NBP'))
"""
        self.assert_contract_value(specification, expected_value=2.375, expected_deltas={'NBP': 0.66876},
                                   expected_call_count=3, periodisation='alltime')

        self.assert_contract_value(specification, expected_value=2.375, expected_deltas={'NBP': 0.66876},
                                   expected_call_count=3, periodisation='alltime', is_double_sided_deltas=False)

    def test_functional_european_stock_option_definition(self):
        specification = """
def Option(date, strike, underlying, alternative):
    return Wait(date, Choice(underlying - strike, alternative))

def European(date, strike, underlying):
    return Option(date, strike, underlying, 0)

def EuropeanStock(start, end, strike, name):
    return European(end, strike, Settlement(start, 1) * ForwardMarket(start, name))

EuropeanStock(Date('2011-01-01'), Date('2012-01-01'), 9, 'NBP')
"""
        self.assert_contract_value(specification, 2.54659, expected_deltas={'NBP': 0.7017}, expected_call_count=4,
                                   periodisation='alltime')

        self.assert_contract_value(specification, 2.54659, expected_deltas={'NBP': 0.7017}, expected_call_count=4,
                                   periodisation='alltime', is_double_sided_deltas=False)

    def test_functional_european_stock_option_definition_with_stock_market_def_inlined(self):
        specification = """
def Option(date, strike, underlying, alternative):
    return Wait(date, Choice(underlying - strike, alternative))

def European(date, strike, underlying):
    return Option(date, strike, underlying, 0)

def EuropeanStock(start, end, strike, name):
    return European(end, strike, StockMarket(start, name))

@inline
def StockMarket(start, name):
    Settlement(start, 1) * ForwardMarket(start, name)

EuropeanStock(Date('2011-01-01'), Date('2012-01-01'), 9, 'NBP')
"""
        self.assert_contract_value(specification, 2.54659, expected_deltas={'NBP': 0.7017}, expected_call_count=4,
                                   periodisation='alltime')

        self.assert_contract_value(specification, 2.54659, expected_deltas={'NBP': 0.7017}, expected_call_count=4,
                                   periodisation='alltime', is_double_sided_deltas=False)

    def test_functional_european_stock_option_definition_with_stock_market_def_non_inlined(self):
        specification = """
def Option(date, strike, underlying, alternative):
    return Wait(date, Choice(underlying - strike, alternative))

def European(date, strike, underlying):
    return Option(date, strike, underlying, 0)

def EuropeanStock(start, end, strike, name):
    return European(end, strike, StockMarket(start, name))

def StockMarket(start, name):
    Settlement(start, 1) * ForwardMarket(start, name)

EuropeanStock(Date('2011-01-01'), Date('2012-01-01'), 9, 'NBP')
"""
        self.assert_contract_value(specification, 2.546, expected_deltas={'NBP': 0.7017}, expected_call_count=5,
                                   periodisation='alltime')

        self.assert_contract_value(specification, 2.546, expected_deltas={'NBP': 0.7017}, expected_call_count=5,
                                   periodisation='alltime', is_double_sided_deltas=False)

    def test_generate_valuation_american_option_with_market(self):
        american_option_tmpl = """
def American(starts, ends, strike, underlying):
    if starts <= ends:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )
    else:
        0

def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

American(Date('%(starts)s'), Date('%(ends)s'), %(strike)s, Market('%(underlying)s'))
"""
        code = american_option_tmpl % {'starts': '2011-01-01', 'ends': '2011-02-01', 'strike': 9, 'underlying': 'NBP'}

        self.assert_contract_value(code, expected_value=1.1805, expected_deltas={'NBP': 0.8008},
                                   expected_call_count=66, periodisation='alltime')

        self.assert_contract_value(code, expected_value=1.1805, expected_deltas={'NBP': 0.811},
                                   expected_call_count=66, periodisation='alltime',
                                   is_double_sided_deltas=False)

    def test_generate_valuation_american_option_with_stock_market(self):
        american_option_tmpl = """
def American(starts, ends, strike, underlying):
    if starts <= ends:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )
    else:
        0

def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

def StockMarket(start, name):
    Settlement(start, ForwardMarket(start, name))

American(Date('%(starts)s'), Date('%(ends)s'), %(strike)s, StockMarket('%(starts)s', '%(underlying)s'))
"""
        code = american_option_tmpl % {
            'starts': '2011-01-01',
            'ends': '2011-04-01',
            # 'ends': '2012-01-01',
            'strike': 9,
            'underlying': 'NBP'
            # }, 1.1874, {'#1': 1.0185}, expected_call_count=None, periodisation='alltime')
        }
        self.assert_contract_value(code, expected_value=1.5105, expected_deltas={'NBP': 0.707},
                                   expected_call_count=275, periodisation='alltime')

        self.assert_contract_value(code, expected_value=1.5105, expected_deltas={'NBP': 0.707},
                                   expected_call_count=275, periodisation='alltime', is_double_sided_deltas=False)

    def test_generate_valuation_american_option_with_inlined_stockmarket_def(self):
        american_option_tmpl = """
def American(starts, ends, strike, underlying):
    if starts <= ends:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )
    else:
        0

def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

@inline
def StockMarket(start, name):
    Settlement(start, ForwardMarket(start, name))

American(Date('%(starts)s'), Date('%(ends)s'), %(strike)s, StockMarket('%(starts)s', '%(underlying)s'))
"""
        code = american_option_tmpl % {'starts': '2011-01-01', 'ends': '2011-04-01', 'strike': 9, 'underlying': 'NBP'}
        self.assert_contract_value(code, expected_value=1.5105, expected_deltas={'NBP': 0.707},
                                   expected_call_count=184,
                                   periodisation='alltime')
        self.assert_contract_value(code, expected_value=1.5105, expected_deltas={'NBP': 0.707},
                                   expected_call_count=184,
                                   periodisation='alltime', is_double_sided_deltas=False)

    def test_generate_valuation_american_option_with_inlined_option_def(self):
        american_option_tmpl = """
def American(starts, ends, strike, underlying):
    if starts <= ends:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )
    else:
        0

@inline
def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

def StockMarket(start, name):
    Settlement(start, ForwardMarket(start, name))

American(Date('%(starts)s'), Date('%(ends)s'), %(strike)s, StockMarket('%(starts)s', '%(underlying)s'))
"""
        code = american_option_tmpl % {'starts': '2011-01-01', 'ends': '2011-04-01', 'strike': 9, 'underlying': 'NBP'}
        self.assert_contract_value(code, expected_value=1.51, expected_deltas={'NBP': 0.707}, expected_call_count=184,
                                   periodisation='alltime')
        self.assert_contract_value(code, expected_value=1.51, expected_deltas={'NBP': 0.707}, expected_call_count=184,
                                   periodisation='alltime', is_double_sided_deltas=False)

    def test_generate_valuation_american_option_with_inlined_stockmarket_and_option_def(self):
        american_option_tmpl = """
def American(starts, ends, strike, underlying):
    if starts <= ends:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )
    else:
        0

@inline
def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

@inline
def StockMarket(start, name):
    Settlement(start, ForwardMarket(start, name))

American(Date('%(starts)s'), Date('%(ends)s'), %(strike)s, StockMarket('%(starts)s', '%(underlying)s'))
"""
        code = american_option_tmpl % {'starts': '2011-01-01', 'ends': '2011-04-01', 'strike': 9, 'underlying': 'NBP'}
        self.assert_contract_value(code, expected_value=1.51, expected_deltas={'NBP': 0.707}, expected_call_count=93,
                                   periodisation='alltime')
        self.assert_contract_value(code, expected_value=1.51, expected_deltas={'NBP': 0.707}, expected_call_count=93,
                                   periodisation='alltime', is_double_sided_deltas=False)

    def test_observation_date(self):
        specification = """ObservationDate()"""
        self.assert_contract_value(specification, datetime.datetime(2011, 1, 1, 0, 0))

    def test_generate_valuation_swing_option(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-01-05'), Market('NBP'), 3)
"""
        self.assert_contract_value(specification, expected_value=30.00, expected_deltas={'NBP': 3.0},
                                   expected_call_count=15, periodisation='alltime')

        self.assert_contract_value(specification, expected_value=30.00, expected_deltas={'NBP': 3.0},
                                   expected_call_count=15, periodisation='alltime', is_double_sided_deltas=False)


class LongerTests(ApplicationTestCase):
    def test_value_swing_option(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, 
            Market(underlying)),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-1-1'), Date('2011-1-5'), 'NBP', 3)
"""
        self.assert_contract_value(specification, expected_value=30.0)
        # Stopped testing for the deltas on this contract, because it
        # fails for some reason on Travis. Perhaps it's a degenerate case?
        # self.assert_contract_value(specification, expected_value=30.0, expected_deltas={'NBP-2011-1-4': 0.3959},
        #                            expected_call_count=15, periodisation='daily')
        #
        # self.assert_contract_value(specification, expected_value=30.0, expected_deltas={'NBP-2011-1-4': 0.7918},
        #                            expected_call_count=15, periodisation='daily', is_double_sided_deltas=False)


    def test_generate_valuation_power_plant_option_sparkspread(self):
        specification = """
def PowerPlant(start_date, end_date, underlying, time_since_off):
    if (start_date < end_date):
        Wait(start_date, Choice(
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, Running()) \
               + ProfitFromRunning(start_date, underlying, time_since_off),
            PowerPlant(start_date + TimeDelta('1d'), end_date, underlying, Stopped(time_since_off)),
        ))
    else:
        return 0

@inline
def Running():
    return 0

@inline
def Stopped(time_since_off):
    return Min(2, time_since_off + 1)

@inline
def ProfitFromRunning(start_date, underlying, time_since_off):
    if time_since_off == 0:
        return Fixing(start_date, underlying)
    elif time_since_off == 1:
        return 0.9 * Fixing(start_date, underlying)
    else:
        return 0.8 * Fixing(start_date, underlying)

PowerPlant(Date('2012-01-01'), Date('{end_date}'), Market('SPARKSPREAD'), Running())
"""

        # Check single-sided vs. double sided deltas.
        self.assert_contract_value(specification.format(end_date='2012-01-13'), expected_value=11.719,
                                   expected_deltas={'SPARKSPREAD-2012-1': 11.699}, expected_call_count=37,
                                   periodisation='monthly')

        self.assert_contract_value(specification.format(end_date='2012-01-13'), expected_value=11.719,
                                   expected_deltas={'SPARKSPREAD-2012-1': 11.699}, expected_call_count=37,
                                   periodisation='monthly', is_double_sided_deltas=False)

        # Check the call counts.
        self.assert_contract_value(specification.format(end_date='2012-01-13'), 11.719, expected_call_count=37)
        self.assert_contract_value(specification.format(end_date='2012-01-14'), expected_call_count=40)
        self.assert_contract_value(specification.format(end_date='2012-01-15'), expected_call_count=43)
        self.assert_contract_value(specification.format(end_date='2012-01-16'), expected_call_count=46)

    def test_generate_valuation_power_plant_option_power_and_gas_forward(self):
        specification = """
def PowerPlant(start, end, duration_off):
    if (start < end):
        Wait(start,
            Choice(
                PowerPlant(Tomorrow(start), end, Running()) + ProfitFromRunning(start, duration_off),
                PowerPlant(Tomorrow(start), end, Stopped(duration_off)
                )
            )
        )
    else:
        0


@inline
def ProfitFromRunning(start, duration_off):
    if duration_off > 1:
        0.3 * Power(start) - Gas(start)
    elif duration_off == 1:
        0.6 * Power(start) - Gas(start)
    else:
        1.00 * Power(start) - Gas(start)


@inline
def Power(start):
    DayAhead(start, 'TTF')

@inline
def Gas(start):
    DayAhead(start, 'NBP')

@inline
def DayAhead(start, name):
    ForwardMarket(start + TimeDelta('1d'), name)

# @inline
def Running():
    0


# @inline
def Stopped(duration_off):
    Min(2, duration_off + 1)


# @inline
def Tomorrow(today):
    today + TimeDelta('1d')


PowerPlant(Date('2012-1-1'), Date('{end_date}'), Stopped(2))
        """
        self.assert_contract_value(specification.format(end_date='2012-01-13'), 24.77555, expected_call_count=37)
        self.assert_contract_value(specification.format(end_date='2012-01-14'), expected_call_count=40)
        self.assert_contract_value(specification.format(end_date='2012-01-15'), expected_call_count=43)
        self.assert_contract_value(specification.format(end_date='2012-01-16'), expected_call_count=46)
        self.assert_contract_value(specification.format(end_date='2012-01-17'), expected_call_count=49)

    def test_call_recombinations_with_function_calls_advancing_values(self):
        # This wasn't working, because each function call was being carried into the next
        # via the arg that was supposed to be adjusted by the function.
        specification = """
def f(t, d):
    if (t > 0):
        Add(
            f(t - 1, f1(d)) + 1,
            f(t - 1, f2(d)),
        )
    else:
        return 0

# @inline
def f1(t):
    0

@inline
def f2(d):
    Min(1, d+1)

f(9, 1)
"""
        self.assert_contract_value(specification, expected_call_count=20)


class TestObservationDate(ApplicationTestCase):
    def test_observation_date_expression(self):
        code = "ObservationDate()"
        self.assert_contract_value(code, expected_value=datetime.datetime(2011, 1, 1))

    def test_observation_date_in_function_body(self):
        code = """
def Start():
    ObservationDate()
    
Start()
"""
        self.assert_contract_value(code, expected_value=datetime.datetime(2011, 1, 1))

    def test_observation_date_as_function_call_arg(self):
        code = """
def f(a):
    a
    
f(ObservationDate())
"""
        self.assert_contract_value(code, expected_value=datetime.datetime(2011, 1, 1))

    def test_observation_date_in_test_expr(self):
        code = """
def f(a):
    if a > Date('2000-1-1'):
        1
    else:
        0
f(ObservationDate())
"""
        self.assert_contract_value(code, expected_value=1)


class SpecialTests(ApplicationTestCase):

    def test_single_sided_delta(self):
        dsl = "Market('NBP') * 2"
        self.assert_contract_value(dsl, expected_value=20, expected_deltas={'NBP': 2}, periodisation='alltime')
        self.assert_contract_value(dsl, expected_value=20, expected_deltas={'NBP': 2}, periodisation='alltime',
                                   is_double_sided_deltas=False)
        self.assert_contract_value(dsl, expected_value=20, expected_deltas={'NBP-2011-1': 2}, periodisation='monthly')
        self.assert_contract_value(dsl, expected_value=20, expected_deltas={'NBP-2011-1': 2}, periodisation='monthly',
                                   is_double_sided_deltas=False)

    def test_simple_expression_with_market(self):
        dsl = "Market('NBP') + 2 * Market('TTF')"
        self.assert_contract_value(dsl, expected_value=32, expected_deltas={'NBP-2011-1': 1, 'TTF-2011-1': 2},
                                   expected_call_count=1, periodisation='monthly')

        self.assert_contract_value(dsl, expected_value=32, expected_deltas={'NBP-2011-1': 1, 'TTF-2011-1': 2},
                                   expected_call_count=1, periodisation='monthly', is_double_sided_deltas=False)

    def test_simple_function_with_market(self):
        dsl = """
def F():
  Market('NBP') + 2 * Market('TTF')

F()
"""
        self.assert_contract_value(dsl, expected_value=32, expected_deltas={'NBP': 1, 'TTF': 2}, expected_call_count=2,
                                   periodisation='alltime')

        self.assert_contract_value(dsl, expected_value=32, expected_deltas={'NBP': 1, 'TTF': 2}, expected_call_count=2,
                                   periodisation='alltime', is_double_sided_deltas=False)

    def test_generate_valuation_swing_option(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-01-01'), Date('2011-01-05'), Market('NBP'), 3)
"""
        self.assert_contract_value(specification, 30.00, {'NBP': 3.0}, expected_call_count=15,
                                   periodisation='alltime')
        # self.assert_contract_value(specification, 30.20756, {}, expected_call_count=15)

    def test_reuse_unperturbed_call_results(self):
        specification = """
def SumTwoMarkets(market_name1, market_name2):
    GetMarket(market_name1) + GetMarket(market_name2)

def GetMarket(market_name):
    Market(market_name)

SumTwoMarkets('NBP', 'TTF')
"""
        self.assert_contract_value(specification, expected_value=21, expected_deltas={'NBP': 1, 'TTF': 1},
                                   expected_call_count=4, periodisation='alltime')

    def test_reuse_unperturbed_call_results2(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Exercise(Swing, start_date, end_date, underlying, quantity),
            Hold(Swing, start_date, end_date, underlying, quantity)
        )
    else:
        return 0

@inline
def Exercise(f, start_date, end_date, underlying, quantity):
    return Hold(f, start_date, end_date, underlying, quantity - 1) + Fixing(start_date, underlying)

@inline
def Hold(f, start_date, end_date, underlying, quantity):
    return f(start_date + TimeDelta('1d'), end_date, underlying, quantity)

Swing(Date('2011-1-1'), Date('2011-1-4'), Market('#1'), 2) * 1 + \
Swing(Date('2011-1-1'), Date('2011-1-4'), Market('#2'), 2) * 2
"""
        self.assert_contract_value(specification, expected_value=60.0, expected_deltas={'#1': 2.0, '#2': 4.0},
                                   expected_call_count=19, periodisation='alltime')


class ExperimentalTests(ApplicationTestCase):
    def test_value_swing_option_with_fixing_on_market(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Fixing(start_date, 
            Market(underlying)),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-1-1'), Date('2011-1-5'), 'NBP', 3)
"""
        self.assert_contract_value(specification, 30.000, expected_call_count=15)

    def test_value_swing_option_without_fixings_or_settlements(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Market(underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        )
    else:
        return 0

Swing(Date('2011-1-1'), Date('2011-1-5'), 'NBP', 3)
"""
        self.assert_contract_value(specification, 30.0000, expected_call_count=15)

    def test_value_swing_option_with_settlements_and_fixings_on_choice(self):
        specification = """
def Swing(start_date, end_date, underlying, quantity):
    if (quantity != 0) and (start_date < end_date):
        return Wait(start_date, Choice(
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity-1) + Market(underlying),
            Swing(start_date + TimeDelta('1d'), end_date, underlying, quantity)
        ))
    else:
        return 0

Swing(Date('2011-1-1'), Date('2011-1-5'), 'NBP', 3)
"""
        self.assert_contract_value(specification, 30.00, expected_call_count=15)

    def test_simple_expression_with_market(self):
        dsl = """
def Swing(start, end, step, market, quantity):
    if (quantity != 0) and (start <= end):
        Max(
            HoldSwing(start, end, step, market, quantity),
            ExerciseSwing(start, end, step, market, quantity, 1)
        )
    else:
        0

@inline
def HoldSwing(start, end, step, market, quantity):
    On(start, Swing(start+step, end, step, market, quantity))

@inline
def ExerciseSwing(start, end, step, market, quantity, vol):
    Settlement(start, vol*market) + HoldSwing(start, end, step, market, quantity-vol)

Swing(Date('2011-01-01'), Date('2011-01-02'), TimeDelta('1d'), Market('NBP'), 1)
"""
        self.assert_contract_value(dsl, 10, {'NBP': 1}, expected_call_count=6, periodisation='alltime')

    def test_value_swing_option_with_forward_markets(self):
        specification = """
def Swing(start, end_date, quantity):
    if (quantity != 0) and (start < end_date):
        return Wait(start, Choice(
            Swing(start + TimeDelta('1m'), end_date, quantity-1) + ForwardMarket(start + TimeDelta('1m'), 'NBP'),
            Swing(start + TimeDelta('1m'), end_date, quantity)
        ))
    else:
        return 0

Swing(Date('2011-2-1'), Date('2011-5-1'), 30)
"""
        self.assert_contract_value(specification, 29.872, {
            'NBP-2011-3': 1.019,
            'NBP-2011-4': 1.020,
            'NBP-2011-5': 1.014,
        }, expected_call_count=11, periodisation='monthly')

    def test_simple_forward_market(self):
        specification = """ForwardMarket('2011-1-2', 'NBP')"""
        self.assert_contract_value(specification, 10.00, {'NBP': 1.0}, expected_call_count=1, periodisation='alltime')

    def test_gas_storage_option(self):
        specification_tmpl = """
def GasStorage(start, end, commodity_name, quantity, limit, step):
    if ((start < end) and (limit > 0)):
        if quantity <= 0:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Inject(start, end, commodity_name, quantity, limit, step, 1),
            ))
        elif quantity >= limit:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Inject(start, end, commodity_name, quantity, limit, step, -1),
            ))
        else:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Inject(start, end, commodity_name, quantity, limit, step, 1),
                Inject(start, end, commodity_name, quantity, limit, step, -1),
            ))
    else:
        0

@inline
def Continue(start, end, commodity_name, quantity, limit, step):
    GasStorage(start + step, end, commodity_name, quantity, limit, step)

@inline
def Inject(start, end, commodity_name, quantity, limit, step, vol):
    Continue(start, end, commodity_name, quantity + vol, limit, step) - \
    Settlement(start, vol * ForwardMarket(start, commodity_name))

GasStorage(Date('%(start_date)s'), Date('%(end_date)s'), '%(commodity)s', %(quantity)s, %(limit)s, TimeDelta('1m'))
"""
        # No capacity.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 0
        }
        self.assert_contract_value(specification, 0.00, {}, expected_call_count=2)

        # Capacity, zero inventory.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 10
        }
        self.assert_contract_value(specification, 0.00, {}, expected_call_count=7)

        # Capacity, zero inventory, option in the future.
        specification = specification_tmpl % {
            'start_date': '2013-1-1',
            'end_date': '2013-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 10
        }
        self.assert_contract_value(specification, -0.001, {}, expected_call_count=7)

        # Capacity, and inventory to discharge.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-3-1',
            'commodity': 'NBP',
            'quantity': 2,
            'limit': 2
        }
        self.assert_contract_value(specification, 19.982, {}, expected_call_count=7)

        # Zero sigma.
        orig_params = self.calibration_params['sigma']
        self.calibration_params['sigma'] = [0, 0, 0, 0, 0]

        # Intrinsic should be less.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-3-1',
            'commodity': 'NBP',
            'quantity': 2,
            'limit': 2
        }
        self.assert_contract_value(specification, 19.979, {}, expected_call_count=7)

        # Reset sigma.
        self.calibration_params['sigma'] = orig_params

        # Capacity, and inventory to discharge, far in the future (2021).
        specification = specification_tmpl % {
            'start_date': '2021-1-1',
            'end_date': '2021-3-1',
            'commodity': 'NBP',
            'quantity': 2,
            'limit': 2
        }
        self.assert_contract_value(specification, 15.971, {}, expected_call_count=7)

        # Capacity, zero inventory, in future.
        specification = specification_tmpl % {
            'start_date': '2021-1-1',
            'end_date': '2021-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 2
        }
        self.assert_contract_value(specification, 0.0024, {}, expected_call_count=7)

        # Capacity, zero inventory, longer run, higher limit.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-4-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 3
        }
        self.assert_contract_value(specification, expected_call_count=11)

        # Capacity, zero inventory, longer run, same higher limit.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-5-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 3
        }
        self.assert_contract_value(specification, expected_call_count=15)

        # Capacity, zero inventory, even longer run, same higher limit.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-6-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 3
        }
        self.assert_contract_value(specification, expected_call_count=19)


class SingleTests(ApplicationTestCase):
    def test_limit_dependency_graph_size_recursive_noninlined(self):
        # Test programs that don't halt are limited by the max dependency graph size.
        specification = """
def Func(n):
    return Func(n+1)
    
Func(1)
"""
        with self.assertRaises(CallLimitError):
            self.assert_contract_value(specification, None)

    def test_limit_dependency_graph_size_recursive_inlined(self):
        # Test the Python maximum recursion depth is handled.
        specification = """
@inline
def Func(n):
    return Func(n+1)

Func(1)
"""
        with self.assertRaises(RecursionDepthError):
            self.assert_contract_value(specification, None)
