import datetime
import threading
import unittest

import sys

import math
from dateutil.relativedelta import relativedelta
import time
from six import print_

from quantdsl.domain.services.parser import dsl_parse
from quantdsl.domain.services.price_processes import get_price_process
from quantdsl.exceptions import DslSyntaxError, DslError
from quantdsl.semantics import Add, Compare, Date, Div, DslExpression, DslNamespace, Fixing, FloorDiv, FunctionCall, \
    FunctionDef, If, IfExp, Max, Mod, Module, Mult, Name, Number, On, Pow, String, Sub, TimeDelta, UnarySub, \
    StochasticObject, Market, AbstractMarket
from quantdsl.domain.services.dependency_graphs import extract_defs_and_exprs
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME, DEFAULT_PATH_COUNT
from quantdsl.syntax import DslParser


class TestDslParser(unittest.TestCase):
    def setUp(self):
        self.p = DslParser()

    def test_empty_string(self):
        # Empty string can be parsed as a module, but not compiled into an expression, or evaluated.
        self.assertTrue(isinstance(dsl_parse(""), Module))
        self.assertRaises(DslSyntaxError, dsl_compile, "")
        self.assertRaises(DslSyntaxError, dsl_eval, "")

    def test_invalid_source(self):
        # Check a non-valid Python string raises an error.
        self.assertRaises(DslSyntaxError, dsl_parse, "1 +")
        # Check a non-string raises an error.
        self.assertRaises(DslSyntaxError, dsl_compile, 1)

    def test_unsupported_elements(self):
        # Check an unsupported element isn't parsed.
        self.assertRaises(DslSyntaxError, dsl_parse, "pass")

    def assertDslExprTypeValue(self, dsl_source, expectedDslType, expectedDslValue, **compile_kwds):
        # Assumes dsl_source is just one statement.
        dsl_module = dsl_parse(dsl_source)

        assert isinstance(dsl_module, Module)

        # Check the parsed DSL can be rendered as a string that is equal to the original source.
        self.assertEqual(str(dsl_module).strip(), dsl_source.strip())

        # Assume this test is dealing with modules that have one statement only.
        dsl_expr = dsl_module.body[0]

        # Check expression type.
        assert isinstance(dsl_expr, DslExpression)
        self.assertIsInstance(dsl_expr, expectedDslType)

        # Compile the module into an simple DSL expression object (no variables or calls to function defs).
        dsl_expr = dsl_expr.reduce(DslNamespace(compile_kwds), DslNamespace())

        # Evaluate the compiled expression.
        dsl_value = dsl_expr.evaluate()

        self.assertEqual(dsl_value, expectedDslValue)

    def test_num(self):
        self.assertDslExprTypeValue("0", Number, 0)
        self.assertDslExprTypeValue("5", Number, 5)
        self.assertDslExprTypeValue("-5", (Number, UnarySub), -5)
        self.assertDslExprTypeValue("5.1", Number, 5.1)
        self.assertDslExprTypeValue("-5.1", (Number, UnarySub), -5.1)

    def test_str(self):
        self.assertDslExprTypeValue("''", String, "")
        self.assertDslExprTypeValue("'#1'", String, "#1")

        # We can have comments too, but comments and trailing whitespaces are ignored.
        dsl = dsl_compile("'#1'  # This is a comment.")
        self.assertIsInstance(dsl, String)
        self.assertEqual(dsl.evaluate(), '#1')

    def test_name(self):
        self.assertDslExprTypeValue("foo", Name, "f", foo='f')
        self.assertDslExprTypeValue("foo", Name, 20, foo=20)

    def test_unaryop(self):
        self.assertDslExprTypeValue("-bar", UnarySub, -5, bar=5)
        self.assertDslExprTypeValue("-Max(1, 3)", UnarySub, -3)
        self.assertDslExprTypeValue("-Max(-1, -3)", UnarySub, 1)
        self.assertDslExprTypeValue("-Max(bar - 4, -9)", UnarySub, 8, bar=-4)

        # Check unsupported unary operators cause DSL errors.
        self.assertRaises(DslSyntaxError, dsl_parse, "~bar")

    def test_binop(self):
        self.assertDslExprTypeValue("5 + 2", Add, 7)
        self.assertDslExprTypeValue("5 - 2", Sub, 3)
        self.assertDslExprTypeValue("5 * 2", Mult, 10)
        self.assertDslExprTypeValue("5 / 2", Div, 2.5)
        self.assertDslExprTypeValue("5.0 / 2", Div, 2.5)
        self.assertDslExprTypeValue("5 / 2.0", Div, 2.5)
        self.assertDslExprTypeValue("5 // 2", FloorDiv, 2)
        self.assertDslExprTypeValue("5 ** 2", Pow, 25)
        self.assertDslExprTypeValue("5 % 2", Mod, 1)

        # Check unsupported binary operators cause DSL errors.
        self.assertRaises(DslSyntaxError, dsl_parse, "2 << 1")  # Bit shift left.
        self.assertRaises(DslSyntaxError, dsl_parse, "2 >> 1")  # Bit shift right.
        self.assertRaises(DslSyntaxError, dsl_parse, "2 & 1")  # Bitwise 'and'.
        self.assertRaises(DslSyntaxError, dsl_parse, "2 | 1")  # Complement
        self.assertRaises(DslSyntaxError, dsl_parse, "2 ^ 1")  # Bitwise exclusive or.

    def test_compare(self):
        self.assertDslExprTypeValue("1 == 1", Compare, True)
        self.assertDslExprTypeValue("1 == 2", Compare, False)
        self.assertDslExprTypeValue("2 != 1", Compare, True)
        self.assertDslExprTypeValue("1 != 1", Compare, False)
        self.assertDslExprTypeValue("1 < 2", Compare, True)
        self.assertDslExprTypeValue("1 < 1", Compare, False)
        self.assertDslExprTypeValue("1 <= 1", Compare, True)
        self.assertDslExprTypeValue("1 <= 2", Compare, True)
        self.assertDslExprTypeValue("1 <= 0", Compare, False)
        self.assertDslExprTypeValue("1 > 0", Compare, True)
        self.assertDslExprTypeValue("1 > 1", Compare, False)
        self.assertDslExprTypeValue("1 >= 1", Compare, True)
        self.assertDslExprTypeValue("2 >= 1", Compare, True)
        self.assertDslExprTypeValue("0 >= 1", Compare, False)

        # Multiple operators
        self.assertDslExprTypeValue("1 < 2 < 3", Compare, True)
        self.assertDslExprTypeValue("1 <= 2 <= 3", Compare, True)
        self.assertDslExprTypeValue("1 <= 2 >= 0", Compare, True)
        self.assertDslExprTypeValue("1 <= 2 >= 3", Compare, False)

    def test_ifexpr(self):
        self.assertDslExprTypeValue("foo if bar else 0", IfExp, 0, foo=0, bar=1)
        self.assertDslExprTypeValue("foo if bar else 0", IfExp, 2, foo=2, bar=1)
        self.assertDslExprTypeValue("foo if bar else 0", IfExp, 4, foo=4, bar=1)
        self.assertDslExprTypeValue("foo if bar else 0", IfExp, 0, foo=5, bar=0)

        self.assertDslExprTypeValue("6 if 1 else 7 if 1 else 8", IfExp, 6)
        self.assertDslExprTypeValue("6 if 0 else 7 if 1 else 8", IfExp, 7)
        self.assertDslExprTypeValue("6 if 0 else 7 if 0 else 8", IfExp, 8)

    def test_if(self):
        dsl_source = """
if bar:
    foo
else:
    0
"""
        self.assertDslExprTypeValue(dsl_source, If, 0, foo=0, bar=1)
        self.assertDslExprTypeValue(dsl_source, If, 2, foo=2, bar=1)
        self.assertDslExprTypeValue(dsl_source, If, 4, foo=4, bar=1)
        self.assertDslExprTypeValue(dsl_source, If, 0, foo=5, bar=0)

        dsl_source = """
if bar:
    foo
elif hee:
    haa
else:
    -1
"""
        self.assertDslExprTypeValue(dsl_source, If, 0, foo=0, bar=1, hee=1, haa=3)
        self.assertDslExprTypeValue(dsl_source, If, 2, foo=2, bar=1, hee=1, haa=3)
        self.assertDslExprTypeValue(dsl_source, If, 4, foo=4, bar=1, hee=1, haa=3)
        self.assertDslExprTypeValue(dsl_source, If, 3, foo=6, bar=0, hee=1, haa=3)
        self.assertDslExprTypeValue(dsl_source, If, -1, foo=6, bar=0, hee=0, haa=3)

    def test_call(self):
        self.assertDslExprTypeValue("Max(1, 2)", Max, 2)
        self.assertDslExprTypeValue("Max(Max(1, 2), 3)", Max, 3)
        self.assertDslExprTypeValue("Max(Max(Max(1, 2), 3), 4)", Max, 4)

        self.assertDslExprTypeValue("Max(1 + 4, 2)", Max, 5)

    def test_date(self):
        self.assertDslExprTypeValue("Date('2014-12-31')", Date, datetime.datetime(2014, 12, 31))

    def test_date_timedelta(self):
        dsl = dsl_compile("TimeDelta('2d')")
        self.assertIsInstance(dsl, TimeDelta)
        self.assertEqual(dsl.evaluate(), relativedelta(days=2))

        dsl = dsl_compile("TimeDelta('2m')")
        self.assertIsInstance(dsl, TimeDelta)
        self.assertEqual(dsl.evaluate(), relativedelta(months=2))

        dsl = dsl_compile("TimeDelta('2y')")
        self.assertIsInstance(dsl, TimeDelta)
        self.assertEqual(dsl.evaluate(), relativedelta(years=2))

        # Some date arithmetic...
        dsl = dsl_compile("Date('2014-12-31') - TimeDelta('1d')")
        self.assertIsInstance(dsl, Sub)
        self.assertEqual(dsl.evaluate(), datetime.datetime(2014, 12, 30))

        dsl = dsl_compile("Date('2014-12-29') + TimeDelta('1d')")
        self.assertIsInstance(dsl, Add)
        self.assertEqual(dsl.evaluate(), datetime.datetime(2014, 12, 30))

        dsl = dsl_compile("Date('2014-12-29') + TimeDelta('1m')")
        self.assertIsInstance(dsl, Add)
        self.assertEqual(dsl.evaluate(), datetime.datetime(2015, 1, 29))

        dsl = dsl_compile("Date('2014-12-29') + TimeDelta('1y')")
        self.assertIsInstance(dsl, Add)
        self.assertEqual(dsl.evaluate(), datetime.datetime(2015, 12, 29))

        dsl = dsl_compile("2 * TimeDelta('1d')")
        self.assertIsInstance(dsl, Mult)
        self.assertEqual(dsl.evaluate(), relativedelta(days=2))

        dsl = dsl_compile("2 * TimeDelta('1m')")
        self.assertIsInstance(dsl, Mult)
        self.assertEqual(dsl.evaluate(), relativedelta(months=2))

        dsl = dsl_compile("2 * TimeDelta('1y')")
        self.assertIsInstance(dsl, Mult)
        self.assertEqual(dsl.evaluate(), relativedelta(years=2))

    def test_date_comparisons(self):
        self.assertDslExprTypeValue("Date('2014-12-30') < Date('2014-12-31')", Compare, True)
        self.assertDslExprTypeValue("Date('2014-12-31') < Date('2014-12-30')", Compare, False)
        self.assertDslExprTypeValue("Date('2014-12-31') == Date('2014-12-31')", Compare, True)
        self.assertDslExprTypeValue("Date('2014-12-30') == Date('2014-12-31')", Compare, False)
        self.assertDslExprTypeValue("Date('2014-12-30') != Date('2014-12-31')", Compare, True)
        self.assertDslExprTypeValue("Date('2014-12-31') != Date('2014-12-31')", Compare, False)

    def test_fixing(self):
        dsl_source = "Fixing('2012-01-01', 5)"
        dsl = dsl_compile(dsl_source)
        self.assertEqual(dsl_source, str(dsl))
        self.assertIsInstance(dsl, Fixing)
        self.assertEqual(dsl.evaluate(), 5)

    def test_on(self):
        dsl_source = "On('2012-01-01', 5)"
        dsl = dsl_compile(dsl_source)
        self.assertEqual(dsl_source, str(dsl))
        self.assertIsInstance(dsl, On)
        self.assertEqual(dsl.evaluate(), 5)

    def test_functiondef_simple(self):
        # Simple one-line body.
        dsl = dsl_compile("def a(): 1")
        self.assertIsInstance(dsl, FunctionDef)
        self.assertEqual(dsl.name, 'a')
        self.assertEqual(len(dsl.call_arg_names), 0)
        self.assertEqual(len(dsl.call_cache), 0)
        aExpr = dsl.apply()
        self.assertIsInstance(aExpr, Number)
        aValue = aExpr.evaluate()
        self.assertEqual(aValue, 1)

        # Check the call is in the cache.
        self.assertEqual(len(dsl.call_cache), 1)

        # Check a freshly parsed function def has a fresh call cache.
        dsl = dsl_compile("def a(): 1")
        self.assertEqual(len(dsl.call_cache), 0)

    def test_functiondef_dsl_max(self):
        # Simple one-line body.
        dsl = dsl_compile("def a(b): return Max(b, 2)")
        self.assertIsInstance(dsl, FunctionDef)
        self.assertEqual(dsl.name, 'a')
        self.assertEqual(dsl.call_arg_names[0], 'b')
        self.assertIsInstance(dsl.body, Max)
        self.assertIsInstance(dsl.body.left, Name)
        self.assertIsInstance(dsl.body.right, Number)
        self.assertEqual(dsl.body.evaluate(b=0), 2)
        self.assertEqual(dsl.body.evaluate(b=4), 4)
        a0 = dsl.apply(b=0)
        self.assertEqual(a0.evaluate(), 2)
        a4 = dsl.apply(b=4)
        self.assertEqual(a4.evaluate(), 4)

        # Return statement is optional, value of last expression is returned.
        dsl = dsl_compile("def a(b): Max(b, 2)")
        self.assertIsInstance(dsl, FunctionDef)
        self.assertEqual(dsl.name, 'a')
        self.assertEqual(dsl.apply(b=0).evaluate(), 2)
        self.assertEqual(dsl.apply(b=4).evaluate(), 4)

    def test_functiondef_dsl_max_conditional(self):
        # Conditional call.
        dsl = dsl_compile("def a(b): Max(b, 2) if b != 0 else 0")
        self.assertIsInstance(dsl, FunctionDef)
        self.assertEqual(dsl.name, 'a')
        self.assertEqual(dsl.call_arg_names[0], 'b')
        self.assertIsInstance(dsl.body, IfExp)
        self.assertEqual(dsl.body.test.evaluate(b=1), True)  # b != 0
        self.assertEqual(dsl.body.test.evaluate(b=0), False)
        self.assertEqual(dsl.body.body.evaluate(b=4), 4)  # Max(b, 2)
        self.assertEqual(dsl.body.body.evaluate(b=0), 2)

        a0 = dsl.apply(b=0)
        self.assertIsInstance(a0, Number)

        a1 = dsl.apply(b=1)
        self.assertIsInstance(a1, Max)
        self.assertIsInstance(a1.left, Number)
        self.assertIsInstance(a1.right, Number)
        self.assertEqual(a1.left.evaluate(), 1)
        self.assertEqual(a1.right.evaluate(), 2)
        self.assertEqual(a1.evaluate(), 2)

        a3 = dsl.apply(b=3)
        self.assertIsInstance(a3, Max)
        self.assertIsInstance(a3.left, Number)
        self.assertIsInstance(a3.right, Number)
        self.assertEqual(a3.left.evaluate(), 3)
        self.assertEqual(a3.right.evaluate(), 2)
        self.assertEqual(a3.evaluate(), 3)

    def test_functiondef_recursive_cached(self):
        # Recursive call.
        fib_def = dsl_compile("def fib(n): return fib(n-1) + fib(n-2) if n > 2 else n")

        # Check the parsed function def DSL object.
        self.assertIsInstance(fib_def, FunctionDef)
        self.assertFalse(fib_def.call_cache)
        self.assertEqual(fib_def.name, 'fib')
        self.assertEqual(fib_def.call_arg_names[0], 'n')
        self.assertIsInstance(fib_def.body, IfExp)
        self.assertEqual(fib_def.body.test.evaluate(n=3), True)
        self.assertEqual(fib_def.body.test.evaluate(n=2), False)
        self.assertIsInstance(fib_def.body.body, Add)
        self.assertIsInstance(fib_def.body.body.left, FunctionCall)
        self.assertIsInstance(fib_def.body.body.left.functionDefName, Name)
        self.assertIsInstance(fib_def.body.body.left.callArgExprs, list)
        self.assertIsInstance(fib_def.body.body.left.callArgExprs[0], Sub)
        self.assertIsInstance(fib_def.body.body.left.callArgExprs[0].left, Name)
        self.assertEqual(fib_def.body.body.left.callArgExprs[0].left.name, 'n')
        self.assertIsInstance(fib_def.body.body.left.callArgExprs[0].right, Number)
        self.assertEqual(fib_def.body.body.left.callArgExprs[0].right.value, 1)

        # Evaluate the function with different values of n.
        # n = 1
        fib_expr = fib_def.apply(n=1)
        self.assertIsInstance(fib_expr, Number)
        fib_value = fib_expr.evaluate()
        self.assertIsInstance(fib_value, (int, float))
        self.assertEqual(fib_value, 1)

        # Check call cache has one call.
        self.assertEqual(len(fib_def.call_cache), 1)

        # n = 2
        fib_expr = fib_def.apply(n=2)
        self.assertIsInstance(fib_expr, Number)
        fib_value = fib_expr.evaluate()
        self.assertIsInstance(fib_value, (int, float))
        self.assertEqual(fib_value, 2)

        # Check call cache has two calls.
        self.assertEqual(len(fib_def.call_cache), 2)

        # n = 3
        fib_expr = fib_def.apply(n=3)
        self.assertIsInstance(fib_expr, Add)
        self.assertIsInstance(fib_expr.left, Number)
        self.assertIsInstance(fib_expr.right, Number)
        fib_value = fib_expr.evaluate()
        self.assertIsInstance(fib_value, (int, float))
        self.assertEqual(fib_value, 3)

        # Check call cache has three calls.
        self.assertEqual(len(fib_def.call_cache), 3)

        # n = 4
        fib_expr = fib_def.apply(n=4)
        self.assertIsInstance(fib_expr, Add)
        self.assertIsInstance(fib_expr.left, Add)
        self.assertIsInstance(fib_expr.left.left, Number)
        self.assertEqual(fib_expr.left.left.evaluate(), 2)  # fib(2) -> 2
        self.assertIsInstance(fib_expr.left.right, Number)
        self.assertEqual(fib_expr.left.right.evaluate(), 1)
        self.assertIsInstance(fib_expr.right, Number)
        self.assertEqual(fib_expr.right.evaluate(), 2)  # fib(2) -> 2    *repeats
        # Check repeated calls have resulted in the same object.
        self.assertEqual(fib_expr.left.left, fib_expr.right)  # fib(2)

        fib_value = fib_expr.evaluate()
        self.assertIsInstance(fib_value, (int, float))
        self.assertEqual(fib_value, 5)

        # Check call cache has four calls.
        self.assertEqual(len(fib_def.call_cache), 4)

        # n = 5
        fib_expr = fib_def.apply(n=5)
        self.assertIsInstance(fib_expr, Add)  # fib(4) + fib(3)
        self.assertIsInstance(fib_expr.left, Add)  # fib(4) -> fib(3) + fib(2)
        self.assertIsInstance(fib_expr.left.left, Add)  # fib(3) -> fib(2) + fib(1)
        self.assertIsInstance(fib_expr.left.left.left, Number)  # fib(2) -> 2
        self.assertEqual(fib_expr.left.left.left.evaluate(), 2)
        self.assertIsInstance(fib_expr.left.left.right, Number)  # fib(1) -> 1
        self.assertEqual(fib_expr.left.left.right.evaluate(), 1)
        self.assertIsInstance(fib_expr.left.right, Number)  # fib(2) -> 2    *repeats
        self.assertEqual(fib_expr.left.right.evaluate(), 2)
        self.assertIsInstance(fib_expr.right, Add)  # fib(3) -> fib(2) + fib(1)    *repeats
        self.assertIsInstance(fib_expr.right.left, Number)  # fib(2) -> 2    *repeats
        self.assertEqual(fib_expr.right.left.evaluate(), 2)
        self.assertIsInstance(fib_expr.right.right, Number)  # fib(1) -> 1    *repeats
        self.assertEqual(fib_expr.right.right.evaluate(), 1)

        # Check repeated calls have resulted in the same object.
        self.assertEqual(fib_expr.right.right, fib_expr.left.left.right)  # fib(1)
        self.assertEqual(fib_expr.right.left, fib_expr.left.left.left)  # fib(2)
        self.assertEqual(fib_expr.left.right, fib_expr.left.left.left)  # fib(2)
        self.assertEqual(fib_expr.right, fib_expr.left.left)  # fib(3)

        fib_value = fib_expr.evaluate()
        self.assertIsInstance(fib_value, (int, float))
        self.assertEqual(fib_value, 8)

        # Check call cache has five calls.
        self.assertEqual(len(fib_def.call_cache), 5)

        # Just check call cache with fib(5) with fresh parser.
        fib_def = dsl_compile("def fib(n): return fib(n-1) + fib(n-2) if n > 2 else n")
        assert isinstance(fib_def, FunctionDef)
        self.assertEqual(len(fib_def.call_cache), 0)
        fib_expr = fib_def.apply(n=5)
        self.assertEqual(len(fib_def.call_cache), 5)
        self.assertEqual(fib_expr.evaluate(), 8)
        self.assertEqual(len(fib_def.call_cache), 5)

    def test_module_block(self):
        # Expression with one function def.
        dsl_source = """
def sqr(n):
    n ** 2
sqr(3)
"""
        dsl_module = dsl_parse(dsl_source)
        self.assertIsInstance(dsl_module, Module)
        self.assertEqual(str(dsl_module), dsl_source.strip())

        dsl_expr = dsl_compile(dsl_source)
        self.assertEqual(dsl_expr.evaluate(), 9)

        dsl_value = dsl_eval(dsl_source)
        self.assertEqual(dsl_value, 9)

        # Expression with two function defs.
        dsl_source = """
def add(a, b):
    a + b
def mul(a, b):
    a if b == 1 else add(a, mul(a, b - 1))
mul(3, 3)
"""
        dsl_module = dsl_parse(dsl_source)
        self.assertIsInstance(dsl_module, Module)
        self.assertEqual(str(dsl_module), dsl_source.strip())

        dsl_expr = dsl_compile(dsl_source)
        #        self.assertEqual(str(dsl_expr), "")
        self.assertEqual(dsl_expr.evaluate(), 9)

        dsl_value = dsl_eval(dsl_source)
        self.assertEqual(dsl_value, 9)


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
    assert isinstance(dsl_expr, DslExpression), type(dsl_expr)

    if is_verbose:
        print_("Duration of compilation: %s" % compile_time_delta)
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
                    print_(
                        "Simulating future prices for Market%s '%s' from observation time %s through fixing dates: "
                        "%s." % (
                            '' if len(market_names) == 1 else 's',
                            ", ".join(market_names),
                            "'%04d-%02d-%02d'" % (observation_date.year, observation_date.month, observation_date.day),
                            # Todo: Only print first and last few, if there are loads.
                            ", ".join(["'%04d-%02d-%02d'" % (d.year, d.month, d.day) for d in fixing_dates[:8]]) + \
                            (", [...]" if len(fixing_dates) > 9 else '') + \
                            ((", '%04d-%02d-%02d'" % (
                            fixing_dates[-1].year, fixing_dates[-1].month, fixing_dates[-1].day)) if len(
                                fixing_dates) > 8 else '')
                        ))
                    print_()

                # Simulate the future prices.
                all_market_prices = price_process.simulate_future_prices(market_names, fixing_dates, observation_date,
                                                                         path_count, market_calibration)

                # Add future price simulation to evaluation_kwds.
                evaluation_kwds['all_market_prices'] = all_market_prices

    # Initialise the evaluation timer variable (needed by showProgress thread).
    evalStartTime = None

    if is_parallel:
        if is_verbose:

            len_stubbed_exprs = len(dsl_expr.stubbed_calls)
            lenLeafIds = len(dsl_expr.leaf_ids)

            msg = "Evaluating %d expressions (%d %s) with " % (
            len_stubbed_exprs, lenLeafIds, 'leaf' if lenLeafIds == 1 else 'leaves')
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
                    sys.stdout.write(
                        "\rProgress: %01.2f%% (%s/%s) %s " % (progress, lenResults, len_stubbed_exprs, rateStr))
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


def dsl_compile(dsl_source, filename='<unknown>', dsl_classes=None, compile_kwds=None, **extraCompileKwds):
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
    return compile_dsl_module(dsl_module, DslNamespace(), compile_kwds)


def find_delivery_points(dsl_expr):
    # Find all unique market names.
    all_delivery_points = set()
    for dsl_market in dsl_expr.find_instances(dsl_type=AbstractMarket):
        # assert isinstance(dsl_market, Market)
        delivery_point = dsl_market.get_delivery_point()
        if delivery_point not in all_delivery_points:  # Deduplicate.
            all_delivery_points.add(delivery_point)
            yield delivery_point


def list_fixing_dates(dsl_expr):
    # Find all unique fixing dates.
    return sorted(list(find_fixing_dates(dsl_expr)))


def find_fixing_dates(dsl_expr):
    for dsl_fixing in dsl_expr.find_instances(dsl_type=Fixing):
        # assert isinstance(dsl_fixing, Fixing)
        if dsl_fixing.date is not None:
            yield dsl_fixing.date


def compile_dsl_module(dsl_module, dsl_locals=None, dsl_globals=None):
    """
    Returns something that can be evaluated.
    """

    # It's a module compilation, so create a new namespace "context".
    if dsl_locals is None:
        dsl_locals = {}
    dsl_locals = DslNamespace(dsl_locals)
    if dsl_globals is None:
        dsl_globals = {}
    dsl_globals = DslNamespace(dsl_globals)

    # Can't do much with an empty module.
    if len(dsl_module.body) == 0:
        raise DslSyntaxError('empty module', node=dsl_module.node)

    function_defs, expressions = extract_defs_and_exprs(dsl_module, dsl_globals)

    # Handle different combinations of functions and module level expressions in different ways.
    # Todo: Simplify this, but support library files first?
    # Can't meaningfully evaluate more than one expression (since assignments are not supported).
    if len(expressions) > 1:
        raise DslSyntaxError('more than one expression in module', node=expressions[1].node)

    # Can't meaningfully evaluate more than one function def without a module level expression.
    elif len(expressions) == 0 and len(function_defs) > 1:
        second_def = function_defs[1]
        raise DslSyntaxError('more than one function def in module without an expression',
                             '"def %s"' % second_def.name, node=function_defs[1].node)

    # If it's just a module with one function, then return the function def.
    elif len(expressions) == 0 and len(function_defs) == 1:
        return function_defs[0]

    # If there is one expression, reduce it with the function defs that it calls.
    else:
        assert len(expressions) == 1
        dsl_expr = expressions[0]
        # assert isinstance(dsl_expr, DslExpression), dsl_expr

        # Compile the module for a single threaded recursive operation (faster but not distributed,
        # so also limited in space and perhaps time). For smaller computations only.
        dsl_obj = dsl_expr.reduce(dsl_locals, DslNamespace(dsl_globals))
        return dsl_obj