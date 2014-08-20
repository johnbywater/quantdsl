import unittest
import datetime
import sys

import mock
import numpy

from quantdsl import DslParser, Number, Add, Max, On, Date, String, FunctionDef, \
    Name, Compare, IfExp, FunctionCall, Sub, QuantDslSyntaxError, Mult, Div, UnarySub, Pow, Mod, FloorDiv, \
    ExpressionStack, If, TimeDelta, Module, parse, compile, eval, utc
from quant.testunit import TestCase


def suite():
    return unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])


class TestDslParser(TestCase):

    def setUp(self):
        self.p = DslParser()

    def test_empty_string(self):
        self.assertTrue(isinstance(parse(""), Module))
        self.assertRaises(QuantDslSyntaxError, compile, "")
        self.assertRaises(QuantDslSyntaxError, eval, "")

    def assertDslExprTypeValue(self, dslSource, expectedDslType, expectedDslValue, **compileKwds):
        # Assumes dslSource is just one statement.
        dslModule = parse(dslSource)

        # Check the parsed DSL can be rendered as a string that is equal to the original source.
        self.assertEqual(str(dslModule).strip(), dslSource.strip())

        # Check the statement's expression type.
        dslExpr = dslModule.body[0]
        self.assertIsInstance(dslExpr, expectedDslType)

        # Compile the module into an simple DSL expression object (no variables or calls to function defs).
        dslExpr = dslModule.compile(compileKwds)

        # Evaluate the compiled expression.
        self.assertEqual(dslExpr.evaluate(), expectedDslValue)

    def test_num(self):
        self.assertDslExprTypeValue("0", Number, 0)
        self.assertDslExprTypeValue("5", Number, 5)
        self.assertDslExprTypeValue("-5", Number, -5)
        self.assertDslExprTypeValue("5.1", Number, 5.1)
        self.assertDslExprTypeValue("-5.1", Number, -5.1)

    def test_str(self):
        self.assertDslExprTypeValue("''", String, "")
        self.assertDslExprTypeValue("'#1'", String, "#1")

        # We can have comments too, but comments and trailing whitespaces are ignored.
        dsl = compile("'#1'  # This is a comment.")
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
        self.assertRaises(QuantDslSyntaxError, parse, "~bar")

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
        self.assertRaises(QuantDslSyntaxError, parse, "2 << 1")  # Bit shift left.
        self.assertRaises(QuantDslSyntaxError, parse, "2 >> 1")  # Bit shift right.
        self.assertRaises(QuantDslSyntaxError, parse, "2 & 1")  # Bitwise 'and'.
        self.assertRaises(QuantDslSyntaxError, parse, "2 | 1")  # Complement
        self.assertRaises(QuantDslSyntaxError, parse, "2 ^ 1")  # Bitwise exclusive or.

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
        dslSource = """
if bar:
    foo
else:
    0
"""
        self.assertDslExprTypeValue(dslSource, If, 0, foo=0, bar=1)
        self.assertDslExprTypeValue(dslSource, If, 2, foo=2, bar=1)
        self.assertDslExprTypeValue(dslSource, If, 4, foo=4, bar=1)
        self.assertDslExprTypeValue(dslSource, If, 0, foo=5, bar=0)

        dslSource = """
if bar:
    foo
elif hee:
    haa
else:
    -1
"""
        self.assertDslExprTypeValue(dslSource, If, 0, foo=0, bar=1, hee=1, haa=3)
        self.assertDslExprTypeValue(dslSource, If, 2, foo=2, bar=1, hee=1, haa=3)
        self.assertDslExprTypeValue(dslSource, If, 4, foo=4, bar=1, hee=1, haa=3)
        self.assertDslExprTypeValue(dslSource, If, 3, foo=6, bar=0, hee=1, haa=3)
        self.assertDslExprTypeValue(dslSource, If, -1, foo=6, bar=0, hee=0, haa=3)

    def test_call(self):
        self.assertDslExprTypeValue("Max(1, 2)", Max, 2)
        self.assertDslExprTypeValue("Max(Max(1, 2), 3)", Max, 3)
        self.assertDslExprTypeValue("Max(Max(Max(1, 2), 3), 4)", Max, 4)

        self.assertDslExprTypeValue("Max(1 + 4, 2)", Max, 5)

    def test_date(self):
        self.assertDslExprTypeValue("Date('2014-12-31')", Date, datetime.datetime(2014, 12, 31, tzinfo=utc))
        self.assertDslExprTypeValue("TimeDelta('1d')", TimeDelta, datetime.timedelta(1))

    def test_date_timedelta(self):
        # Some date arithmetic...
        dsl = compile("Date('2014-12-31') - TimeDelta('1d')")
        self.assertIsInstance(dsl, Sub)
        self.assertEqual(dsl.evaluate(), datetime.datetime(2014, 12, 30, tzinfo=utc))

        dsl = compile("Date('2014-12-29') + TimeDelta('1d')")
        self.assertIsInstance(dsl, Add)
        self.assertEqual(dsl.evaluate(), datetime.datetime(2014, 12, 30, tzinfo=utc))

        dsl = compile("2 * TimeDelta('1d')")
        self.assertIsInstance(dsl, Mult)
        self.assertEqual(dsl.evaluate(), datetime.timedelta(2))

    def test_date_comparisons(self):
        self.assertDslExprTypeValue("Date('2014-12-30') < Date('2014-12-31')", Compare, True)
        self.assertDslExprTypeValue("Date('2014-12-31') < Date('2014-12-30')", Compare, False)
        self.assertDslExprTypeValue("Date('2014-12-31') == Date('2014-12-31')", Compare, True)
        self.assertDslExprTypeValue("Date('2014-12-30') == Date('2014-12-31')", Compare, False)
        self.assertDslExprTypeValue("Date('2014-12-30') != Date('2014-12-31')", Compare, True)
        self.assertDslExprTypeValue("Date('2014-12-31') != Date('2014-12-31')", Compare, False)


    def test_on(self):
        dslSource = "On('2012-01-01', 5)"
        dsl = compile(dslSource)
        self.assertEqual(dslSource, str(dsl))
        self.assertIsInstance(dsl, On)
        self.assertEqual(dsl.evaluate(), 5)

    def test_functiondef_simple(self):
        # Simple one-line body.
        dsl = compile("def a(): 1")
        self.assertIsInstance(dsl, FunctionDef)
        self.assertEqual(dsl.name, 'a')
        self.assertEqual(len(dsl.callArgNames), 0)
        self.assertEqual(len(dsl.callCache), 0)
        aExpr = dsl.apply()
        self.assertIsInstance(aExpr, Number)
        aValue = aExpr.evaluate()
        self.assertEqual(aValue, 1)

        # Check the call is in the cache.
        self.assertEqual(len(dsl.callCache), 1)

        # Check a freshly parsed function def has a fresh call cache.
        dsl = compile("def a(): 1")
        self.assertEqual(len(dsl.callCache), 0)

    def test_functiondef_dsl_max(self):
        # Simple one-line body.
        dsl = compile("def a(b): return Max(b, 2)")
        self.assertIsInstance(dsl, FunctionDef)
        self.assertEqual(dsl.name, 'a')
        self.assertEqual(dsl.callArgNames[0], 'b')
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
        dsl = compile("def a(b): Max(b, 2)")
        self.assertIsInstance(dsl, FunctionDef)
        self.assertEqual(dsl.name, 'a')
        self.assertEqual(dsl.apply(b=0).evaluate(), 2)
        self.assertEqual(dsl.apply(b=4).evaluate(), 4)

    def test_functiondef_dsl_max_conditional(self):
        # Conditional call.
        dsl = compile("def a(b): Max(b, 2) if b != 0 else 0")
        self.assertIsInstance(dsl, FunctionDef)
        self.assertEqual(dsl.name, 'a')
        self.assertEqual(dsl.callArgNames[0], 'b')
        self.assertIsInstance(dsl.body, IfExp)
        self.assertEqual(dsl.body.test.evaluate(b=1), True)  # b != 0
        self.assertEqual(dsl.body.test.evaluate(b=0), False)
        self.assertEqual(dsl.body.body.evaluate(b=4), 4)     # Max(b, 2)
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
        fibDef = compile("def fib(n): return fib(n-1) + fib(n-2) if n > 2 else n")

        # Check the parsed function def DSL object.
        self.assertIsInstance(fibDef, FunctionDef)
        self.assertFalse(fibDef.callCache)
        self.assertEqual(fibDef.name, 'fib')
        self.assertEqual(fibDef.callArgNames[0], 'n')
        self.assertIsInstance(fibDef.body, IfExp)
        self.assertEqual(fibDef.body.test.evaluate(n=3), True)
        self.assertEqual(fibDef.body.test.evaluate(n=2), False)
        self.assertIsInstance(fibDef.body.body, Add)
        self.assertIsInstance(fibDef.body.body.left, FunctionCall)
        self.assertIsInstance(fibDef.body.body.left.functionDefName, Name)
        self.assertIsInstance(fibDef.body.body.left.callArgExprs, list)
        self.assertIsInstance(fibDef.body.body.left.callArgExprs[0], Sub)
        self.assertIsInstance(fibDef.body.body.left.callArgExprs[0].left, Name)
        self.assertEqual(fibDef.body.body.left.callArgExprs[0].left.name, 'n')
        self.assertIsInstance(fibDef.body.body.left.callArgExprs[0].right, Number)
        self.assertEqual(fibDef.body.body.left.callArgExprs[0].right.value, 1)

        # Evaluate the function with different values of n.
        # n = 1
        fibExpr = fibDef.apply(n=1)
        self.assertIsInstance(fibExpr, Number)
        fibValue = fibExpr.evaluate()
        self.assertIsInstance(fibValue, (int, float))
        self.assertEqual(fibValue, 1)

        # Check call cache has one call.
        self.assertEqual(len(fibDef.callCache), 1)

        # n = 2
        fibExpr = fibDef.apply(n=2)
        self.assertIsInstance(fibExpr, Number)
        fibValue = fibExpr.evaluate()
        self.assertIsInstance(fibValue, (int, float))
        self.assertEqual(fibValue, 2)

        # Check call cache has two calls.
        self.assertEqual(len(fibDef.callCache), 2)

        # n = 3
        fibExpr = fibDef.apply(n=3)
        self.assertIsInstance(fibExpr, Add)
        self.assertIsInstance(fibExpr.left, Number)
        self.assertIsInstance(fibExpr.right, Number)
        fibValue = fibExpr.evaluate()
        self.assertIsInstance(fibValue, (int, float))
        self.assertEqual(fibValue, 3)

        # Check call cache has three calls.
        self.assertEqual(len(fibDef.callCache), 3)

        # n = 4
        fibExpr = fibDef.apply(n=4)
        self.assertIsInstance(fibExpr, Add)
        self.assertIsInstance(fibExpr.left, Add)
        self.assertIsInstance(fibExpr.left.left, Number)
        self.assertEqual(fibExpr.left.left.evaluate(), 2)  # fib(2) -> 2
        self.assertIsInstance(fibExpr.left.right, Number)
        self.assertEqual(fibExpr.left.right.evaluate(), 1)
        self.assertIsInstance(fibExpr.right, Number)
        self.assertEqual(fibExpr.right.evaluate(), 2)  # fib(2) -> 2    *repeats
        # Check repeated calls have resulted in the same object.
        self.assertEqual(fibExpr.left.left, fibExpr.right)  # fib(2)

        fibValue = fibExpr.evaluate()
        self.assertIsInstance(fibValue, (int, float))
        self.assertEqual(fibValue, 5)

        # Check call cache has four calls.
        self.assertEqual(len(fibDef.callCache), 4)

        # n = 5
        fibExpr = fibDef.apply(n=5)
        self.assertIsInstance(fibExpr, Add)  # fib(4) + fib(3)
        self.assertIsInstance(fibExpr.left, Add)  # fib(4) -> fib(3) + fib(2)
        self.assertIsInstance(fibExpr.left.left, Add)  # fib(3) -> fib(2) + fib(1)
        self.assertIsInstance(fibExpr.left.left.left, Number)  # fib(2) -> 2
        self.assertEqual(fibExpr.left.left.left.evaluate(), 2)
        self.assertIsInstance(fibExpr.left.left.right, Number)  # fib(1) -> 1
        self.assertEqual(fibExpr.left.left.right.evaluate(), 1)
        self.assertIsInstance(fibExpr.left.right, Number)  # fib(2) -> 2    *repeats
        self.assertEqual(fibExpr.left.right.evaluate(), 2)
        self.assertIsInstance(fibExpr.right, Add)  # fib(3) -> fib(2) + fib(1)    *repeats
        self.assertIsInstance(fibExpr.right.left, Number)  # fib(2) -> 2    *repeats
        self.assertEqual(fibExpr.right.left.evaluate(), 2)
        self.assertIsInstance(fibExpr.right.right, Number)  # fib(1) -> 1    *repeats
        self.assertEqual(fibExpr.right.right.evaluate(), 1)

        # Check repeated calls have resulted in the same object.
        self.assertEqual(fibExpr.right.right, fibExpr.left.left.right)  # fib(1)
        self.assertEqual(fibExpr.right.left, fibExpr.left.left.left)  # fib(2)
        self.assertEqual(fibExpr.left.right, fibExpr.left.left.left)  # fib(2)
        self.assertEqual(fibExpr.right, fibExpr.left.left)  # fib(3)

        fibValue = fibExpr.evaluate()
        self.assertIsInstance(fibValue, (int, float))
        self.assertEqual(fibValue, 8)

        # Check call cache has five calls.
        self.assertEqual(len(fibDef.callCache), 5)

        # Just check call cache with fib(5) with fresh parser.
        fibDef = compile("def fib(n): return fib(n-1) + fib(n-2) if n > 2 else n")
        assert isinstance(fibDef, FunctionDef)
        self.assertEqual(len(fibDef.callCache), 0)
        fibExpr = fibDef.apply(n=5)
        self.assertEqual(len(fibDef.callCache), 5)
        self.assertEqual(fibExpr.evaluate(), 8)
        self.assertEqual(len(fibDef.callCache), 5)

    def test_module_block(self):
        # Expression with one function def.
        dslSource = """
def sqr(n):
    n ** 2
sqr(3)
"""
        dslModule = parse(dslSource)
        self.assertIsInstance(dslModule, Module)
        self.assertEqual(str(dslModule), dslSource.strip())

        dslExpr = compile(dslSource)
        self.assertEqual(dslExpr.evaluate(), 9)

        dslValue = eval(dslSource)
        self.assertEqual(dslValue, 9)

        # Expression with two function defs.
        dslSource = """
def add(a, b):
    a + b
def mul(a, b):
    a if b == 1 else add(a, mul(a, b - 1))
mul(3, 3)
"""
        dslModule = parse(dslSource)
        self.assertIsInstance(dslModule, Module)
        self.assertEqual(str(dslModule), dslSource.strip())

        dslExpr = compile(dslSource)
#        self.assertEqual(str(dslExpr), "")
        self.assertEqual(dslExpr.evaluate(), 9)

        dslValue = eval(dslSource)
        self.assertEqual(dslValue, 9)


    def test_parallel_fib(self):
        # Branching function calls.

        fibIndex = 6
        expectedValue = 13
        expectedLenStubbedExprs = fibIndex + 1

        dslSource = """
def fib(n): fib(n-1) + fib(n-2) if n > 2 else n
fib(%d)
""" % fibIndex

        # # Check the source works as a serial operation.
        # dslExpr = parse(dslSource, inParallel=False)
        # self.assertIsInstance(dslExpr, Add)
        # dslValue = dslExpr.evaluate()
        # self.assertEqual(dslValue, expectedValue)

        # Check the source works as a parallel operation.
        dslExpr = compile(dslSource, isParallel=True)

        # Expect an expression stack object.
        self.assertIsInstance(dslExpr, ExpressionStack)

        # Remember the number of stubbed exprs - will check it after the value.
        actualLenStubbedExprs = dslExpr.stubbedExprs.qsize()

        # Evaluate the stack.
        dslValue = dslExpr.evaluate()

        # Check the value is expected.
        self.assertEqual(dslValue, expectedValue)

        # Check the number of stubbed exprs is expected.
        self.assertEqual(actualLenStubbedExprs, expectedLenStubbedExprs)

        # Also check the runner call count is the same.
        self.assertEqual(dslExpr._runnerCallCount, expectedLenStubbedExprs)

    def test_parallel_american_option(self):
        # Branching function calls.

        expectedValue = 5
        expectedLenStubbedExprs = 7

        dslSource = """
# NB using Max instead of Choice, to save development time.

def Option(date, strike, underlying, alternative):
    return Wait(date, Max(underlying - strike, alternative))

def American(starts, ends, strike, underlying, step):
    Option(starts, strike, underlying, 0) if starts == ends else \
    Option(starts, strike, underlying, American(starts + step, ends, strike, underlying, step))

American(Date('2012-01-01'), Date('2012-01-03'), 5, 10, TimeDelta('1d'))
"""

        dslExpr = compile(dslSource, isParallel=True)

        # Expect an expression stack object.
        self.assertIsInstance(dslExpr, ExpressionStack)

        # Remember the number of stubbed exprs - will check it after the value.
        actualLenStubbedExprs = dslExpr.stubbedExprs.qsize()

        # Evaluate the stack.
        image = mock.Mock()
        image.priceProcess.getDurationYears.return_value = 1
        kwds = {
            'image': image,
            'interestRate': 0,
            'presentTime': datetime.datetime(2011, 1, 1),
        }
        dslValue = dslExpr.evaluate(**kwds)

        # Check the value is expected.
        self.assertEqual(dslValue, expectedValue)

        # Check the number of stubbed exprs is expected.
        self.assertEqual(actualLenStubbedExprs, expectedLenStubbedExprs)

    def test_parallel_swing_option(self):
        # Branching function calls.

        expectedValue = 20
        expectedLenStubbedExprs = 7

        dslSource = """
def Swing(starts, ends, underlying, quantity):
    if (quantity != 0) and (starts < ends):
        return Max(
            Swing(starts + TimeDelta('1d'), ends, underlying, quantity-1) \
            + Fixing(starts, underlying),
            Swing(starts + TimeDelta('1d'), ends, underlying, quantity)
        )
    else:
        return 0
Swing(Date('2012-01-01'), Date('2012-01-03'), 10, 500)
"""

        dslExpr = compile(dslSource, isParallel=True)

        # Remember the number of stubbed exprs - will check it after the value.
        actualLenStubbedExprs = dslExpr.stubbedExprs.qsize()

        # Evaluate the stack.
        image = mock.Mock()
        image.priceProcess.getDurationYears.return_value = 1
        kwds = {
            'image': image,
            'interestRate': 0,
            'presentTime': datetime.datetime(2011, 1, 1),
        }
        dslValue = dslExpr.evaluate(**kwds)

        # Check the value is expected.
        self.assertEqual(dslValue, expectedValue)

        # Check the number of stubbed exprs is expected.
        self.assertEqual(actualLenStubbedExprs, expectedLenStubbedExprs)

    def test_multiprocessed_swing_option(self):
        # Branching function calls.

        expectedValue = 20
        expectedLenStubbedExprs = 7

        dslSource = """
def Swing(starts, ends, underlying, quantity):
    if (quantity == 0) or (starts >= ends):
        0
    else:
        Choice(
            Swing(starts + TimeDelta('1d'), ends, underlying, quantity - 1) + Fixing(starts, underlying),
            Swing(starts + TimeDelta('1d'), ends, underlying, quantity)
        )
Swing(Date('2012-01-01'), Date('2012-01-03'), 10, 500)
"""

        dslExpr = compile(dslSource, isParallel=True)
        assert isinstance(dslExpr, ExpressionStack)

        # Remember the number of stubbed exprs - will check it after the value.
        actualLenStubbedExprs = dslExpr.stubbedExprs.qsize()

        # Create a mock valuation environment.
        # image = mock.Mock()
        # image.priceProcess.getDurationYears.return_value = 1
        # mock isn't pickleable :( so that doesn't work with multiprocessing
        kwds = {
            'image': MockImage(MockPriceProcess()),
            'interestRate': 0,
            'presentTime': datetime.datetime(2011, 1, 1),
            'allRvs': {
                '#1': dict([(datetime.datetime(2011, 1, 1) + datetime.timedelta(1) * i, numpy.array([10])) for i in range(0, 30)])

            },
        }


        # Evaluate the stack.
        dslValue = dslExpr.evaluate(isMultiprocessing=True, **kwds).mean()

        # Check the value is expected.
        self.assertEqual(dslValue, expectedValue)

        # Check the number of stubbed exprs is expected.
        self.assertEqual(actualLenStubbedExprs, expectedLenStubbedExprs)


class MockImage(object):

    def __init__(self, priceProcess):
        self.priceProcess = priceProcess


class MockPriceProcess(object):
    def getDurationYears(self):
        return 1
