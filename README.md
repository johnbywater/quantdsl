Quant DSL
=========

Quant DSL is a Python implementation of the Quant DSL language.

Installation
------------

Install the `quantdsl` Python package.

```
pip install quantdsl
```

Introduction
------------

Using the Python language, get started by creating a Quant DSL Parser.

```python
>>> import quantdsl
>>> parser = quantdsl.Parser()
```

The parser's `parse()` method takes DSL source code statements and returns a DSL object.

```python
>>> expr = parser.parse("10 + 20")

>>> print type(expr)
<class 'quantdsl.Add'>

>>> isinstance(expr, quantdsl.DslObject)
True

>>> print expr
'10 + 20'

>>> expr.evaluate()
30
```

There are two kinds of statements: expressions and functions. Quant DSL source code can have zero to many functions, and zero or one expressions.

### Expressions

Expressions are evaluated to produce a resulting value.

Let's start with numbers and strings.

```python
>>> expr = parser.parse("10")

>>> print expr.evaluate()
10

>>> expr = parser.parse("-10")
>>> print expr.evaluate()
-10

>>> expr = parser.parse("-0.1")
>>> print expr.evaluate()
-0.1

>>> expr = parser.parse("'hello world'")
>>> print expr.evaluate()
'hello world'
```

Binary operations, such as addition, substraction, multiplication and division are also supported.

```python
>>> expr = parser.parse("10 + 4")
>>> print expr.evaluate()
14

>>> expr = parser.parse("10 - 4")
>>> print expr.evaluate()
6

>>> expr = parser.parse("10 * 4")
>>> print expr.evaluate()
40

>>> expr = parser.parse("10 / 4")
>>> print expr.evaluate()
2.5

```

The parser also supports dates and time deltas. Time deltas can be multiplied by numbers and added to, or subtracted from, dates.

```python
>>> expr = parser.parse("Date('2014-1-1')")
>>> print expr.evaluate()
datetime.datetime(2014, 1, 1, 0, 0)

>>> expr = parser.parse("TimeDelta('1d')")
>>> print expr.evaluate()
datetime.timedelta(1)

>>> expr = parser.parse("Date('2014-1-1') + 10 * TimeDelta('1d')")
>>> print expr.evaluate()
datetime.datetime(2014, 1, 11, 0, 0)
```

Variables can be used in expressions. Variables must be defined before the expression is evaluated.

```python
>>> expr = parser.parse("a + 4")
>>> print expr.evaluate(a=10)
14

>>> expr = parser.parse("a + b")
>>> print expr.evaluate(a=10, b=5)
15

>>> expr = parser.parse("TimeDelta('1d') * a")
>>> print expr.evaluate(a=10)
datetime.timedelta(10)
```

Numbers can be compared with numbers, and dates can be compared with dates. Numbers cannot be compared with dates.

```python
>>> expr = parser.parse("10 > 4")
>>> print expr.evaluate()
True

>>> expr = parser.parse("Date('2011-01-01') + a * TimeDelta('1d') < Date('2011-01-03')")
>>> print expr.evaluate(a=1)
True
>>> print expr.evaluate(a=3)
False
```


### Function Expressions

Expressions can involve user defined functions. Functions return a DSL expression.

The functions must be defined in the source code passed to the parser.

```python
>>> source = """
... def sqr(x):
...    x * x
... sqr(x)
... """
>>> expr = parser.parse(source)
>>> print expr.evaluate(x=10)
100
```

Functions can have a conditional expression, but each leg of the conditional can only have one expression.

```python
>>> source = """
... def f(x): 
...     if x < 0:
...         0
...     elif x < 1:
...         x
...     else:
...         x ** 2
... f(5)
... """
>>> expr = parser.parse(source)
>>> print expr.evaluate()
25
```

Functions are reentrant and can recurse.

```python
>>> source = """
... def fib(n): return fib(n-1) + fib(n-2) if n > 2 else n
... fib(5)
... """
>>> expr = parser.parse(source)
>>> print expr.evaluate()
8
```

Rather than computing values, a function actually returns a DSL expression which is subsituted in the calling expression in place of the function call. Such expressions are reduced by replacing references to function parameters with the call argument values. A function with a conditional expression has the test expression evaluated, and accordingly one leg of the conditional expression is selected, reduced with the call argument values, and returned.

```python
>>> source = """
... def f(x): x + 2
... f(2)
... """
>>> expr = parser.parse(source)
>>> print expr
2 + 2
```

If the selected expression calls a function, it is similarly substituted. And so on, until a DSL expression is obtained which does not involve any calls to used defined functions.


### Stochastic Expressions

To support stochastic calculus, Quant DSL has various pre-defined built-in expressions. `Market` is an expression of value that refers to a price at the given time, possibly a simulated future price. `Fixing` binds an expression of value to a fixed date, making its expression a function of that time,regardless of the given time. `Settlement` discounts an expression of value from a fixed date to the given time. `Wait` effectively combines settlement and fixing, so that an expression is both fixed at a particular time, and also discounted back to the given time. `Choice` implements the least-squares monte-carlo approach suggested by Longstaff-Schwartz.

Examples of function definitions that can be created using the built in expressions follow.

```python
"""
def Swing(starts, ends, underlying, quantity):
    if (quantity == 0) or (starts >= ends):
        0
    else:
        Choice(
            Swing(starts + TimeDelta('1d'), ends, underlying, quantity - 1) + Fixing(starts, underlying),
            Swing(starts + TimeDelta('1d'), ends, underlying, quantity)
        )
Swing(Date('2012-01-01'), Date('2012-02-01'), Market('NBP'), 500)
"""
```

Todo: Parse and pretty print the reduced monolithic DSL expression.

Todo: Parse and pretty print the reduced stubbed DSL expression stack.

Todo: More about the executing stubbed DSL expression stack in parallel using multiprocessing library.

Todo: More about the executing stubbed DSL expression stack in parallel using Redis.

Todo: More about the executing stubbed DSL expression stack in parallel using Celery.

