Quant DSL
=========

INCOMPLETE, UNDER DEVELOPMENT  -- NB A new Python package quantdsl will be released in a few days, so if you happen to see this page and are interested, please come back in a few days :)

Quant DSL is a functional programming language, written in Python, that can generate and evaluate complex stochastic expressions of the value of derivative contracts.

A paper defining the syntax and semantics of Quant DSL expressions was published in 2011.

http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf

More recently, the language has been expanded to support common elements of a functional programming language, so that long and complex expressions can be represented concisely, as was envisaged in Section 6 of the 2011 paper ("Future Development").

Evaluation of an expression is optimised so that computational redundancy is eliminated and any branches can be executed in parallel. Computation can be distributed across multiple processes on a single machine, or across multiple nodes on a network. It can also be done in a single thread.

A dependency graph for the computation can be constructed, and progressively worked through in an event driven manner, so that there is no need for long running processes. Intermediate values can be stored, so that there is no need to keep everything in memory.

Large computations are possible with limited hardware, because the computation is firstly modelled as a network of partial expressions, the model is stored as a dependency graph, and then the model is progressively evaluated until the value of the original expression is known.

As illustative examples of a Quant DSL module, please consider the following definition of an American option.

```python
def Option(date, strike, underlying, alternative):
    return Wait(date, Choice(underlying - strike, alternative))

def American(starts, ends, strike, underlying):
    if starts >= ends:
        Option(starts, strike, underlying, 0)
    else:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )

American(Date('2016-04-01'), Date('2016-10-01'), 9, Market('TTF'))
```

Here's a Swing option.

```python
def Swing(starts, ends, underlying, quantity):
    if (quantity == 0) or (starts < ends):
        return 0
    else:
        return Choice(
            Swing(starts + TimeDelta('1d'), ends, underlying,
                quantity - 1) + Fixing(starts, underlying),
            Swing(starts + TimeDelta('1d'), ends, underlying,
                quantity)
        )

Swing(Date('2016-04-01'), Date('2016-10-01'), Market('NBP'), 2)
```

The rest of this article will try to explain what's going on. :)

Installation
------------

To install Quant DSl, simple install the `quantdsl` Python package.

To avoid disturbing your system's site packages, it is recommended to install Quant DSL into a new virtual Python environment, using `virtualenv`.

```
pip install quantdsl
```

Quant DSl depends on NumPy and SciPy. On Linux systems these can now be automatically installed, normally without any problems.

Windows users may not be able to install NumPy and SciPy because they do not have a compiler installed. If so, one solution would be to install PythonXY so that you have NumPy and SciPy, and then create a virtual environment with the `--system-site-packages` so that numpy and scipy will be available. If you are using PythonXY v2.6, you will need to install virtualenv with the `easy_install` program that comes with PythonXY. Pehaps the simpler alternative is to install Quant DSL directly into your PythonXY installation, using `easy_install quantdsl` (or `pip` if it is available) and forget about virtual Python environments - you could always reinstall PythonXY if something goes wrong.


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

