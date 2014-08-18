Quant DSL
=========

*Quant DSL* is a domain specific functional programming language for quantitative analytics. *Quant DSL* supports consice declaration and distributed evaluation of stochastic models of derivative financial instruments. *Quant DSL* is written in *Python*, released with the BSD licence, and is [available to download](https://pypi.python.org/pypi/quantdsl) from the Python Package Index.

A paper defining the [syntax and semantics of *Quant DSL* expressions](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf) was published in 2011. An implementation was released as part of the *[Quant](https://pypi.python.org/pypi/quant)* package. More recently, in 2014, the language was expanded to support common elements of functional programming languages, to support the extensions envisaged in Section 6 of the 2011 paper ("*Future Development*"). Now, functions can define parameterised and recursive fragments of *Quant DSL*, and use them to generate a single primitive expression that can be very long, interpretted as a model of the computation, stored, assessed, evaluated under different conditions, reused, and so on. The original *Quant DSL* code has been improved and factored into an independent *Python* package, and relicensed with the BSD licence.

The *Quant DSL* continues to be a strict subset of the *Python* language syntax. As an illustative example of a *Quant DSL* module, please consider the following definition of an American option. You can see two user defined functions (*Option* and *American*), and an expression which states the specific terms of the option. The terms *Wait*, *Choice*, *TimeDelta*, and *Market* are built-in elements of *Quant DSL*.

```python
def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

def American(starts, ends, strike, underlying):
    if starts < ends:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )
    else:
        Option(starts, strike, underlying, 0)

American(Date('2016-04-01'), Date('2016-10-01'), 15, Market('NBP'))
```

Evaluation of *Quant DSL* expressions can be optimised so that computational redundancy is eliminated, and so that any branches can be executed in parallel. Parallel computation can be distributed across multiple processes on a single machine, or across multiple nodes on a network. A dependency graph for the computation can be constructed, and progressively worked through in an event driven manner, until the value of the expression is known, so that there is no need for long running processes. Intermediate values can be stored, so that there is no need to keep them in memory. Alternatively, the evaluation work can be completed entirely in memory using a single thread.

Here's a basic swing option.

```python
def Swing(starts, ends, underlying, quantity):
    if (quantity > 0) and (starts < ends):
        Wait(starts, Choice(
            Swing(starts + TimeDelta('1d'), ends, underlying,
                quantity - 1) + Fixing(starts, underlying),
        
            Swing(starts + TimeDelta('1d'), ends, underlying,
                quantity)
        ))
    else:
        0

Swing(Date('2016-04-01'), Date('2016-10-01'), Market('NBP'), 20)
```

Here's a simple gas storage model.

```python
def Storage(starts, ends, underlying, inventory, lowerlimit, upperlimit):
    if starts >= ends:
        0
    elif inventory < lowerlimit + 1:
        Wait(starts, Choice(
            Storage(starts + TimeDelta('1d'), ends, underlying,
                inventory),
                
            Storage(starts + TimeDelta('1d'), ends, underlying,
                inventory + 1) - Fixing(starts, underlying)
        )
    elif inventory > upperlimit - 1:
        Wait(starts, Choice(
            Storage(starts + TimeDelta('1d'), ends, underlying,
                inventory - 1) + Fixing(starts, underlying),
                
            Storage(starts + TimeDelta('1d'), ends, underlying,
                inventory)
        )
    else:
        Wait(starts, Choice(
            Storage(starts + TimeDelta('1d'), ends, underlying,
                inventory - 1) + Fixing(starts, underlying),
                
            Storage(starts + TimeDelta('1d'), ends, underlying,
                inventory + 1) - Fixing(starts, underlying)
        )

Storage(Date('2016-04-01'), Date('2017-04-01'), Market('NBP'), 200, 100, 5000)
```


Installation
------------

To install *Quant DSl*, install the `quantdsl` *Python* package.

```
pip install quantdsl
```

If you are operating behind a corporate firewall, then you may need to [download the distribution](https://pypi.python.org/pypi/quantdsl) and then use the path to the downloaded file instead of the package name.

```
pip install C:\Downloads\quantdsl-0.0.0.tar.gz
```

To avoid disturbing your system's site packages, it is recommended to install *Quant DSL* into a new virtual *Python* environment, using *[Virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/)*.

*Quant DSl* depends on *NumPy* and *SciPy*. On *Linux* systems these should be automatically installed as dependencies.

*Windows* users may not be able to install *NumPy* and *SciPy* because they do not have a compiler installed. If so, one solution would be to install the *[PythonXY](https://code.google.com/p/pythonxy/wiki/Downloads?tm=2)* distribution of *Python*, so that you have *NumPy* and *SciPy*, and then create a virtual environment with the `--system-site-packages` option of `virtualenv` so that *NumPy* and *SciPy* will be available in your virtual environment. (If you are using *PythonXY* v2.6, you will need to install virtualenv with the `easy_install` program that comes with *PythonXY*.) If you get bogged dow, the simpler alternative is to install *Quant DSL* directly into your *PythonXY* installation, using `pip install quantdsl` (or `easy_install quantdsl` if *Pip* is not available).


Getting Started
---------------

Using Python, get started by importing the `quantdsl` package.

```python
>>> import quantdsl
```

The convenience function `quantdsl.parse()` takes *Quant DSL* source code and returns a *Quant DSL* module object.

```python
>>> module = quantdsl.parse("10 + 20")
```

When converted to a string, a *Quant DSL* module (and all other *Quant DSL* objects) render themselves as equivalent source code.

```python
>>> print module
'10 + 20'
```

When a *Quant DSL* module is compiled, a *Quant DSL* expression is obtained.

```python
>>> expr = module.compile()
>>> print expr
'10 + 20'
```

The convenience function `quantdsl.compile()` takes *Quant DSL* source code and returns a *Quant DSL*  expression.

```python
>>> expr = quantdsl.compile("10 + 20")
```

A *Quant DSL* expression can be evaluated to a numerical value.

```python
>>> expr.evaluate()
30
```

The convenience function `quantdsl.eval()` takes *Quant DSL* source code and returns a numeric value.

```python
>>> quantdsl.eval("10 + 20")
30
```

Todo: Rewrite the rest of this doc!


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

