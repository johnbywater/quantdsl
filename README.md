Quant DSL
=========

***Domain specific language for quantitative analytics in finance.***

*Quant DSL* is a hybrid functional programming language for modelling derivative financial instruments. *Quant DSL* is written in Python, works with Python, looks like Python, and is available to [download from the Python Package Index](https://pypi.python.org/pypi/quantdsl).

Here is an example of a *Quant DSL* module that models an American option. There are two user defined functions (*Option* and *American*), and an expression which states the specific terms of the option. The terms *Wait*, *Choice* and *Market* are primitive elements of *Quant DSL* (*Date* and *TimeDelta* are constant value objects of *Quant DSL*, and so are the integers.)

```python
def American(starts, ends, strike, underlying):
    if starts < ends:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )
    else:
        Option(starts, strike, underlying, 0)

def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

American(Date('2016-04-01'), Date('2016-10-01'), 15, Market('NBP'))
```

*Quant DSL* source strings can be evaluated like this:

```python
import quantdsl

marketCalibration = {
    'NBP-LAST-PRICE': 10,
    'NBP-ACTUAL-HISTORICAL-VOLATILITY': 50,
}

quantdsl.eval(dslSource, marketCalibration=marketCalibration, interestRate=2.5, pathCount=500000)
```


Overview of the Language
------------------------

The core of *Quant DSL* is a set of primitive elements which encapsulate common elements of stochastic models, for example the least-squares Monte Carlo approach (coded as "*Choice*" in *Quant DSL*), Brownian motions ("*Market*"), and time value of money calculations ("*Wait*").

The primitive elements are supplemented with a set of binary operators (addition, subtraction, multiplication, etc.) and composed into probablistic expressions of value. The *Quant DSL* expressions are parsed into a *Quant DSL* object tree, which can be evaluated to generate an estimated value of the modelled contract terms. A paper defining the [syntax and semantics of *Quant DSL* expressions](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf) was published in 2011. (Proofs for the mathematical semantics are included in that paper.) An implementation of the 2011 *Quant DSL* expression language was released as part of the *[Quant](https://pypi.python.org/pypi/quant)* package.

More recently, in 2014, *Quant DSL* was expanded to involve common elements of functional programming languages, so that more extensive models could be expressed concisely. At this time, the original *Quant DSL* code was factored into a new Python package, and released with the BSD licence (this package).

*Quant DSL* expressions can now involve calls to user-defined functions. In turn, *Quant DSL* functions can define parameterized and conditional *Quant DSL* expressions - expressions which may be a function of call arguments, which involve further calls to user-defined functions, and which may be situated inside an 'if' clause. Because only primitive *Quant DSL* expressions can be evaluated directly, *Quant DSL* modules which contain an expression that depends on function definitions can now be compiled into a single primitive expression, so that the value of the model can be obtained.

Primitive *Quant DSL* expressions generated in this way can be much more extensive, relative to the short expressions it is possible to write by hand. Such compiled expressions constitute a step-wise object model of the computation, and can be constituted and persisted as a dependency graph ready for parallel and distributed execution. The compiled expressions can be evaluated under a variety of underlying conditions, with results from unaffected branches being reused (and not recalculated). The computational model can be used to measure and predict compuational load, form the basis for tracking progress through a long calculation, and it is possible to retry a stalled computation.

Evaluation of *Quant DSL* expressions can be optimised so that computational redundancy is eliminated, and so that any branches can be executed in parallel. Parallel computation can be distributed across multiple processes on a single machine, or across multiple nodes on a network. A dependency graph for the computation can be constructed, and progressively worked through in an event driven manner, until the value of the expression is known, so that there is no need for long running processes. Intermediate values can be stored, so that there is no need to keep them in memory. Alternatively, the evaluation work can be completed entirely in memory using a single thread.

The *Quant DSL* syntax continues to be a strict subset of the Python language syntax. There are various restrictions, which can lead to parse- and compile-time syntax errors. Here is a basic summary of the restrictions:
* a module is restricted to have any number of function definitions, and one expression only;
* there are no assignments, loops, comprehensions, or generators;
* the only valid names in a function body are the names of the call arguments, plus the names of the other functions, plus the built-in elements of the language;
* a function body and the sections of an 'if' clause can only have one statement;
* a statement is either an expression or an 'if' clause (binary and unary operators are supported);
* all 'if' clauses must end with en 'else' expression ('elif' is supported).
* the test compare expression of an 'if' clause cannot contain any of the primitive elements.

There are also some slight changes to the semantics of a function: in particular the return value of a function is  not the result of evaluting the expressions and returning a numeric value, but rather it is the result of selecting an expression by evaluating the test compare expression of 'if' statements and then compiling the selected expression into a primitive expression by making any function calls that are declared and substituting them with their return value.

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
        ))
    elif inventory > upperlimit - 1:
        Wait(starts, Choice(
            Storage(starts + TimeDelta('1d'), ends, underlying,
                inventory - 1) + Fixing(starts, underlying),
                
            Storage(starts + TimeDelta('1d'), ends, underlying,
                inventory)
        ))
    else:
        Wait(starts, Choice(
            Storage(starts + TimeDelta('1d'), ends, underlying,
                inventory - 1) + Fixing(starts, underlying),
                
            Storage(starts + TimeDelta('1d'), ends, underlying,
                inventory + 1) - Fixing(starts, underlying)
        ))

Storage(Date('2016-04-01'), Date('2017-04-01'), Market('NBP'), 200, 100, 5000)
```

Acknowledgments
---------------

*Quant DSL* was inspired by the paper *[Composing contracts: an adventure in financial engineering (functional pearl)](http://research.microsoft.com/en-us/um/people/simonpj/Papers/financial-contracts/contracts-icfp.ps.gz)* by Simon Peyton Jones and others. The idea of using a dependency graph, to help with parallel and distributed execution of the value process was inspired by a [talk about dependency graphs by Kirat Singh](https://www.youtube.com/watch?v=lTOP_shhVBQ). *Quant DSL* makes lots of use of SciPy and NumPy, and the Python AST.


Installation
------------

To install *Quant DSL*, install the `quantdsl` Python package.

```
pip install quantdsl
```

If you are operating behind a corporate firewall, then you may need to [download the distribution](https://pypi.python.org/pypi/quantdsl) and then use the path to the downloaded file instead of the package name.

```
pip install C:\Downloads\quantdsl-0.0.0.tar.gz
```

To avoid disturbing your system's site packages, it is recommended to install *Quant DSL* into a new virtual Python environment, using *[Virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/)*.

*Quant DSl* depends on *NumPy* and *SciPy*. On *Linux* systems these should be automatically installed as dependencies.

*Windows* users may not be able to install *NumPy* and *SciPy* because they do not have a compiler installed. If so, one solution would be to install the [PythonXY](https://code.google.com/p/pythonxy/wiki/Downloads?tm=2) distribution of Python, so that you have *NumPy* and *SciPy*, and then create a virtual environment with the `--system-site-packages` option of `virtualenv` so that *NumPy* and *SciPy* will be available in your virtual environment. (If you are using PythonXY v2.6, you will need to install virtualenv with the `easy_install` program that comes with PythonXY.) If you get bogged dow, the simpler alternative is to install *Quant DSL* directly into your PythonXY installation, using `pip install quantdsl` (or `easy_install quantdsl` if *Pip* is not available).


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

