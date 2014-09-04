Quant DSL
=========

***Domain specific language for quantitative analytics in finance and trading.***

[![Build Status](https://secure.travis-ci.org/johnbywater/quantdsl.png)](https://travis-ci.org/johnbywater/quantdsl)

*Quant DSL* is a functional programming language for modelling derivative instruments.

The reason for having a domain specific language for quantitative analytics is to avoid implementing each new contract individually. By defining elements which can be combined into expressions, it becomes possible to describe and value new contracts quickly without writing new software.

At the heart of *Quant DSL* is a set of built-in elements (e.g. *"Market"*, *"Choice"*, *"Wait"*) that encapsulate maths used in finance and trading (i.e. models of market dynamics, the least-squares Monte Carlo approach, time value of money calculations) and which can be composed into executable expressions of value.

User defined functions are supported, and can be used to generate massive expressions. The syntax of *Quant DSL* expressions has been formally defined, and the semantic model is supported with [mathematical proofs](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf). The Python package `quantdsl` is an implementation in Python of the *Quant DSL* syntax and semantics.

Stable releases of `quantdsl` are available to [download from the Python Package Index](https://pypi.python.org/pypi/quantdsl). `quantdsl` has been tested with Python 2.7 on GNU/Linux (Ubuntu 14.04) and on Windows 7 (using PythonXY v2.7.6.1). You may wish to [contribute improvements on GitHub](https://github.com/johnbywater/quantdsl).


Introduction
------------

Here is an American option expressed in *Quant DSL*. There are two user defined functions (*"American"* and *"Option"*), and an expression which declares that the owner of the option may at any time during April 2015 buy one unit of "NBP" at a strike price of 9 units of currency. (The terms *Wait*, *Choice*, *Market*, *Date*, *TimeDelta* and *nostub* are elements of *Quant DSL*.)


```python
def American(starts, ends, strike, underlying):
    if starts < ends:
        Option(starts, strike, underlying,
            American(starts + TimeDelta('1d'), ends, strike, underlying)
        )
    else:
        Option(starts, strike, underlying, 0)

@nostub
def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

American(Date('2015-04-01'), Date('2015-05-01'), 9, Market('NBP'))
```

A command line interface program called `quant-dsl.py` is provided by the `quantdsl` Python package, so that *Quant DSL* source code can be easily evaluated.

```
$ quant-dsl.py -h
usage: quant-dsl.py [-h] [-c CALIBRATION] [-n NUM_PATHS] [-p PRICE_PROCESS]
                    [-i INTEREST_RATE] [-m [MULTIPROCESSING_POOL]] [-q] [-s]
                    SOURCE

Evaluates 'Quant DSL' code in SOURCE, given price process parameters in CALIBRATION.

positional arguments:
  SOURCE                DSL source URL or file path ("-" to read from STDIN)

optional arguments:
  -h, --help            show this help message and exit
  -c CALIBRATION, --calibration CALIBRATION
                        market calibration URL or file path (default: None)
  -n NUM_PATHS, --num-paths NUM_PATHS
                        number of paths in price simulations (default: 50000)
  -p PRICE_PROCESS, --price-process PRICE_PROCESS
                        price process model of market dynamics (default:
                        quantdsl:BlackScholesPriceProcess)
  -i INTEREST_RATE, --interest-rate INTEREST_RATE
                        annual percent interest rate (default: 2.5)
  -m [MULTIPROCESSING_POOL], --multiprocessing-pool [MULTIPROCESSING_POOL]
                        evaluate with multiprocessing pool (option value is
                        pool size, which defaults to cpu count) (default: 0)
  -q, --quiet           don't show progress info (default: False)
  -s, --show-source     show source code and compiled expression stack
```

Market calibration parameters are required to evaluate an expression which involves an underlying *Market*. For such expressions, `quantdsl` provides a multi-market Black-Scholes price process (a one factor "spot/vol" model of market dynamics). It can simulate a correlated evolution of future prices for a number of different markets. `quantdsl` can use other price processes, however no other price processes have (so far) been developed for this package.

The Black-Scholes price process provided by this package needs one-factor "spot/vol" market calibration parameters, with a correlation parameter for each pair of markets.

For two correlated markets called 'NBP' and 'TTF', under the default Black-Scholes price process, the market calibration parameters may look like this:

```python
{
   "NBP-LAST-PRICE": 10,
   "NBP-ACTUAL-HISTORICAL-VOLATILITY": 50,
   "TTF-LAST-PRICE": 11,
   "TTF-ACTUAL-HISTORICAL-VOLATILITY": 40,
   "NBP-TTF-CORRELATION": 0.4
}
```

The default Black-Scholes price process can be replaced with your own model of market dynamics, by using the `--price-process` option of the `quant-dsl.py` command line program. The market calibration parameters are used only by the price process (which uses them to simulate an evolution of future prices that *Quant DSL* "*Market*" objects will consume - so if you have developed a model of market dynamics, the market calibration parameters will contain whatever is needed by your price process object).

With the above example *Quant DSL* American option saved in a file called `americanoption.quantdsl` (the filename is arbitrary, and could be replaced with a URL), and with the above market calibration parameters in a file called `calibration.json` (again, the filename is arbitrary, and could be a URL), the following shell command evaluates that American option with those market calibration parameters under the default one-factor model of market dynamics.

```
$ quant-dsl.py -c calibration.json  americanoption.quantdsl 
DSL source from: americanoption.quantdsl

Calibration from: calibration.json

Compiling DSL source, please wait...

Duration of compilation: 0:00:00.500619

Compiled DSL source into 368 partial expressions (root ID: dfca11e1-e131-4588-945b-c8c924fb53e6).

Price process class: quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess

Path count: 50000

Finding all Market names and Fixing dates...

Simulating future prices for Market 'NBP' from observation time '2014-09-03' to '2015-04-01', '2015-04-02', '2015-04-03', '2015-04-04', '2015-04-05', '2015-04-06', '2015-04-07', '2015-04-08', [...], '2016-04-01'.

Evaluating 368 expressions (1 leaf) with a single thread, please wait...

Duration of evaluation: 0:00:30.817102    (11.94 expr/s)

Result: {
    "mean": 2.3512557789843007, 
    "stderr": 0.014112068993018262
}
```


Installation
------------

Use Pip to install the *Quant DSL* Python package.

```
pip install quantdsl
```

If you are working on a corporate LAN, with an HTTP proxy that requires authentication, then pip may fail to find the Python Package Index. In this case you may need to [download the distribution](https://pypi.python.org/pypi/quantdsl) (and dependencies) by hand, and then use the path to the downloaded files instead of the package name in the `pip` command:

```
pip install C:\MyDownloads\quantdsl-X.X.X.tar.gz
```

To avoid disturbing your system's site packages, it is recommended to install *Quant DSL* into a new virtual Python environment, using [Virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

*Quant DSL* depends on NumPy and SciPy. On Linux systems these should be automatically installed as dependencies. If you don't want to build the binaries, you can just install the system packages and create a virtual environment that uses system site packages (`--system-site-packages`).

(Windows users may not be able to install NumPy and SciPy because they do not have a compiler installed. If so, one solution would be to install the [PythonXY](https://code.google.com/p/pythonxy/wiki/Downloads?tm=2) distribution of Python, so that you have NumPy and SciPy, and then create a virtual environment with the `--system-site-packages` option of `virtualenv` so that NumPy and SciPy will be available in your virtual environment. (If you are using PythonXY v2.6, you will need to install virtualenv with the `easy_install` program that comes with PythonXY.) If you get bogged down, the simpler alternative is to install *Quant DSL* directly into your PythonXY installation, using `pip install quantdsl` or `easy_install quantdsl` if `pip` is not available.)


Overview of the Language
------------------------

The core of *Quant DSL* is a set of primitive elements which encapsulate common elements of stochastic models, for example price process evolution (coded as "*Market*" in *Quant DSL*), the least-squares Monte Carlo approach ("*Choice*"), and time value of money calculations ("*Wait*").

The primitive elements are supplemented with a set of binary operators (addition, subtraction, multiplication, etc.) and composed into expressions of value. The *Quant DSL* expressions are parsed into an object graph, which is evaluated to generate an estimated value of the modelled contract. A paper defining the [syntax and semantics of *Quant DSL* expressions](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf) was published in 2011. (Proofs for the mathematical semantics are included in that paper.) An implementation of the 2011 *Quant DSL* expression language was released as part of the *[Quant](https://pypi.python.org/pypi/quant)* package.

More recently, in 2014, *Quant DSL* was expanded to involve common elements of functional programming languages, so that more extensive models could be expressed concisely. At this time, the original *Quant DSL* code was factored into a new Python package, and released with the BSD licence (this package). *Quant DSL* expressions can now involve calls to user-defined functions. In turn, *Quant DSL* functions can define parameterized and conditional *Quant DSL* expressions. *Quant DSL* modules which contain an expression that depends on function definitions must be compiled into a single primitive expression (or dependency graph) before they can be evaluated.

Primitive *Quant DSL* expressions generated in this way can be much more extensive, relative to the short expressions it is possible to write by hand. Such compiled expressions form an object model of the computation, and can be constructed and stored as a dependency graph ready for parallel and distributed execution. The compiled expressions can be evaluated under a variety of underlying conditions, with results from unaffected branches being reused (and not recalculated). The computational model can be used to measure and predict computational load, for tracking progress through a long calculation, and to retry a stalled computation.

Evaluation of *Quant DSL* expressions can be optimised so that computational redundancy is eliminated, and so that any branches can be executed in parallel. Parallel computation can be distributed across multiple processes on a single machine, or across multiple nodes on a network. A dependency graph for the computation can be constructed, and progressively worked through in an event driven manner, until the value of the expression is known, so that there is no need for long running processes. Intermediate values can be stored, so that there is no need to keep them in memory. Alternatively, the evaluation work can be completed entirely in memory using a single thread.

The *Quant DSL* syntax is a strict subset of the Python language syntax. There are various restrictions, which can lead to parse- and compile-time syntax errors. Here is a basic summary of the restrictions:
* a module is restricted to have any number of function definitions, and one expression;
* there are no assignments, loops, comprehensions, or generators;
* the only valid names in a function body are the names of the call arguments, plus the names of the other functions, plus the built-in elements of the language;
* a function body and the sections of an 'if' clause can only have one statement;
* a statement is either an expression or an 'if' clause (binary and unary operators are supported);
* all 'if' clauses must end with en 'else' expression ('elif' is also supported).
* the test compare expression of an 'if' clause cannot contain any of the primitive elements of *Quant DSL*.

There are also some slight changes to the semantics of a function: in particular the return value of a function is not the result of evaluating the expressions, but rather it is the result of selecting an expression by evaluating the test compare expression of 'if' statements, and then compiling the selected expression into a primitive expression by making any function calls that are declared and substituting them with their return value. That is, what a function returns is a *Quant DSL* expression, rather than the evaluation of such an expression.


Acknowledgments
---------------

*Quant DSL* was partly inspired by the paper *[Composing contracts: an adventure in financial engineering (functional pearl)](http://research.microsoft.com/en-us/um/people/simonpj/Papers/financial-contracts/contracts-icfp.htm)* by Simon Peyton Jones and others. The idea of orchestrating evaluations with a dependency graph, to help with parallel and distributed execution, was inspired by a [talk about dependency graphs by Kirat Singh](https://www.youtube.com/watch?v=lTOP_shhVBQ). *Quant DSL* makes lots of use of design patterns, SciPy and NumPy, and the Python AST.


Getting Started
---------------

The *Quant DSL* Python package is designed to be integrated into other software applications. This can be done by using the command line interface (see above), by writing a Python program which imports code from `quantdsl`, or as a service accessed via HTTP.

The *Quant DSL* package provides three convenience functions: `parse()`, `compile()`, and `eval()`.

Let's get started by opening an interactive Python session.

```bash
$ python
Python 2.7.5+ (default, Feb 27 2014, 19:37:08) 
[GCC 4.8.1] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

You can "import" the *Quant DSL* convenience functions from `quantdsl.services`.

```python
>>> from quantdsl.services import parse
```
The convenience function `parse()` takes a piece of *Quant DSL* source code and returns a *Quant DSL* Module object.

```python
>>> parse("10 + 20")
<quantdsl.semantics.Module object at 0x7fadb5f23ad0>
```

When converted to a string, a *Quant DSL* Module (and all other *Quant DSL* objects) renders itself as equivalent *Quant DSL* source code.

```python
>>> print parse("10 + 20")
10 + 20
```

When a *Quant DSL* module is compiled, a *Quant DSL* expression is obtained. In the case of `10 + 20`, an Add expression is obtained.

```python
>>> parse("10 + 20").compile()
<quantdsl.semantics.Add object at 0x7fadb8888510>
```

The convenience function `compile()` takes *Quant DSL* source code and directly returns a *Quant DSL* expression.

```python
>>> from quantdsl.services import compile
>>> compile("10 + 20")
<quantdsl.semantics.Add object at 0x7fadb8888510>
>>> print compile("10 + 20")
10 + 20
```

A *Quant DSL* expression can be evaluated to a numeric value.

```python
>>> compile("10 + 20").evaluate()
30
```

The convenience function `eval()` takes *Quant DSL* source code and directly returns a numeric result.

```python
>>> from quantdsl.services import eval
>>> eval("10 + 20")
30
```


### Scalar Numbers

As we have seen, *Quant DSL* expressions can be evaluated to produce a numeric value. Here is a negative floating point number.

```python
>>> eval("-1.2345")
-1.2345
```

### Binary Operators

Various binary operations are supported, such as addition, substraction, multiplication and division.

```python
>>> eval("10 + 4")
14

>>> eval("10 - 4")
6

>>> eval("10 * 4")
40

>>> eval("10 / 4")
2.5
```

Python precedence rules are followed. Parentheses can be used to change the order.

```python
>>> print parse("10 * 4 + 11 * 5")
(10 * 4) + (11 * 5)

>>> print parse("10 * (4 + 11) * 5")
(10 * (4 + 11)) * 5
```

### Dates and Time Deltas

The parser supports dates (`Date`) and time deltas (`TimeDelta`).

```python
>>> eval("Date('2014-1-1')")
datetime.datetime(2014, 1, 1, 0, 0, tzinfo=<UTC>)

>>> eval("TimeDelta('1d')")
datetime.timedelta(1)
```

Time deltas can be multiplied by integer numbers. They can be added to (and subtracted from) dates and other time deltas.

```python
>>> eval("10 * TimeDelta('1d')")
datetime.timedelta(10)

>>> eval("Date('2014-1-1') + TimeDelta('1d')")
datetime.datetime(2014, 1, 2, 0, 0, tzinfo=<UTC>)

>>> eval("Date('2014-1-1') + 10 * TimeDelta('1d')")
datetime.datetime(2014, 1, 11, 0, 0, tzinfo=<UTC>)
```

### Random Variables

To support stochastic calculus, *Quant DSL* has built-in primitive elements that work with random variables.

A *Market* is an expression of value that refers by name to a simulated future price, fixed at a given "present time".

A  *Market* evaluates to a number of samples from a continuous random variable, and a summary of the random variable (the mean and standard error of the random variable as a normal distribution) is returned by the `eval` convenience function.

The simulation of future prices involves a number of "paths" - one for each sample from the random variable.
Prices are evolved from the "observation time" (the "present time" of the continaing *Quant DSL* module), to the "present time" of the *Market* object (by adding random increments to each path according to the length of the duration - see the price process object class for details).

The "present time" is given to a *Market* when it is evaluated, so that the value of a *Market* can be conditioned by surrounding *Quant DSL* elements (e.g. by fixing the market to an agreed date in the future).

As expected, the simulated value of an expression made simply of a single market object (e.g. the expression `"Market('NBP')"`) is exactly the corresponding spot price in the market calibration parameters. The standard error of zero because the duration from observation time to the effective present time is zero, so there are no random increments.

```python
>>> import datetime
>>> from quantdsl import utc
>>> observationTime = datetime.datetime(2011,1,1, tzinfo=utc)
>>> marketCalibration = {'TTF-LAST-PRICE': 11, 'TTF-ACTUAL-HISTORICAL-VOLATILITY': 40}
>>> eval("Market('TTF')", observationTime=observationTime, marketCalibration=marketCalibration)
{'stderr': 0.0, 'mean': 11.0}

>>> marketCalibration = {'NBP-LAST-PRICE': 10, 'NBP-ACTUAL-HISTORICAL-VOLATILITY': 50}
>>> eval("Market('NBP')", observationTime=observationTime, marketCalibration=marketCalibration)
{'stderr': 0.0, 'mean': 10.0}
```

In *Quant DSL*, a *"Fixing"* is an expression that contains a date and another expression. A *Fixing* will bind its expression to its date, so that when the contained expression is evaluated, the present time of the contained expression is set to the date of the *Fixing*. For example, a *Fixing* can set the future date of a simulated *Market* price to be different from the observed time of the module evaluation. In this case, the standard error of the result is non-zero.

The mean of the random variable has moved from the last price in the calibration, but the move is comparable to the stderr.

```python
>>> eval("Fixing('2014-01-01', Market('NBP'))", observationTime=observationTime, marketCalibration=marketCalibration)
{'stderr': 0.074021663992697703, 'mean': 10.008802421966257}
```

The standard error of the result increases as the duration from observation time to the effective present time of the *Market* becomes longer.

```python
>>> eval("Fixing('2024-01-01', Market('NBP'))", observationTime=observationTime, marketCalibration=marketCalibration)
{'stderr': 0.35422549196595243, 'mean': 10.268412057325369}
```

The standard error can be reduced by increasing the number of paths in the simulation (default path count: 20000). (Memory usage will increase, and computation steps will take longer, but since `quantdsl` discards intermediate results, once the price simulations have been generated, the memory profile is flat even for very large computations.)

```python
>>> eval("Fixing('2024-01-01', Market('NBP'))", observationTime=observationTime, marketCalibration=marketCalibration, pathCount=200000)
{'stderr': 0.1060611734866011, 'mean': 10.067874599156543}
```

`Settlement` discounts an expression of value from a fixed date to the given time.

Todo: More about Settlement.

`Wait` effectively combines settlement and fixing, so that an expression is both fixed at a particular time, and also discounted back to the given time.

Todo: More about Wait.

`Choice` implements the least-squares monte-carlo approach suggested by Longstaff-Schwartz.

Todo: More about Choice.


### Variables

Variables, such as those defined by function parameters - see below, can be used in expressions. In general, variables must be defined before the expression is compiled.

```python
>>> eval("a", compileKwds={'a': 2})
2
```

Variables can be combined with numbers and dates.

```python
>>> eval("a + 4", compileKwds={'a': 2})
6
>>> eval("TimeDelta('1d') * a"))
datetime.timedelta(10)
```


### Function Definitions

Expressions can involve calls to user defined functions. Functions return a *Quant DSL* expression, rather than a numeric result.

```python
>>> source = """
... def sqr(x):
...    x * x
... """
>>> print compile(source).apply(x=10)
10 * 10
```

Functions can have a conditional expression, but each leg of the conditional can only have one expression. When the function is called, the test compare expression is evaluated. According to the result of the comparison, one of the expressions is compiled in the context of the function call arguments, and returned as the result of the function call. (It follows that the stocastic elements cannot be used in test compare expressions.)


```python
>>> source = """
... def f(x): 
...     if x <= 0:
...         0.0
...     elif x < 1:
...         x
...     else:
...         1.0 + x
... """
>>> func = compile(source)
>>> print func.apply(x=-1)
0.0
>>> print func.apply(x=0.5)
0.5
>>> print func.apply(x=5)
1.0 + 5.0
```

All the usual compare operators are supported (`==`, `!=`, `<`, `>`, etc.). The compare operators can be used with numerical expressions, and also expressions involving dates and time deltas. Numbers can be compared with numbers, and dates can be compared with dates. (Numbers cannot be compared with dates or datetimes, and datetimes cannot be compared with dates.)

```python
>>> eval("10 > 4")
True

>>> eval("Date('2011-01-01') < Date('2011-01-03')")
True
```

Comparisons can involve variables, and expressions that combine with numbers and dates.

```python
>>> source = "Date('2011-01-01') + a * TimeDelta('1d') < Date('2011-01-03')"
>>> eval(source, compileKwds={'a': 1})
True

>>> eval(source, compileKwds={'a': 3})
False
```

Functions are reentrant and can recurse.

```python
>>> source = """
... def fib(n): return fib(n-1) + fib(n-2) if n > 2 else n
... """
>>> print compile(source).apply(n=5)
((2 + 1) + 2) + (2 + 1)
```

Todo: Parse and pretty print the reduced monolithic DSL expression.

Todo: Parse and pretty print the reduced stubbed DSL expression stack.

Todo: More about the executing stubbed DSL expression stack in parallel using multiprocessing library.

Todo: More about the executing stubbed DSL expression stack in parallel using Redis.

Todo: More about the executing stubbed DSL expression stack in parallel using Celery.

