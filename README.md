# Quant DSL

***Domain specific language for quantitative analytics in finance and trading.***

[![Build Status](https://secure.travis-ci.org/johnbywater/quantdsl.png)](https://travis-ci.org/johnbywater/quantdsl)


## Install

Use pip to install the [latest distribution](https://pypi.python.org/pypi/quantdsl) from
the Python Package Index. To avoid disturbing your system's site packages, it is recommended to install
into a new virtual Python environment, using [Virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

```
pip install --upgrade pip
pip install quantdsl
```

Please register any [issues on GitHub](https://github.com/johnbywater/quantdsl/issues).


## Overview

Quant DSL is a functional programming language for modelling derivative instruments.

At the heart of Quant DSL is a set of elements - e.g. *Settlement*, *Fixing*, *Choice*, *Market* - which encapsulate 
maths used in finance and trading. The elements of the language can be freely composed into expressions
of value. User defined functions generate extensive dependency graphs that effectively model and evaluate exotic
derivatives.

The syntax of Quant DSL expressions has been
[formally defined](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf),
the semantic model is supported with mathematical proofs.

This package is an implementation of the Quant DSL in Python. 

## Implementation

In addition to the Quant DSL expressions, to ease construction of Quant DSL expressions, function definition
statements `def` are supported. And the `import` statement is supported, to allow function definitions to be used
from a library (see below).

Steps for evaluating a contract include: specification of a model of a contract; calibration of a stochastic process
for the underlying prices; simulation using the price process of future prices underlying the contract; and evaluation
of the contract model against the simulation.

The library provides an application class `QuantDslApplication` which has methods that support these steps:
`compile()`, `simulate()` and `evaluate()`. During compilation of the specification source code, the application 
constructs a dependency graph of function calls. The simulation is generated according to requirements derived 
from the depedency graph, and a calibrated price process. During evaluation, nodes are evaluated when they are ready
 to be evaluated, and intermediate call results are discarded as soon as they are no longer required, which means 
 memory usage is mostly constant during evaluation. For the delta calculations, nodes are selectively 
 re-evaluated with perturbed values, according to the periods and markets they involve.

The examples below use the library function `calc()` to evaluate Quant DSL source code. `calc()` uses the methods 
of the `QuantDslApplication` described above.

```python
from quantdsl.interfaces.calcandplot import calc
```

## Introduction

Simple calculations.

```python
results = calc("2 + 3 * 4 - 10 / 5")

assert results.fair_value == 12, results.fair_value
```

Other binary operations.

```python
results = calc("Max(9 // 2, Min(2**2, 12 % 7))")

assert results.fair_value == 4, results.fair_value
```

Boolean operations.

```python
assert calc("1 and 2").fair_value == True
assert calc("0 and 2").fair_value == False
assert calc("2 and 0").fair_value == False
assert calc("0 and 0").fair_value == False

assert calc("1 or 1").fair_value == True
assert calc("1 or 0").fair_value == True
assert calc("0 or 1").fair_value == True
assert calc("0 or 0").fair_value == False
```

Date and time values and operations.

```python
import datetime


results = calc("Date('2011-1-1')")
assert results.fair_value == datetime.datetime(2011, 1, 1), results.fair_value

results = calc("Date('2011-1-10') - Date('2011-1-1')")
assert results.fair_value == datetime.timedelta(days=9), results.fair_value

results = calc("Date('2011-1-1') + 5 * TimeDelta('1d') < Date('2011-1-10')")
assert results.fair_value == True, results.fair_value

results = calc("Date('2011-1-1') + 10 * TimeDelta('1d') < Date('2011-1-10')")
assert results.fair_value == False, results.fair_value
```

### Function definitions

Function definitions can be used to structure complex expressions.
 
When evaluating an expression that involves calls to function definitions, the call to the function definition is 
firstly replaced with the expression 
ed by the function definition, so that a larger expression is formed.

The call args of the function definition can be used as names in the function definition's expressions. The call arg 
values will be used to evaluate the expression returned by the function.

```python
results = calc("""
def Contract1(a):
    a * Contract2() + 1000 * Contract3(a)


def Contract2():
    25


def Contract3(a):
    a * 1.1


Contract1(10)
""")

assert results.fair_value == 11250, results.fair_value
```   

The call args of the function definition can be used in an if-else block, so that different expressions can be 
returned depending upon the function call argument values.

Please note, the expressions preceding the colon in the if-else block must be simple expressions involving the call 
args and must not involve any Quant DSL stochastic elements, such as `Market`, `Choice`, `Wait`, `Settlement`,
`Fixing`. Calls to function definitions from these expressions are currently not supported.

Each call to a (non-inlined) function definition becomes a node on a dependency graph. Each call is internally  
memoised, so it is only called once with the same argument values, and the result of such a call is reused.



```python
results = calc("""
def Fib(n):
    if n > 1:
        Fib(n-1) + Fib(n-2)
    else:
        n

Fib(60)
""")

assert results.fair_value == 1548008755920, results.fair_value
```   

### Market

Underlying prices are expressed with the `Market` element. A price process is used to simulate future 
prices.

```python

results = calc("Market('GAS')",
    observation_date='2011-1-1',
    price_process={
        'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
        'market': ['GAS'],
        'sigma': [0.0],
        'curve': {
            'GAS': [
                ('2011-1-1', 10)
            ]
        },
    }
)

assert results.fair_value.mean() == 10, results.fair_value
```

### Fixing

The `Fixing` element is used to condition the effective present time of included expressions. In the example below, 
the expression evaluates to the 'GAS' market price on '2112-1-1'.

The forward curve is used to estimate future prices, with zero-order hold from the last known value.

```python

results = calc("Fixing('2112-1-1', Market('GAS'))",
    observation_date='2011-1-1',
    price_process={
        'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
        'market': ['GAS'],
        'sigma': [0.0],
        'curve': {
            'GAS': [
                ('2011-1-1', 10),
                ('2111-1-1', 1000)
            ]
        },
    },
    interest_rate=2.5,
)

assert results.fair_value.mean() == 1000, results.fair_value.mean()
```   

With geometric brownian motion, there may be future price movements.

```python

results = calc("Fixing('2112-1-1', Max(1000, Market('GAS')))",
    observation_date='2011-1-1',
    price_process={
        'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
        'market': ['GAS'],
        'sigma': [0.2],
        'curve': {
            'GAS': [
                ('2011-1-1', 10),
                ('2111-1-1', 1000)
            ]
        },
    },
    interest_rate=2.5,
)

assert results.fair_value.mean() > 1000, results.fair_value.mean()
```   

### Settlement

Discounting to net present value with `Settlement`. A hundred years at 2.5% gives heavy discounting from 10 to less 
than 1.

```python
results = calc("Settlement('2111-1-1', 10)",
    observation_date='2011-1-1',
    interest_rate=2.5,
)

assert results.fair_value < 1, results.fair_value
```

If the effective present time is the same as the settlement date, there is no discounting.

```python
results = calc("Settlement('2111-1-1', 10)",
    observation_date='2111-1-1',
    interest_rate=2.5,
)
assert results.fair_value == 10, results.fair_value

results = calc("Fixing('2111-1-1', Settlement('2111-1-1', 10))",
    observation_date='2011-1-1',
    interest_rate=2.5,
)
assert results.fair_value == 10, results.fair_value
```

### Wait

The `Wait` element combines `Settlement` and `Fixing`, so that a single date value is used both to condition the 
effective present time of the included expression, and also the value of that expression is discounted to the 
effective present time of the including expression.

```python

results = calc("Wait('2112-1-1', Market('GAS'))",
    observation_date='2011-1-1',
    price_process={
        'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
        'market': ['GAS'],
        'sigma': [0.2],
        'curve': {
            'GAS': [
                ('2011-1-1', 10),
                ('2111-1-1', 1000)
            ]
        },
    },
    interest_rate=2.5,
)

assert results.fair_value.mean() < 100, results.fair_value.mean()
```   

### Choice

The `Choice` element uses the least-squares Monte Carlo approach proposed by Longstaff and 
Schwartz (1998) to compare the conditional expected value of each alternative.

```python
results = calc("""
Choice(
    Wait('2012-1-1', Market('GAS')),
    Wait('2112-1-1', Market('GAS')),
)
""",
    observation_date='2011-1-1',
    price_process={
        'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
        'market': ['GAS'],
        'sigma': [0.2],
        'curve': {
            'GAS': [
                ('2011-1-1', 10),
                ('2111-1-1', 1000)
            ]
        },
    },
    interest_rate=2.5,
)

assert 70 < results.fair_value.mean() < 100, results.fair_value.mean()
```   

### European and american options

In general, an option can be expressed as a "wait" until a date for a "choice" between, on one hand, the 
difference between the price of an underlying and a strike price, and, on the other hand, an alternative expression.

```python

def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

```

A european option can then be expressed as an option to buy an underlying commodity at a given strike price 
on a given date, the alternative being to do nothing.

```python
def European(date, strike, underlying):
    Option(date, strike, underlying, 0)
```

Similarly, an american option is an option to exercise at a given strike price on the start date, with an alternative being
 an american option starting the next date (after a `step` in time, such as one day), and so on until the end date.


```python
def American(start, end, strike, underlying, step):
    if start <= end:
        Option(start, strike, underlying,
            American(start + step, end, strike, underlying, step)
        )
    else:
        0
```

If the strike price is the same as the underlying, without any volatility (`sigma`) there is no value holding an 
option.

```python
results = calc("""from quantdsl.lib.american1 import American

American(Date('2011-1-1'), Date('2011-1-10'), 10, Market('GAS'), TimeDelta('1d'))
""",
    observation_date='2011-1-1',
    price_process={
        'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
        'market': ['GAS'],
        'sigma': [0.0],
        'curve': {
            'GAS': [
                ('2011-1-1', 10),
            ]
        },
    },
    interest_rate=2.5,
)
assert results.fair_value.mean() == 0, results.fair_value.mean()
```

If the strike price is the same as the underlying, with some volatility in the price of the underlying (`sigma`) there
 is some value in the option.

```python
results = calc("""from quantdsl.lib.american1 import American

American(Date('2012-1-1'), Date('2012-1-10'), 10, Market('GAS'), TimeDelta('1d'))
""",
    observation_date='2011-1-1',
    price_process={
        'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
        'market': ['GAS'],
        'sigma': [0.9],
        'curve': {
            'GAS': [
                ('2011-1-1', 10),
            ]
        },
    },
    interest_rate=2.5,
)
assert results.fair_value.mean() > 3, results.fair_value.mean()
```


## Examples of usage

The examples below use the library function `calc_print_plot()` to evaluate contracts, and print and plot results.

```python
from quantdsl.interfaces.calcandplot import calc_print_plot
```

If you run these examples, the deltas for each market in each period will be calculated, and estimated risk neutral 
 hedge positions will be printed for each market in each period, along with the overall fair value. A plot will be 
 displayed showing underlying prices and the cumulative hedge positions for each underlying, and the net cash from the 
 hedge positions (profit and loss).

The plot will also show the statistical distribution of the simulated prices, and the statistical error of the hedge 
 positions and the cash flow. Comparing the resulting net cash position with the fair value gives an indication of 
 how well the deltas are performing.


### Gas storage

An evaluation of a gas storage facility. This example uses a forward curve that reflects seasonal variations across 
the term of the contract. 

```python
results = calc_print_plot(
    title="Gas Storage",
    
    source_code="""
def GasStorage(start, end, commodity_name, quantity, target, limit, step):
    if ((start < end) and (limit > 0)):
        if quantity <= 0:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, target, limit, step),
                Inject(start, end, commodity_name, quantity, target, limit, step, 1),
            ))
        elif quantity >= limit:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, target, limit, step),
                Inject(start, end, commodity_name, quantity, target, limit, step, -1),
            ))
        else:
            Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, target, limit, step),
                Inject(start, end, commodity_name, quantity, target, limit, step, 1),
                Inject(start, end, commodity_name, quantity, target, limit, step, -1),
            ))
    else:
        if target < 0 or target == quantity:
            0
        else:
            BreachOfContract()


@inline
def Continue(start, end, commodity_name, quantity, target, limit, step):
    GasStorage(start + step, end, commodity_name, quantity, target, limit, step)


@inline
def Inject(start, end, commodity_name, quantity, target, limit, step, vol):
    Continue(start, end, commodity_name, quantity + vol, target, limit, step) - \
    vol * Market(commodity_name)


@inline
def BreachOfContract():
    -10000000000000000


GasStorage(Date('2011-6-1'), Date('2011-12-1'), 'GAS', 0, 0, 50000, TimeDelta('1m'))
""",

    observation_date='2011-1-1',
    interest_rate=2.5,
    path_count=20000,
    perturbation_factor=0.01,
    periodisation='monthly',

    price_process={
        'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
        'market': ['GAS'],
        'sigma': [0.5],
        'rho': [[1.0]],
        'curve': {
            'GAS': (
                ('2011-1-1', 13.5),
                ('2011-2-1', 11.0),
                ('2011-3-1', 10.0),
                ('2011-4-1', 9.0),
                ('2011-5-1', 7.5),
                ('2011-6-1', 7.0),
                ('2011-7-1', 6.5),
                ('2011-8-1', 7.5),
                ('2011-9-1', 8.5),
                ('2011-10-1', 10.0),
                ('2011-11-1', 11.5),
                ('2011-12-1', 12.0),
                ('2012-1-1', 13.5),
                ('2012-2-1', 11.0),
                ('2012-3-1', 10.0),
                ('2012-4-1', 9.0),
                ('2012-5-1', 7.5),
                ('2012-6-1', 7.0),
                ('2012-7-1', 6.5),
                ('2012-8-1', 7.5),
                ('2012-9-1', 8.5),
                ('2012-10-1', 10.0),
                ('2012-11-1', 11.5),
                ('2012-12-1', 12.0)
            )
        }
    }
)

assert 5 < results.fair_value.mean() < 7, results.fair_value.mean()
```

### Power station

An evaluation of a power station. This example imports a power station model from the library. It 
uses a market model with two correlated markets. The source code for the power station model is copied in below.

```python
results = calc_print_plot(
    title="Power Station",

    source_code="""from quantdsl.lib.powerplant2 import PowerPlant, Running
        
PowerPlant(Date('2012-1-1'), Date('2012-1-6'), Running())
""",

    observation_date='2011-1-1',
    interest_rate=2.5,
    path_count=20000,
    perturbation_factor=0.01,
    periodisation='daily',

    price_process={
        'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
        'market': ['GAS', 'POWER'],
        'sigma': [0.5, 0.3],
        'rho': [[1.0, 0.8], [0.8, 1.0]],
        'curve': {
            'GAS': [
                ('2011-1-1', 13.0),
                ('2012-1-1', 13.0),
                ('2012-1-2', 13.1),
                ('2012-1-3', 12.8),
                ('2012-1-4', 15.9),
                ('2012-1-5', 13.1),
            ],
            'POWER': [
                ('2011-1-1', 2.5),
                ('2012-1-1', 5.6),
                ('2012-1-2', 5.6),
                ('2012-1-3', 12.9),
                ('2012-1-4', 26.9),
                ('2012-1-5', 1.8),
            ]
        }
    }
)

assert 8 < results.fair_value.mean() < 10, results.fair_value.mean()
```

### Library

There is a small collection of Quant DSL modules in a library under `quantdsl.lib`. Putting Quant DSL source code in 
dedicated Python files makes it much easier to develop and maintain Quant DSL function definitions in a Python IDE.

Below is a copy of the Quant DSL source code for the library's power plant model `quantdsl.lib.powerplant2`, as used
 in the example above.

```python
from quantdsl.semantics import Choice, Market, TimeDelta, Wait, inline


def PowerPlant(start, end, duration_off):
    if (start < end):
        Wait(start,
            Choice(
                ProfitFromRunning(duration_off) + PowerPlant(
                    Tomorrow(start), end, Running()
                ),
                PowerPlant(
                    Tomorrow(start), end, Stopped(duration_off)
                )
            )
        )
    else:
        0


@inline
def ProfitFromRunning(duration_off):
    if duration_off > 1:
        0.75 * Power() - Gas()
    elif duration_off == 1:
        0.90 * Power() - Gas()
    else:
        1.00 * Power() - Gas()


@inline
def Power():
    Market('POWER')


@inline
def Gas():
    Market('GAS')


@inline
def Running():
    0


@inline
def Stopped(duration_off):
    duration_off + 1


@inline
def Tomorrow(today):
    today + TimeDelta('1d')

```
