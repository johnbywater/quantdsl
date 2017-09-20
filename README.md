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


## Definition and implementation

The syntax of Quant DSL expressions has been
[formally defined](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf),
the semantic model is supported with mathematical proofs.

This package is an implementation of the language in Python. Function definitions are also supported, to ease 
construction of Quant DSL expressions. The import statement is also supported to allow function definitions to be 
used from a library (see below).

Steps for evaluating a contract include: specification of a model of a contract; calibration of a stochastic process
for the underlying prices; simulation using the price process of future prices underlying the contract; and evaluation
of the contract model against the simulation.

The library provides an application class `QuantDslApplication` which 
has methods that support these steps: `compile()`, `simulate()` and `evaluate()`.


## Introduction
The examples below use the library function calc_print_plot() to evaluate contracts.

```python
from quantdsl.interfaces.calcandplot import calc
```

Simple calculations.

```python
from quantdsl.interfaces.calcandplot import calc

results = calc("2 + 3 * 4 - 10 / 5")

assert results.fair_value == 12, results.fair_value
```

Other binary operations.

```python
results = calc("Max(9 // 2, Min(2**2, 12 % 7))")

assert results.fair_value == 4, results.fair_value
```

Logical operations.

```python
assert calc("1 and 2").fair_value == True
assert calc("0 and 2").fair_value == False
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
firstly replaced with the expression returned by the function definition, so that a larger expression is formed.

The call args of the function definition can be used as names in the function definition's expressions. The call arg 
values will be used to evaluate the expression returned by the function.

```python
results = calc(source_code="""
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

Each call to a (non-inlined) function definition becomes a node on a dependency graph. Each call is internally  
memoised, so it is only called once with the same argument values, and the result of such a call is reused.

```python
results = calc(source_code="""
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

results = calc(
    source_code="""Market('GAS')""",
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

results = calc(
    source_code="""Fixing('2112-1-1', Market('GAS'))""",
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

results = calc(
    source_code="""Fixing('2112-1-1', Max(1000, Market('GAS')))""",
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
results = calc(
    source_code="""Settlement('2111-1-1', 10)""",
    observation_date='2011-1-1',
    interest_rate=2.5,
)

assert results.fair_value < 1, results.fair_value
```

If the effective present time is the same as the settlement date, there is no discounting.

```python
results = calc(
    source_code="""Settlement('2111-1-1', 10)""",
    observation_date='2111-1-1',
    interest_rate=2.5,
)
assert results.fair_value == 10, results.fair_value

results = calc(
    source_code="""Fixing('2111-1-1', Settlement('2111-1-1', 10))""",
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

results = calc(
    # source_code="""Settlement('2112-1-1', Fixing('2112-1-1', Market('GAS')))""",
    source_code="""Wait('2112-1-1', Market('GAS'))""",
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


## Examples of usage

The examples below use the library function `calc_print_plot()` to evaluate contracts, and print and plot results.

```python
from quantdsl.interfaces.calcandplot import calc_print_plot
```

If you run these examples, the deltas for each market in each period will be calculated, and estimated risk neutral 
 hedge positions will be printed for each market in each period, along with the overall fair value. A plot will be 
 displayed showing underlying prices, the cumulative hedge positions, and the cummulate cash position from the hedge
 positions.

The plot will also show the statistical distribution of the simulated prices, and the statistical error of the hedge 
 positions and the cash flow. Comparing the resulting net cash position with the fair value gives an indication of 
 how well the deltas are performing.


### Lift

The examples here use the `Lift` element to specify deltas with respect to each period (daily, 
monthly, yearly) for each market across the term of the contract.
 

### Gas Storage

An evaluation of a gas storage facility. This example uses a forward curve that reflects seasonal variations across 
the term of the contract. 

```python
results = calc_print_plot(
    title="Gas Storage",
    
    source_code="""
def GasStorage(start, end, commodity_name, quantity, target, limit, step, period):
    if ((start < end) and (limit > 0)):
        if quantity <= 0:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step, period, target),
                Inject(start, end, commodity_name, quantity, limit, step, period, target, 1),
            ))
        elif quantity >= limit:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step, period, target),
                Inject(start, end, commodity_name, quantity, limit, step, period, target, -1),
            ))
        else:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step, period, target),
                Inject(start, end, commodity_name, quantity, limit, step, period, target, 1),
                Inject(start, end, commodity_name, quantity, limit, step, period, target, -1),
            ))
    else:
        if target < 0 or target == quantity:
            return 0
        else:
            return BreachOfContract()


@inline
def Continue(start, end, commodity_name, quantity, limit, step, period, target):
    GasStorage(start + step, end, commodity_name, quantity, target, limit, step, period)


@inline
def Inject(start, end, commodity_name, quantity, limit, step, period, target, vol):
    Continue(start, end, commodity_name, quantity + vol, limit, step, period, target) - \
    vol * Lift(commodity_name, period, Market(commodity_name))


@inline
def BreachOfContract():
    -10000000000000000


GasStorage(Date('2011-6-1'), Date('2011-12-1'), 'GAS', 0, 0, 50000, TimeDelta('1m'), 'monthly')
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
        'alpha': [0.1],
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

assert 8 < results.fair_value.mean() < 10, results.fair_value.mean()
```

### Power Station

An evaluation of a power station. This time, the source code imports a power station model from the library. It uses
 a market model with two correlated markets. The source code for the power station model is copied in below.

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

### Library of models

Quant DSL source code for the library power plant model `quantdsl.lib.powerplant2`, as used in the example 
above.

```python
from quantdsl.semantics import Choice, Lift, Market, TimeDelta, Wait, inline


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
        return 0


@inline
def ProfitFromRunning(duration_off):
    if duration_off > 1:
        return 0.75 * Power() - Gas()
    elif duration_off == 1:
        return 0.90 * Power() - Gas()
    else:
        return 1.00 * Power() - Gas()


@inline
def Power():
    Lift('POWER', 'daily', Market('POWER'))


@inline
def Gas():
    Lift('GAS', 'daily', Market('GAS'))


@inline
def Running():
    return 0


@inline
def Stopped(duration_off):
    return duration_off + 1


@inline
def Tomorrow(today):
    return today + TimeDelta('1d')

```
