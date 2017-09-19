# Quant DSL

***Domain specific language for quantitative analytics in finance and trading.***

[![Build Status](https://secure.travis-ci.org/johnbywater/quantdsl.png)](https://travis-ci.org/johnbywater/quantdsl)


## Install

Use pip to install the [latest distribution](https://pypi.python.org/pypi/quantdsl) from
the Python Package Index.

```
pip install --upgrade pip
pip install quantdsl
```

Please register any [issues on GitHub](https://github.com/johnbywater/quantdsl/issues).

To avoid disturbing your system's site packages, it is recommended to install
into a new virtual Python environment, using [Virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

## Overview

Quant DSL is a functional programming language for modelling derivative instruments.

At the heart of Quant DSL is a set of elements - e.g. *Settlement*, *Fixing*, *Choice*, *Market* - which encapsulate 
maths used in finance and trading. The elements of the language can be freely composed into expressions
of value. User defined functions generate extensive dependency graphs that effectively model and evaluate exotic
derivatives.

The syntax of Quant DSL expressions has been
[formally defined](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf),
the semantic model is supported with mathematical proofs.

This package is an implementation of the language in Python.

Function definitions are also supported, to ease construction of Quant DSL expressions.

The import statement is also supported to allow function definitions to be used from a library.


## Usage Example

The library provides a convenience function `calc_and_plot()` can be used to evaluate contracts.

Steps for evaluating a contract include: specification of a model of a contract; calibration of a price process
for the underlying prices; simulation of future prices underlying the contract; and evaluation of the contract model
against the simulation. The library provides an application class `QuantDslApplication` which has methods that 
support these steps: `compile()`, `simulate()` and `evaluate()`. The function `calc_and_plot()` uses those methods of 
that application to evaluate contracts.
 

```python
from quantdsl.interfaces.calcandplot import calc_and_plot
```

### Gas Storage

Here's an evaluation of a gas storage facility.

```python
calc_and_plot(
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
def BreachOfContract():
    -10000000000000000

@inline
def Continue(start, end, commodity_name, quantity, limit, step, period, target):
    GasStorage(start + step, end, commodity_name, quantity, target, limit, step, period)


@inline
def Inject(start, end, commodity_name, quantity, limit, step, period, target, vol):
    Continue(start, end, commodity_name, quantity + vol, limit, step, period, target) - \
    vol * Lift(commodity_name, period, Market(commodity_name))


GasStorage(Date('2011-6-1'), Date('2011-9-1'), 'GAS', 0, 0, 50000, TimeDelta('1m'), 'monthly')
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

```

### Power Station

Here's an evaluation of a power station.

```python
calc_and_plot(
    title="Power Station",

    source_code="""
def PowerStation(start, end, gas, power, duration_off):
    if (start < end):
        Wait(start,
            Choice(
                ProfitFromRunning(gas, power, duration_off) + PowerStation(
                    Tomorrow(start), end, gas, power, Running()
                ),
                PowerStation(
                    Tomorrow(start), end, gas, power, Stopped(duration_off)
                )
            )
        )
    else:
        return 0

@inline
def ProfitFromRunning(gas, power, duration_off):
    if duration_off > 1:
        return 0.75 * power - gas
    elif duration_off == 1:
        return 0.90 * power - gas
    else:
        return 1.00 * power - gas

@inline
def Running():
    return 0

@inline
def Stopped(duration_off):
    return duration_off + 1

@inline
def Tomorrow(today):
    return today + TimeDelta('1d')

PowerStation(Date('2012-01-01'), Date('2012-01-13'), Market('GAS'), Market('POWER'), Running())
""",

    observation_date='2011-1-1',
    interest_rate=2.5,
    path_count=20000,
    perturbation_factor=0.01,
    periodisation='monthly',

    price_process={
        'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
        'market': ['GAS', 'POWER'],
        'sigma': [0.5, 0.3],
        'rho': [[1.0, 0.8], [0.8, 1.0]],
        'curve': {
            'GAS': [
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
            ],
            'POWER': [
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
            ]
        }
    }
)

```
