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

## Introduction

The examples below use the library function `calc()` to evaluate Quant DSL source code. `calc()` uses the methods 
of the `QuantDslApplication` described above.

```python
from quantdsl.interfaces.calcandplot import calc
```

Simple numerical operations.

```python
results = calc("2 + 3 * 4 - 10 / 5")

assert results.fair_value == 12
```

Other numerical operations.

```python
results = calc("Max(9 // 2, Min(2**2, 12 % 7))")

assert results.fair_value == 4
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

Date and time values.

```python
import datetime


results = calc("Date('2011-1-1')")
assert results.fair_value == datetime.datetime(2011, 1, 1)

results = calc("Date('2011-1-10') - Date('2011-1-1')")
assert results.fair_value == datetime.timedelta(days=9)

results = calc("Date('2011-1-1') + 5 * TimeDelta('1d') < Date('2011-1-10')")
assert results.fair_value == True

results = calc("Date('2011-1-1') + 10 * TimeDelta('1d') < Date('2011-1-10')")
assert results.fair_value == False
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

assert results.fair_value == 11250
```   

The call args of the function definition can be used in an if-else block, so that different expressions can be 
returned depending upon the function call argument values.

Please note, the test expressions preceding the colon in the if-else block must be simple expressions involving the 
call args and must not involve any Quant DSL stochastic elements introduces below, such as `Market`, `Choice`, `Wait`, 
`Settlement`, `Fixing`. Also, calls to function definitions from test expressions in if statements are currently not 
supported.

Each function call becomes a node on a dependency graph. Each call is internally memoised, so if a function is 
called many times with the same argument values (and at the same effective present time), the function is only 
evaluated once, and the result is memoised and reused. This allows branched calculations to recombine efficienctly. 
For example, the following Finboncci function definition will evaluate in linear time (proportional to `n`).

```python
results = calc("""
def Fib(n):
    if n > 1:
        Fib(n-1) + Fib(n-2)
    else:
        n

Fib(60)
""")

assert results.fair_value == 1548008755920
```   

### Market

Underlying prices can be included in an expression with the `Market` element.

When a `Market` element is evaluated, the price of the named underlying commodity, at a particular date, is selected
from a simulation of market prices.

Simulated prices are generated by evolving a forward curve from from an `observation_date` by the given 
`price_process` according to the requirements (dates and markets) of the expression. A forward curve provides
estimates at the observation date of future prices for each market. The price process is calibrated with parameters,
 such as `sigma` (annualised historical volatility).

In this example, we use a one-factor multi-market Black Scholes price process (geometric Brownian motion). The 
calibration parameters it requires are `'market'`, a list of market names, and `'sigma'`, a list of annualised 
historical volatilities (expressed as a fraction of 1, rather than as a percentage). If there is more than one 
market, an additional parameter `'rho'` is required, which is a list of lists expressing the correlation between the
 factors. 

```python
price_process = {
    'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
    'market': ['GAS', 'POWER'],
    'sigma': [0.0, 0.0],
    'rho': [
        [1.0, 0.8],
        [0.8, 1.0]
    ],
    'curve': {
        'GAS': [
            ('2011-1-1', 10),
            ('2111-1-1', 1000)
        ],
        'POWER': [
            ('2011-1-1', 11),
            ('2111-1-1', 1100)
        ]
    },
}
```

Requirements for the simulation (dates and markets) are derived from the expression to be evaluated, so that if the 
expression requires a simulated price in a particular market at a particular date, the simulation will work to 
provide that value.

The date used by the `Market` element to select a price from the simulation is the "effective present time". The 
effective present time of a single `Market` element is the given `observation_time`.
Therefore, evaluating a single `Market` element will simply return the last value from the given forward curve for 
that market at the given `observation_date`.

```python
results = calc("Market('GAS')",
    observation_date='2011-1-1',
    price_process=price_process,
)
assert results.fair_value.mean() == 10
```

The number of samples in the random variable corresponds to the number of paths in the Monte Carlo simulation used 
to evaluate the expression, which defaults to `20000`.

```python
assert len(results.fair_value) == 20000
```
The number of paths in the Monte Carlo simulation can be adjusted by setting `path_count`. The accuracy of results 
can be doubled by increasing the path count by a factor of four.

```python
results = calc("Market('GAS')",
    observation_date='2011-1-1',
    price_process=price_process,
    path_count=80000,
)
assert len(results.fair_value) == 80000
```

Since `sigma` (above) is `0.0`, there is effectively no stochastic evolution of the forward curve, so the standard 
deviation of the result value is zero.

```python
assert results.fair_value.std() == 0.0
```

Evaluating at a later observation date will return a later value from the forward curve. The standard deviation of 
the result will still be zero, since the observation date and the date of the price are the still the same.

```python
results = calc("Market('GAS')",
    observation_date='2111-1-1',
    price_process=price_process,
)
assert results.fair_value.mean() == 1000
assert results.fair_value.std() == 0.0
```

Values at particular dates are selected from the given forward curve, with zero-order hold from the last known 
value. For example, using the `price_process` defined above, evaluating with `observation_date` of `2051-1-1` will 
fall back onto the value in the forward curve for `2011-1-1`, which is `10`.

```python
results = calc("Market('GAS')",
    observation_date='2051-1-1',
    price_process=price_process,
)
assert results.fair_value.mean() == 10
assert results.fair_value.std() == 0.0
```

### Settlement

The `Settlement` element expresses discounting a value to its net present value. Discounting is a function of the 
`interest_rate` and the duration in time between the date of the `Settlement` and the effective present time of its 
evaluation.

For example, an `interest_rate` of `2.5` percent per year discounts the value of `1000` in `'2111-1-1'` to a net 
present value of less than 100 in `2011-1-1`.

```python
results = calc("Settlement('2111-1-1', 1000.0)",
    observation_date='2011-1-1',
    interest_rate=2.5,
)

assert round(results.fair_value, 3) == 81.950, results.fair_value
raise Exception()
```

If the effective present time of the `Settlement` is the same as the settlement date, there is no discounting.

```python
results = calc("Fixing('2111-1-1', Settlement('2111-1-1', 10))",
    observation_date='2011-1-1',
    interest_rate=2.5,
)
assert_equal(results.fair_value, 10.0)
```

Similarly, if the `interest_rate` is `0.0`, there is no discounting.

```python
results = calc("Settlement('2111-1-1', 10)",
    observation_date='2011-1-1',
    interest_rate=0,
)
assert_equal(results.fair_value, 10.0)
```


### Fixing

The `Fixing` element is used to condition the "effective present time" of its included expression. If a
`Market` element is included in the `Fixing` element, then the effective present time of the 
`Market` element will be the date of the `Fixing`. 

The expression below represents the simulated price of the `'GAS'` market on `'2111-1-1'`.

Since `sigma` is still zero, the value of the expression observed at `'2011-1-1` is the value in the forward 
curve at `'2111-1-1'`.

```python
results = calc("Fixing('2111-1-1', Market('GAS'))",
    observation_date='2011-1-1',
    price_process=price_process,
)

assert results.fair_value.mean() == 1000
assert results.fair_value.std() == 0
```   

Now, let's recalibrate the price process to have a non-zero `sigma`, so that simulated prices will have some 
volatility.

```python
price_process['sigma'] = [0.02, 0.02]  # 2% historical annualised volatility.
```

With non-zero `sigma`, and with a difference between the `observation_date` and the date
 of the `Fixing`, results in stochastic evolution of the simulated price, and non-zero standard deviation
  in the value of the expression.
  
The non-zero standard deviation implies the mean of the samples will not be equal to the 
 scalar value take from the forward curve (but will tend towards that value as the number of samples tends to 
 infinity).

```python
results = calc("Fixing('2111-1-1', Market('GAS'))",
    observation_date='2011-1-1',
    price_process=price_process,
)

assert results.fair_value.mean() != 1000
assert results.fair_value.std() != 0
```   

Before continuing with the stochastic examples below, setting the random seed helps make test results 
repeatable. And to help keep things readable, some "helper" functions are defined: `assert_equal` and 
`assert_almost_equal`.

```python
import scipy

# Setting the random seed to make the results in the examples repeatable.
scipy.random.seed(1234)

def assert_equal(a, b):
    a_round = round(a, 3)
    b_round = round(b, 3)
    assert a_round == b_round, (a_round, b_round, a_round - b_round)

def assert_almost_equal(a, b):
    diff = abs(a - b)
    avg = max(1, (a + b) / 2)
    tol = 0.05 * avg
    assert diff < tol, (a, b, diff, tol)
```

### Wait

The `Wait` element combines `Settlement` and `Fixing`, so that a single date value is used both to condition the 
effective present time of the included expression, and also the value of that expression is discounted to the 
effective present time of the including expression.

```python
results = calc("Wait('2111-1-1', Market('GAS'))",
    observation_date='2011-1-1',
    price_process=price_process,
    interest_rate=2.5,
)

assert_almost_equal(82.044, results.fair_value.mean())
```   

### Choice

The `Choice` element uses the least-squares Monte Carlo approach proposed by Longstaff and 
Schwartz (1998) to compare the conditional expected value of each alternative.

```python
source_code = """
Choice(
    Wait('2051-1-1', Market('GAS')),
    Wait('2111-1-1', Market('GAS'))
)
"""
```

With a low interest rate, the later market is chosen.

```python

results = calc(source_code,
    observation_date='2050-1-1',
    price_process=price_process,
    interest_rate=2.5,
)
assert_almost_equal(217.390, results.fair_value.mean())
```

With a higher interest rate, the earlier market is chosen.

```python
results = calc(source_code,
    observation_date='2050-1-1',
    price_process=price_process,
    interest_rate=10,
)
assert_almost_equal(9.048, results.fair_value.mean())

```   

### European and American options

In general, an option can be expressed as a "wait" until a date for a "choice" between, on one hand, the 
difference between the price of an underlying and a strike price and, on the other hand, an alternative.

```python

def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

```

A European option can then be expressed as an option to transact an underlying commodity at a given strike price 
on a given date, the alternative being zero.

```python
def European(date, strike, underlying):
    Option(date, strike, underlying, 0)

def EuropeanPut(date, strike, underlying):
    European(date, -strike, -underlying)
```

An American option is similar: it is an option to exercise at a given strike price on the start date, with the 
alternative being an american option starting on the next date - and so on until the end date when the alternative is
 zero.

```python
def American(start, end, strike, underlying, step):
    if start <= end:
        Option(start, strike, underlying,
            American(start + step, end, strike, underlying, step)
        )
    else:
        0
```

The following function will make it easy below to recalculate the value of a european option.

```python
def calc_european(spot, strike, sigma):
    source_code = """
def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

def European(date, strike, underlying):
    Option(date, strike, underlying, 0)
   
European(Date('2012-1-10'), {strike}, Market('GAS'))
    """.format(strike=strike)
    
    results = calc(
        source_code=source_code,
        observation_date='2011-1-1',
        price_process={
            'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
            'market': ['GAS'],
            'sigma': [sigma],
            'curve': {
                'GAS': [
                    ('2011-1-1', spot),
                ]
            },
        },
        interest_rate=0,
    )
    return results.fair_value.mean()
    
    
```

If the strike price of a European option is the same as the price of the underlying, without any volatility (`sigma` 
is `0`) the value is zero.

```python
assert_equal(0.0, calc_european(spot=10, strike=10, sigma=0))
```

If the strike price is less than the underlying, without any volatility, the value is the difference between the 
strike and the underlying.

```python
assert_equal(2.0, calc_european(spot=10, strike=8, sigma=0))
```

If the strike price is greater than the underlying, without any volatility, the value is zero.

```python
assert_equal(0, calc_european(spot=10, strike=12, sigma=0.0))
```

If the strike price is the same as the underlying, with some volatility in the price of the underlying, there
 is some value in the option.

```python
assert_almost_equal(3.522, calc_european(spot=10, strike=10, sigma=0.9))
```

If the strike price is less than the underlying, with some volatility in the price of the underlying (`sigma`) there
 is more value in the option than without volatility.

```python
assert_almost_equal(4.252, calc_european(spot=10, strike=8, sigma=0.9))
```

If the strike price is greater than the underlying, with some volatility in the price of the underlying (`sigma`) there
 is still a little bit of value in the option than without volatility.

```python
assert_almost_equal(2.935, calc_european(spot=10, strike=12, sigma=0.9))
```

These results can be compared with results from the Black-Scholes analytic formula for European options.

```python
from scipy.stats import norm
import math

def european_blackscholes(spot, strike, sigma):

    S = float(spot) # spot price
    K = float(strike) # strike price
    r = 0.0 # annual risk free rate / 100
    t = 1.0  # duration (years)
    sigma = max(0.00000001, sigma) # annual historical volatility / 100
    
    sigma_squared_t = sigma**2.0 * t
    sigma_root_t = sigma * math.sqrt(t)
    try:
        math.log(S / K)
    except:
        raise Exception((S, K))
    d1 = (math.log(S / K) + t * r + 0.5 * sigma_squared_t) / sigma_root_t
    d2 = d1 - sigma_root_t
    Nd1 = norm(0, 1).cdf(d1)
    Nd2 = norm(0, 1).cdf(d2)
    e_to_minus_rt = math.exp(-1.0 * r * t)
    # Put option.
    # optionValue = (1-Nd2)*K*e_to_minus_rt - (1-Nd1)*S
    # Call option.
    return Nd1 * S - Nd2 * K * e_to_minus_rt


spot = 10

for strike in [9, 10, 11]:
    for sigma in [0, 0.1, 0.2, 0.3]:
    
        assert_almost_equal(
            european_blackscholes(spot, strike, sigma),
            calc_european(spot, strike, sigma)
        )
```

The american option can be evaluated in the same way.

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
assert_almost_equal(3.69, results.fair_value.mean())
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
    periodisation='monthly',
    verbose=True,
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

assert_almost_equal(6.15, results.fair_value.mean())
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
    periodisation='daily',
    verbose=True,
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

assert_almost_equal(9.077, results.fair_value.mean())
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

## Acknowledgments

The *Quant DSL* language was partly inspired by the paper
*[Composing contracts: an adventure in financial engineering (functional pearl)](
http://research.microsoft.com/en-us/um/people/simonpj/Papers/financial-contracts/contracts-icfp.htm
)* by Simon Peyton Jones and others. The idea of orchestrating evaluations with a dependency graph,
to help with parallel and distributed execution, was inspired by a [talk about dependency graphs by
Kirat Singh](https://www.youtube.com/watch?v=lTOP_shhVBQ). The `quantdsl` Python package makes lots
of use of design patterns, the NumPy and SciPy packages, and the Python `ast` ("Absract Syntax Trees")
module. We have also been encourged by members of the [London Financial Python User Group](
https://www.google.co.uk/search?q=London+Financial+Python+User+Group), where the  *Quant DSL*
expression syntax and semantics were first presented.
