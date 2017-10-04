# Quant DSL

***Domain specific language for quantitative analytics in finance and trading.***

[![Build Status](https://secure.travis-ci.org/johnbywater/quantdsl.png)](https://travis-ci.org/johnbywater/quantdsl)


## Install

Use pip to install the [latest distribution](https://pypi.python.org/pypi/quantdsl) from
the Python Package Index.

```
pip install quantdsl
```

To avoid disturbing your system's site packages, it is recommended to install
into a new virtual Python environment, using [Virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

Please note, this library depends on SciPy, which fails to install with some older versions of pip. In case of 
difficulty, please try again after upgrading pip.

```
pip install --upgrade pip
pip install quantdsl
```

Please register any [issues on GitHub](https://github.com/johnbywater/quantdsl/issues).


## Overview

Quant DSL is domain specific language for quantitative analytics in finance and trading.

At the heart of Quant DSL is a set of elements (e.g. `Settlement`, `Fixing`, `Market`, `Wait`, `Choice`) which 
encapsulate maths used in finance and trading. The elements of the language can be freely composed into expressions
of value.

The syntax of Quant DSL expressions has been defined with
[Backus–Naur Form](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form).

```
<Expression> ::= <Constant>
    | "Settlement(" <Date> "," <Expression> ")"
    | "Market(" <MarketId> ")"
    | "Fixing(" <Date> "," <Expression> ")"
    | "Wait(" <Date> "," <Expression> ")"
    | "Choice(" <Expression> "," <Expression> ")"
    | "Max(" <Expression> "," <Expression> ")"
    | <Expression> "+" <Expression>
    | <Expression> "-" <Expression>
    | <Expression> "*" <Expression>
    | <Expression> "/" <Expression>
    | "-" <Expression>
```

The semantics are defined with mathematical expressions commonly used within quantitative analytics, such as 
discounting from future to present value (`Settlement`, `Wait`), geometric Brownian motion (`Market`), and the
least-squares Monte Carlo approach proposed by Longstaff and Schwartz (`Choice`).

In the table below, expression `v` defines a function `[[v]](t)` from present time `t` to a random
variable in a probability space. For market `i`, the last price `Si` and volatility `σi` are determined
using only market price data generated before `t0`. Brownian motion `z` is used in diffusion.
Constant interest rate `r` is used in discounting. Expectation `E` is conditioned
on filtration `F`.

```
[[Settlement(d, x)]](t) = e ** (r * (t−d)) * [[x]](t)

[[Fixing(d, x)]](t) = [[x]](d)

[[Market(i)]](t) = Si * e ** (σi * z(t−t0)) − 0.5 * σi ** 2 * (t−t0)

[[Wait(d, x)]](t) = [[Settlement(d, Fixing(d, x))]](t)

[[Choice(x, y)]](t) = max(E[[[x]](t)|F(t)], E[[[y]](t)|F(t)])

[[x + y]](t) = [[x]](t) + [[y]](t)
```

The validity of Monte Carlo simulation for all possible expressions in the language is  
[proven by induction](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf).


## Implementation

This package is an implementation of the Quant DSL in Python. 

In addition to the Quant DSL expressions, function `def`
statements are supported. User defined functions can be used
to generate an extensive dependency graph of Quant DSL expressions
that efficiently model complex optionality.

The `import` statement is also supported, to allow Quant DSL function 
definitions and expressions to be used and maintained in a library as
normal Python code.


## Introduction

The work of a quantitative analyst involves modelling the value of a derivative,
calibrating a stochastic process for the underlying prices, simulating future prices
of the underlyings, and evaluating of the model of the derivative against the simulation.
This library provides an application class `QuantDslApplication` which has methods that
support this work: `compile()`, `simulate()` and `evaluate()`.

During compilation of the specification source code, the application constructs a
dependency graph of function calls. The simulation is generated according to requirements
derived from the depedency graph, and a calibrated price process. During evaluation, nodes
are evaluated when they are ready to be evaluated, and intermediate call results are discarded
as soon as they are no longer required, which means memory usage is mostly constant during
evaluation. For the delta calculations, nodes are selectively re-evaluated with perturbed
values, according to the periods and markets they involve.

The examples below use the library function `calc()` to evaluate Quant DSL source code.

`calc()` uses the methods of the `QuantDslApplication` described above.

```python
from quantdsl.interfaces.calcandplot import calc
```

`calc()` returns a results object, with an attribute `fair_value` that is the computed value of the given 
Quant DSL expression. 

```python
results = calc("0")
assert results.fair_value == 0
```

```python
results = calc("2 + 3")
assert results.fair_value == 5
```

```python
results = calc("2 * 3")
assert results.fair_value == 6
```

### Settlement

The `Settlement` element discounts the value of the included expression from the given date to the effective present
 time.

```
<Settlement> ::= "Settlement(" <Date> ", " <Expression> ")"
```

Discounting is a function of the `interest_rate` and the duration in time between the date of the `Settlement` 
element and the effective present time of its evaluation. The formula used for discounting by the `Settlement` 
element is `e**-rt`. The `interest_rate` is the therefore the continuously compounding risk free rate (not the 
annual equivalent rate).

For example, with a continuously compounding `interest_rate` of `2.5` percent per year, the value `10` settled in 
`'2111-1-1'` has a present value at the `observation_date` of `'2011-1-1'` of `82.085`.

```python
results = calc("Settlement('2111-1-1', 1000)",
    observation_date='2011-1-1',
    interest_rate=2.5,
)

assert round(results.fair_value, 2) == 82.08, results.fair_value
```

Similarly, the value of `82.085` settled in `'2011-1-1'` has a present value at the `observation_date` of `'2111-1-1'` 
of `1000.00`.

```python
results = calc("Settlement('2011-1-1', 82.085)",
    observation_date='2111-1-1',
    interest_rate=2.5,
)

assert round(results.fair_value, 2) == 1000.00, results.fair_value
```

Todo: Support annual equivalent rate?


### Fixing

The `Fixing` element conditions with its give date the effective present time of its included expression.

```
<Fixing> ::= "Fixing(" <Date> "," <Expression> ")"
```

For example, if a `Fixing` element includes a `Settlement` element, then the effective present time of the 
included `Settlement` element will be the given date of the `Fixing`.

The expression below represents the present value in `'2051-1-1'` of the value of `1000` to be settled on 
`'2111-1-1'`.

```python
results = calc("Fixing('2051-1-1', Settlement('2111-1-1', 1000))",
    interest_rate=2.5,
)

assert round(results.fair_value, 2) == 223.13, results.fair_value
```   

*Before continuing with the stochastic examples below, setting the random seed helps make test results 
repeatable. And to help keep things readable, some "helper" functions are defined: `assert_equal` and 
`assert_almost_equal`.*

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


### Market

The `Market` element effectively estimates spot prices that could be agreed in the future.

```
<Market> ::= "Market(" <MarketId> ")"
```

When a `Market` element is evaluated, it returns a random variable selected from a simulation
 of market prices. Selecting an estimated price from the simulation requires the ID (or name)
 of the market, and a fixing date and a delivery date: when the price would be agreed, and when
 the goods would be delivered. The ID of the `Market` is included in the element (e.g. `'GAS'` or `'POWER'` 
 above). Both the fixing date and the delivery date are determined by the "effective present time"
when the element is evaluated (see `Fixing`).

The price simulation is generated by a price process. In this example, the library's one-factor
multi-market Black  Scholes price process `BlackScholesPriceProcess` is used to generate correlated
geometric Brownian motions.

The calibration parameters required by `BlackScholesPriceProcess` are `market` (a list of market names), and 
`sigma`, (a list of annualised historical volatilities, expressed as a fraction of 1, rather than as a 
percentage).

When the simulation involves two or more markets, an additional parameter `rho` is required, which represents
the correlation between the markets (a symmetric matrix expressed as a list of lists).

A forward `curve` is required to provide estimates of current prices for each market at the given 
`observation_date`. The prices in the forward curve are prices that can be agreed at the `observation_date` for 
delivery at the specified dates. These prices are evolved into estimates of prices that could be agreed at future 
dates.

Requirements for the simulation (dates and markets) are derived from the expression to be evaluated. If the 
expression only involves the price in a particular market for goods to be delivered and agreed at a particular 
date, the simulation will provide that value.


```python
price_process = {
    'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
    'market': ['GAS', 'POWER'],
    'sigma': [0.02, 0.02],
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

A `Market` element evaluated at the `observation_date` will simply return the last value from the given forward
 curve for that market at the given `observation_date`.
 
```python
results = calc("Market('GAS')",
    observation_date='2011-1-1',
    price_process=price_process,
)
```

Since the `Market` element uses random variables from the price simulation, so the results are random variables, and
 we need to take the `mean()` to obtain a scalar value.

```python
assert results.fair_value.mean() == 10
```

If the forward curve doesn't contain a price at the required delivery date, a price at 
an earlier delivery date is used (with zero order hold).

```python
results = calc("Market('GAS')",
    observation_date='2012-3-4',
    price_process=price_process,
)

assert results.fair_value.mean() == 10
```

Evaluating at a much later observation date will return the later value from the forward curve.
 
```python
results = calc("Market('GAS')",
    observation_date='2111-1-1',
    price_process=price_process,
)
assert results.fair_value.mean() == 1000
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

Values at particular dates are selected from the given forward curve, with zero-order hold from the last known 
value. For example, using the `price_process` defined above, evaluating with `observation_date` of `2051-1-1` will 
fall back onto the value in the forward curve for `2011-1-1`, which is `10`.

```python
results = calc("Market('GAS')",
    observation_date='2051-1-1',
    price_process=price_process,
)
assert results.fair_value.mean() == 10
```

In the examples so far, there has bben no difference between the effective present time of the `Market` element and 
the `observation_date` of the evaluation. Therefore, there is no stochastic evolution of the forward curve, and the 
standard deviation of the result value is zero.

```python
assert results.fair_value.std() == 0.0
```

If a `Market` element is included within a `Fixing` element, the value of the expression will be the price 
that can be expected to be agreed at the date provided by the `Fixing` element.

With Brownian motion provided by the price process, the random variable used to estimate a price that could be 
agreed in the future has a statistical distribution with non-zero standard deviation. The mean value of the 
expected price will only approximate to the value taken from the forward `curve`.

```python
results = calc("Fixing('2051-1-1', Market('GAS'))",
    observation_date='2011-1-1',
    price_process=price_process,
)
assert results.fair_value.std() > 0.0
assert results.fair_value.mean() != 10
```

The `ForwardMarket` element can be used to specify a delivery date that is different from the fixing date (when the 
price for that delivery would be agreed).

For each price that is required, the simulation uses the delivery date to select from the forward `curve` a price 
that can be agreed at the `observation_date`, and then evolves that price through the fixing dates, according to the 
configured `price_process`.


### Wait

The `Wait` element combines `Settlement` and `Fixing`, so that a single date value is used both to condition the 
effective present time of the included expression, and also the value of that expression is discounted to the 
present time effective when evaluating the `Wait` element.

```
<Wait> ::= "Wait(" <Date> "," <Expression> ")"
```

For example, the present value at the `observation_date` of `'2011-1-1'` of one unit of `'GAS'` delivered on 
`'2111-1-1'` is approximately `82.08`.

```python
results = calc("Wait('2111-1-1', Market('GAS'))",
    price_process=price_process,
    observation_date='2011-1-1',
    interest_rate=2.5,
)
assert_almost_equal(results.fair_value.mean(), 82.08)
```


### Choice

The `Choice` element uses the least-squares Monte Carlo approach proposed by Longstaff and 
Schwartz (1998) to compare the conditional expected value of each alternative.

```
<Choice> ::= "Choice(" <Expression> "," <Expression> ")"
```

For example, the value of the choice at `observation_date` of `'2011-1-1'` between one unit of `'GAS'` either on 
`'2051-1-1'` or `'2111-1-1'` is `217.39`.

```python
source_code = """
Choice(
    Wait('2051-1-1', Market('GAS')),
    Wait('2111-1-1', Market('GAS'))
)
"""

results = calc(source_code,
    observation_date='2011-1-1',
    price_process=price_process,
    interest_rate=2.5,
)
assert_almost_equal(82.08, results.fair_value.mean())
assert_almost_equal(16.67, results.fair_value.std())
```

Todo: When does this differ in value from Max()?


### Functions definitions

Quant DSL source code can include function definitions. Expressions can involve calls to functions.

When evaluating an expression that involves a call to a function definitions, the call to the 
function definition is effectively replaced with the expression returned by the function definition,
so that a larger expression is formed.

The call args of the function definition can be used as names in the function definition's expressions. The call arg 
values will be used to evaluate the expression returned by the function.

```python
results = calc("""
def Function(a):
    2 * a

Function(10)
""")

assert results.fair_value == 20
```   

The call args of the function definition can be used in an if-else block, so that different expressions can be 
returned depending upon the function call argument values.

Each function call becomes a node on a dependency graph. For efficiency, each call is internally memoised, so if a 
function is called many times with the same argument values (and at the same effective present time), the function 
is only evaluated once, and the result is memoised and reused. This allows branched calculations to recombine 
efficienctly. For example, the following Finboncci function definition will evaluate in linear time (proportional to
 `n`).

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

Function definitions can be used to refactor complex expressions. For example, if the expression is the sum of a 
series of settlements on different dates, the expression without a function definition might be:

```python
source_code = """
Settlement(Date('2011-1-1'), 10) + Settlement(Date('2011-2-1'), 10) + Settlement(Date('2011-3-1'), 10) + \
Settlement(Date('2011-4-1'), 10) + Settlement(Date('2011-5-1'), 10) + Settlement(Date('2011-6-1'), 10) + \
Settlement(Date('2011-8-1'), 10) + Settlement(Date('2011-8-1'), 10) + Settlement(Date('2011-9-1'), 10) + \
Settlement(Date('2011-10-1'), 10) + Settlement(Date('2011-11-1'), 10) + Settlement(Date('2011-12-1'), 10)
"""
results = calc(source_code,
    observation_date='2011-1-1',
    interest_rate=10,
)
assert_almost_equal(results.fair_value, 114.592)
```
 
Instead the expression could be refactored with a function definition.
 
```python
source_code = """
def Repayments(start, end, installment):
    if start <= end:
        Settlement(start, installment) + Repayments(start + TimeDelta('1m'), end, installment)
    else:
        0
        
Repayments(Date('2011-1-1'), Date('2011-12-1'), 10)
"""
results = calc(source_code,
    observation_date='2011-1-1',
    interest_rate=10,
)
assert_almost_equal(results.fair_value, 114.592)
```

Please note, any `if` statement test expressions (the expressions preceding the colons in the `if` statement) must be 
simple expressions involving the call args, and must not involve any Quant DSL stochastic elements, such 
as `Market`, `Choice`, `Wait`, `Settlement`, `Fixing`. Calls to function definitions from test expressions in if 
statements is supported, but the function definitions must not contain any of the stochastic elements.


### Derivative options

In general, an option can be expressed as waiting until an `expiry` date to choose between, on one hand, the 
difference between the value of an `underlying` expression and a `strike` expression,
and, on the other hand, an `alternative` expression.

```python
def Option(expiry, strike, underlying, alternative):
    Wait(expiry, Choice(underlying - strike, alternative))
```

A European option can then be expressed simply as an `Option` with zero alternative.

```python
def EuropeanOption(expiry, strike, underlying):
    Option(expiry, strike, underlying, 0)
   
```
The `AmericanOption` can be expressed as an `Option` to exercise at a given `strike` price on 
the `start` date, with the alternative being another `AmericanOption` starting on the next date - and so on until the 
`expiry` date, when the `alternative` becomes zero.

```python
def AmericanOption(start, expiry, strike, underlying, step):
    if start <= expiry:
        Option(start, strike, underlying,
            AmericanOption(start + step, expiry, strike, underlying, step)
        )
    else:
        0
```

A European put option can be expressed as a `EuropeanOption`, with negated underlying and strike expressions.

```python
def EuropeanPut(expiry, strike, underlying):
    EuropeanOption(expiry, -strike, -underlying)
```

A European stock option can be expressed as a `EuropeanOption`, with the `underlying` being the spot price at the 
start of the contract, discounted forward from `start`, and observed at the option `expiry` time.

```python
def EuropeanStockOption(expiry, strike, stock):
    EuropeanOption(expiry, strike, StockMarket(stock))

def StockMarket(stock):
    Settlement(ObservationDate(), ForwardMarket(ObservationDate(), stock))
```

The built-in `ObservationDate` element evaluates to the `observation_date` passed to the the `calc()` function.

Let's evaluate a European stock option at different strike prices, volatilities, and interest rates.

The following function `calc_european` will make it easier to evaluate the option several times.

```python
def calc_european(spot, strike, sigma, rate):
    source_code = """
def Option(expiry, strike, underlying, alternative):
    Wait(expiry, Choice(underlying - strike, alternative))

def EuropeanOption(expiry, strike, underlying):
    Option(expiry, strike, underlying, 0)
   
def EuropeanStockOption(expiry, strike, stock):
    EuropeanOption(expiry, strike, StockMarket(stock))

def StockMarket(stock):
    Settlement(ObservationDate(), ForwardMarket(ObservationDate(), stock))
    
EuropeanStockOption(Date('2012-1-1'), {strike}, 'ACME')
    """.format(strike=strike)
    
    results = calc(
        source_code=source_code,
        observation_date='2011-1-1',
        price_process={
            'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
            'market': ['ACME'],
            'sigma': [sigma],
            'curve': {
                'ACME': [
                    ('2011-1-1', spot),
                ]
            },
        },
        interest_rate=rate,
    )
    return results.fair_value.mean()
```

If the strike price of a European option is the same as the price of the underlying, without any volatility (`sigma` 
is `0`) the value is zero.

```python
assert_equal(0.0, calc_european(spot=10, strike=10, sigma=0, rate=0))
```

If the strike price is less than the underlying, without any volatility, the value is the difference between the 
strike and the underlying.

```python
assert_equal(2.0, calc_european(spot=10, strike=8, sigma=0, rate=0))
```

If the strike price is greater than the underlying, without any volatility, the value is zero.

```python
assert_equal(0, calc_european(spot=10, strike=12, sigma=0, rate=0))
```

If the strike price is the same as the underlying, with some volatility in the price of the underlying, there
 is some value in the option.

```python
assert_almost_equal(3.522, calc_european(spot=10, strike=10, sigma=0.9, rate=0))
```

If the strike price is less than the underlying, with some volatility in the price of the underlying (`sigma`) there
 is more value in the option than without volatility.

```python
assert_almost_equal(4.252, calc_european(spot=10, strike=8, sigma=0.9, rate=0))
```

If the strike price is greater than the underlying, with some volatility in the price of the underlying (`sigma`) there
 is still a little bit of value in the option.

```python
assert_almost_equal(2.935, calc_european(spot=10, strike=12, sigma=0.9, rate=0))
```

These results compare well with results from the Black-Scholes analytic formula for European stock options.


### Gas storage

An evaluation of a gas storage facility. The value obtained is the extrinsic value. The intrinstic value can be 
obtained by setting the volatility `sigma` to `0`, and evaluating with `path_count` of `1`.

This example uses a forward curve that has seasonal variation (prices are high in winter and low in 
summer).

This example uses the library function `calc_print_plot()` to calculate, print, and plot results.

```python
from quantdsl.interfaces.calcandplot import calc_print_plot
```

The deltas for each market in each period will be calculated, and estimated risk neutral 
 hedge positions will be printed for each market in each period, along with the overall fair value. A plot will be 
 displayed showing underlying prices and the cumulative hedge positions for each underlying, and the net cash from the 
 hedge positions (profit and loss).

The plot will also show the statistical distribution of the simulated prices, and the statistical error of the hedge 
 positions and the cash flow. Comparing the resulting net cash position with the fair value gives an indication of 
 how well the deltas are performing.

```python
source_code = """
def GasStorage(start, end, market, quantity, target, limit, step):
    if ((start < end) and (limit > 0)):
        if quantity <= 0:
            Wait(start, Choice(
                Continue(start, end, market, quantity, target, limit, step),
                Inject(start, end, market, quantity, target, limit, step, 1),
            ))
        elif quantity >= limit:
            Wait(start, Choice(
                Continue(start, end, market, quantity, target, limit, step),
                Inject(start, end, market, quantity, target, limit, step, -1),
            ))
        else:
            Wait(start, Choice(
                Continue(start, end, market, quantity, target, limit, step),
                Inject(start, end, market, quantity, target, limit, step, 1),
                Inject(start, end, market, quantity, target, limit, step, -1),
            ))
    else:
        if target < 0 or target == quantity:
            0
        else:
            BreachOfContract()


@inline
def Continue(start, end, market, quantity, target, limit, step):
    GasStorage(start + step, end, market, quantity, target, limit, step)


@inline
def Inject(start, end, market, quantity, target, limit, step, vol):
    Continue(start, end, market, quantity + vol, target, limit, step) - \
    vol * market


@inline
def BreachOfContract():
    -10000000000000000

@inline
def Empty():
    0

@inline
def Full():
    50000


GasStorage(Date('2011-6-1'), Date('2011-12-1'), Market('GAS'), Empty(), Empty(), Full(), TimeDelta('1m'))
"""

results = calc_print_plot(
    title="Gas Storage",
    source_code=source_code,
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

Todo: Discounting of forward contracts when calculating hedge positions, so the quantities will be larger with 
a higher interest rate. Also the price of the hedge is the forward price at the observation time, rather than the 
spot price at the forward time.


### Power station

An evaluation of a power station. This example imports a power station model from the library. It 
uses a market model with two correlated markets. The source code for the power station model is copied in below.

```python
source_code = """
from quantdsl.lib.powerplant2 import PowerPlant, Running
        
PowerPlant(Date('2012-1-1'), Date('2012-1-6'), Running())
"""

results = calc_print_plot(
    title="Power Station",
    source_code=source_code,
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
                ProfitFromRunning(start, duration_off) + PowerPlant(
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
def ProfitFromRunning(start, duration_off):
    if duration_off > 1:
        0.75 * Power(start) - Gas(start)
    elif duration_off == 1:
        0.90 * Power(start) - Gas(start)
    else:
        1.00 * Power(start) - Gas(start)


@inline
def Power(start):
    Market('POWER', start + TimeDelta('1d'))


@inline
def Gas(start):
    Market('GAS', start + TimeDelta('1d'))


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
