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

Quant DSL is domain specific language for quantitative analytics in finance and trading.

At the heart of Quant DSL is a set of elements - *Settlement*, *Fixing*, *Choice*, *Market* - which encapsulate 
maths used in finance and trading. The elements of the language can be freely composed into expressions
of value. User defined functions generate extensive dependency graphs that effectively model and evaluate exotic
derivatives.

The syntax of Quant DSL expressions has been
[formally defined](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf),
the semantic model is supported with mathematical proofs.

## Implementation

This package is an implementation of the Quant DSL in Python. 

In addition to the Quant DSL expressions, function definition
statements `def` are supported, to allow expressions to be generated
concisely. The `import` statement is also supported, to allow Quant DSL function 
definitions to be used and maintained as normal Python code.

The work of a quantitative analyst includes: modelling the value of a derivative;
calibrating a stochastic process for the underlying prices; simulating future prices
of the underlyings; and evaluating of the model of the derivative against the simulation.

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

### Numerical expressions

```python
results = calc("0")
assert results.fair_value == 0
```

```python
results = calc("5")
assert results.fair_value == 5
```

```python
results = calc("-5")
assert results.fair_value == -5
```

```python
results = calc("5 + 2")
assert results.fair_value == 7
```

```python
results = calc("5 - 2")
assert results.fair_value == 3
```

```python
results = calc("5 * 2")
assert results.fair_value == 10
```

```python
results = calc("5 / 2")
assert results.fair_value == 2.5
```

```python
results = calc("5 // 2")
assert results.fair_value == 2
```

```python
results = calc("5 ** 2")
assert results.fair_value == 25
```

```python
results = calc("5 % 2")
assert results.fair_value == 1
```

```python
results = calc("Max(5, 2)")
assert results.fair_value == 5
```

```python
results = calc("Min(5, 2)")
assert results.fair_value == 2
```


### Boolean expressions

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

### Date and time expressions

The `Date` element can be used to indicate a date-time value. `TimeDelta` elements can be added and subtracted 
from `Date` elements. `Date` elements can be compared, but `TimeDelta` elements cannot be compared.

```python
import datetime
from dateutil.relativedelta import relativedelta
from quantdsl.exceptions import DslCompareArgsError


results = calc("Date('2011-1-1')")
assert results.fair_value == datetime.datetime(2011, 1, 1)
```

```python
results = calc("Date('2011-1-10') - Date('2011-1-1')")
assert results.fair_value == relativedelta(days=9)
```

```python
results = calc("Date('2012-1-1') - Date('2011-1-1')")
assert results.fair_value == relativedelta(years=1)
```

```python
results = calc("Date('2012-1-1') + TimeDelta('1y') == Date('2013-1-1')")
assert results.fair_value == True
```

```python
results = calc("Date('2011-1-1') + TimeDelta('1y') == Date('2012-1-1')")
assert results.fair_value == True
```

```python
results = calc("Date('2012-1-1') + 366 * TimeDelta('1d') == Date('2013-1-1')")
assert results.fair_value == True
```

```python
results = calc("Date('2011-1-1') + 365 * TimeDelta('1d') == Date('2012-1-1')")
assert results.fair_value == True
```

```python
results = calc("Date('2012-1-1') + 12 * TimeDelta('1m') == Date('2013-1-1')")
assert results.fair_value == True
```

```python
results = calc("Date('2011-1-1') + 12 * TimeDelta('1m') == Date('2012-1-1')")
assert results.fair_value == True
```

```python
results = calc("Date('2011-1-1') + 5 * TimeDelta('1d') < Date('2011-1-10')")
assert results.fair_value == True
```

```python
results = calc("Date('2011-1-1') + 10 * TimeDelta('1d') > Date('2011-1-10')")
assert results.fair_value == True
```

```python
try:
    calc("365 * TimeDelta('1d') == TimeDelta('1y')")
except DslCompareArgsError:
    pass
```

### Settlement

The `Settlement` element discounts the value of the included expression. Discounting is a function of the 
`interest_rate` and the duration in time between the date of the `Settlement` and the effective present time of its 
evaluation.

If the effective present time of the `Settlement` is the same as the settlement date, there is no discounting.

```python
results = calc("Settlement('2111-1-1', 10)",
    observation_date='2111-1-1',
    interest_rate=2.5,
)
assert results.fair_value == 10.0
```

Similarly, if the `interest_rate` is `0.0`, there is no discounting.

```python
results = calc("Settlement('2111-1-1', 10)",
    observation_date='2011-1-1',
    interest_rate=0,
)
assert results.fair_value == 10.0
```

However with a non-zero `interest_rate` of `2.5` percent per year, the value of `1000` in `'2111-1-1'` 
has a present value of less than `100` in `'2011-1-1'`.

```python
results = calc("Settlement('2111-1-1', 1000.0)",
    observation_date='2011-1-1',
    interest_rate=2.5,
)

# assert round(results.fair_value, 3) == 84.647, results.fair_value
assert round(results.fair_value, 3) == 82.085, results.fair_value
```

The formula used for discounting by the `Settlement` element is `e**-rt`. The `interest_rate` (above) is the 
therefore the continuously compounding risk free rate, rather than annual equivalent rate.

Todo: Support annual equivalent rate?

### Fixing

The `Fixing` element is used to condition the "effective present time" of its included expression.

For example, if a `Settlement` element is included in the `Fixing` element, then the effective present time of the 
`Settlement` element will be the date of the `Fixing`.

The expression below represents the present value in `'2111-1-1'` of the value of `10` delivered on `'2111-1-1'`, 
observed on `'2011-1-1'`.

```python
results = calc("Fixing('2111-1-1', Settlement('2111-1-1', 10))",
    observation_date='2011-1-1',
    interest_rate=7,
)

assert results.fair_value == 10
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


### Market

The `Market` element effectively estimates spot prices that could be agreed in the future.

When a `Market` element is evaluated, it returns a random variable selected from a simulation
 of market prices. Selecting an estimated price from the simulation requires the name
 of the market, and a fixing date and a delivery date - when the price would be agreed, when
 the goods would be delivered. The name of the `Market` is included in the element (e.g. `'GAS'` or `'POWER'` above).
Both the fixing date and the delivery date are set by the "effective present time"
when the element is evaluated.

Because a `Market` element depends a price simulation, it cannot be evaluated unless a price process is also 
configured. In this example, the library's one-factor multi-market Black Scholes price process 
`BlackScholesPriceProcess` is used to generate correlated geometric Brownian motions.

The calibration parameters required by `BlackScholesPriceProcess` are `market` (a list of market names), and 
`sigma`, (a list of annualised historical volatilities, expressed as a fraction of 1, rather than as a 
percentage). In these examples, at first the volatilities `sigma` of both 'GAS' and 'POWER' markets are set to zero.

Since there is more than one market, an additional parameter `rho` is required, which represents the 
correlation between the markets (a symmetric matrix expressed as a list of lists).

A forward `curve` is required to provide estimates of current prices for each market at the given 
`observation_date`. The prices in the forward curve are prices that can be agreed at the `observation_date` for 
delivery at the specified dates.

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

The price simulation has estimates of prices that could be agreed at future dates. These estimated prices are 
generated by evolving a forward curve from the `observation_date`, according to both the configured `price_process`
 and the requirements of the expression.

Requirements for the simulation (dates and markets) are derived from the expression to be evaluated, so that if the 
expression involves agreeing at a particular date a price for particular goods to be delivered at a particular 
date, then the simulation will work to provide that value.

For each price that is required, the simulation uses the delivery date to select from the forward `curve` a price 
that can be agreed at the `observation_date`, and then evolves that price through the fixing dates, according to the 
configured `price_process`.

A `Market` element evaluated at the `observation_date` will simply return the last value from the given forward
 curve for that market at the given `observation_date`.
 
```python
results = calc("Market('GAS')",
    observation_date='2011-1-1',
    price_process=price_process,
)
```

The `Market` element uses random variables from the price simulation, so the results are random variables, and we 
need to take the `mean()` to obtain a scalar value.

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

With Brownian motion provided by the price process, the random variables used to estimate the price that can be 
agreed in the future have statistical distributions with non-zero standard deviation, and so the mean value will 
only approximate to the value taken from the forward `curve`.

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


### Wait

The `Wait` element combines `Settlement` and `Fixing`, so that a single date value is used both to condition the 
effective present time of the included expression, and also the value of that expression is discounted to the 
present time effective when evaluating the `Wait` element.

```python
results = calc("Wait('2111-1-1', Market('GAS'))",
    price_process=price_process,
    observation_date='2110-1-1',
    interest_rate=1,
)
assert_almost_equal(results.fair_value.mean(), 990.0)
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

With a low interest rate, the value is dominated by the larger, later alternative.

```python

results = calc(source_code,
    observation_date='2050-1-1',
    price_process=price_process,
    interest_rate=2.5,
)
assert_almost_equal(217.390, results.fair_value.mean())
```

With a high interest rate, the value is dominated by the smaller, earlier alternative.

```python
results = calc(source_code,
    observation_date='2050-1-1',
    price_process=price_process,
    interest_rate=10,
)
assert_almost_equal(9.048, results.fair_value.mean())
```


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

In general, an option can be expressed as waiting to choose between, on one hand, the 
difference between the value of an underlying expression and a strike expression,
and, on the other hand, an alternative expression.

```python
def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))
```

A European option can then be expressed simply as an `Option` with zero alternative.

```python
def EuropeanOption(date, strike, underlying):
    Option(date, strike, underlying, 0)
```
The `AmericanOption` below is similar to the `EuropeanOption`: it is an option to exercise at a given strike price on 
the start date, with the alternative being an `AmericanOption` starting on the next date - and so on until the end 
date when the alternative is zero.

```python
def AmericanOption(start, end, strike, underlying, step):
    if start <= end:
        Option(start, strike, underlying,
            AmericanOption(start + step, end, strike, underlying, step)
        )
    else:
        0
```

A European put option can be expressed as a `EuropeanOption`, with negated underlying and strike expressions.

```python
def EuropeanPut(date, strike, underlying):
    EuropeanOption(date, -strike, -underlying)
```

A European stock option can be expressed as a `EuropeanOption`, with the `underlying` being the spot price at the 
start of the contract, discounted forward from `start`, and observed at the option expiry time `end`.

```python
def EuropeanStockOption(start, end, strike, stock):
    EuropeanOption(end, strike, Settlement(start, ForwardMarket(start, stock)))
```

The results from this function definition compare well with the well-known Black-Scholes stock option formulae across a 
range of strike prices, volatilities, and interest rates (see below).

Let's evaluate a European stock option at different strike prices, volatilities, and interest rates.

The following function `calc_european` will make it easier to evaluate the option several times.

```python
def calc_european(spot, strike, sigma, rate):
    source_code = """
def Option(date, strike, underlying, alternative):
    Wait(date, Choice(underlying - strike, alternative))

def EuropeanOption(date, strike, underlying):
    Option(date, strike, underlying, 0)
   
def StockMarket(start, stock):
    Settlement(start, ForwardMarket(start, stock)) 
    
def EuropeanStockOption(start, end, strike, stock):
    EuropeanOption(end, strike, StockMarket(start, stock))

EuropeanStockOption(Date('2011-1-1'), Date('2012-1-1'), {strike}, 'ACME')
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


```python
from scipy.stats import norm
import math

def black_scholes(spot, strike, sigma, rate):
    S = float(spot)  # spot price
    K = float(strike)  # strike price
    r = rate / 100.0  # annual risk free rate / 100
    t = 1.0  # duration (years)
    sigma = max(0.0001**2, sigma)  # annual historical volatility / 100

    sigma_squared_t = sigma ** 2.0 * t
    sigma_root_t = sigma * math.sqrt(t)
    math.log(S / K)
    d1 = (math.log(S / K) + t * r + 0.5 * sigma_squared_t) / sigma_root_t
    d2 = d1 - sigma_root_t
    Nd1 = norm(0, 1).cdf(d1)
    Nd2 = norm(0, 1).cdf(d2)
    e_to_minus_rt = math.exp(-1.0 * r * t)
    return Nd1 * S - Nd2 * K * e_to_minus_rt
    # Put option.
    # optionValue = (1-Nd2)*K*e_to_minus_rt - (1-Nd1)*S

# Check the Quant DSL valuation matches the well-known analytic formula.
spot = 10.0
for rate in [0.0, 20.0, 50.0]:
    for strike in [9.0, 10.0, 11.0]:
        for sigma in [0.0, 0.2, 0.5]:
            assert_almost_equal(
                calc_european(spot, strike, sigma, rate),
                black_scholes(spot, strike, sigma, rate)
            )
```

Todo: Use Black 76 to compare commodity option model? https://www.glynholton.com/notes/black_1976/ 


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

Todo: Discounting of forward contracts when calculating hedge positions, so the quantities will be larger with 
larger interest rates. Also the price of the hedge is the forward price at the observation time, rather than the 
spot price at the forward time?

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
