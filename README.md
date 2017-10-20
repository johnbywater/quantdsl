# Quant DSL

***Domain specific language for quantitative analytics in finance and trading.***

[![Build Status](https://secure.travis-ci.org/johnbywater/quantdsl.png)](https://travis-ci.org/johnbywater/quantdsl)
[![Coverage Status](https://coveralls.io/repos/github/johnbywater/quantdsl/badge.svg)](https://coveralls.io/github/johnbywater/quantdsl)

Example of an American option expressed in Quant DSL. 

```python skip-test
AmericanOption(Date('2011-1-1'), Date('2012-1-1'), Market('Copper'), 7000, TimeDelta('1d'))

def AmericanOption(start, expiry, underlying, strike, step):
    if start <= expiry:
        Option(start, underlying, strike, AmericanOption(
            start + step, expiry, underlying, strike, step
        ))
    else:
        0

def Option(expiry, underlying, strike, alternative):
    Wait(expiry, Choice(underlying - strike, alternative))
```

## Install

Use pip to install the [latest distribution](https://pypi.python.org/pypi/quantdsl) from
the Python Package Index. You may feedback [issues on GitHub](https://github.com/johnbywater/quantdsl/issues).

```
pip install quantdsl
```

Please note, this library depends on SciPy, which can fail to install with some older versions of pip. In case of 
difficulty, please try again after upgrading pip.

```
pip install --upgrade pip
```

After successfully installing this library, the test suite should pass.

```
python -m unittest discover quantdsl
```


## Overview

Quant DSL is domain specific language for quantitative analytics in finance and trading.

At the heart of Quant DSL is a set of elements (e.g. "Settlement", "Fixing", "Market", "Choice").
The elements involve mathematical expressions commonly used within quantitative analytics, 
such as: 
[present value discounting](https://en.wikipedia.org/wiki/Present_value);
[geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion); and 
[least squares Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_option_pricing#Least_Square_Monte_Carlo).

The elements of the language can be freely composed into expressions of value. The validity of
Monte Carlo simulation for all possible expressions in the language is
[proven by induction](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf).


### Syntax

The syntax of Quant DSL expressions is defined with
[Backus–Naur Form](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form).

```
<Expression> ::= <Constant>
    | "Settlement(" <Date> "," <Expression> ")"
    | "Fixing(" <Date> "," <Expression> ")"
    | "Market(" <MarketName> ")"
    | "Wait(" <Date> "," <Expression> ")"
    | "Choice(" <Expression> "," <Expression> ")"
    | "Max(" <Expression> "," <Expression> ")"
    | <Expression> "+" <Expression>
    | <Expression> "-" <Expression>
    | <Expression> "*" <Expression>
    | <Expression> "/" <Expression>
    | "-" <Expression>

<Constant> ::= <Float> | <Integer>

<Date> ::= "'"<Year>"-"<Month>"-"<Day>"'"

<MarketName> ::= "'"<MarketName><NameChar> | <NameChar>"'"

<Year> ::= <Digit><Digit><Digit><Digit>

<Month> ::= <Digit><Digit>

<Day> ::= <Digit><Digit>

<Float> ::= <Integer>"."<Integer>

<Integer> ::= <Integer><Digit> | <Digit>

<NameChar> ::= "A" | "B" | "C" | "D" | "E" | "F" |
               "G" | "H" | "I" | "J" | "K" | "L" |
               "M" | "N" | "O" | "P" | "Q" | "R" |
               "S" | "T" | "U" | "V" | "W" | "X" |
               "Y" | "Z" | "_" | <Digit>

<Digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
```


### Semantics

In the definitions below, Quant DSL expression `v` defines a
function `[[v]](t)` from present time `t` to a random
variable in a probability space.

Constant interest rate `r` is used in discounting settlements.

For market `i`, the last price `Si` and volatility `σi` are determined
using only market price data generated before `t0`. Brownian
motion `z` is used in diffusion. In the software, this `Market` semantic is
a user option, so that other market models can be used to simulate prices,
for example models with multiple factors, mean reversion, and jump diffusion.

Choices are made using conditioning, expectation `E` is conditioned
on filtration `F` at `t`, so that choices are made only with
information at the time of the choice. In practice, information is "lost"
from included expressions by fitting the value of the expression (a
random variable in a probability space) to the underlying simulated 
prices at that time using least squares. This leads to a quantitative
difference from maximisation ("Max").

```
[[Settlement(d, x)]](t) = e ** (r * (t−d)) * [[x]](t)
```

```
[[Fixing(d, x)]](t) = [[x]](d)
```

```
[[Market(i)]](t) = Si * e ** (σi * z(t − t0)) − 0.5 * σi ** 2 * (t − t0)
```

```
[[Wait(d, x)]](t) = [[Settlement(d, Fixing(d, x))]](t)
```

```
[[Choice(x, y)]](t) = max(E[[[x]](t) | F(t)], E[[[y]](t) | F(t)])
```

```
[[Max(x, y)]](t) = max([[x]](t), [[y]](t))
```

```
[[x + y]](t) = [[x]](t) + [[y]](t)
```

```
[[x - y]](t) = [[x]](t) - [[y]](t)
```

```
[[x * y]](t) = [[x]](t) * [[y]](t)
```

```
[[x / y]](t) = [[x]](t) / [[y]](t)
```

```
[[-x]](t) = -[[x]](t)
```

Todo: Format the above as proper math symbols.

Please note, these default semantics can be substituted
using the `dsl_classes` arg of `calc()` below.


### Software

The scope of the work of a quantitative analyst involves modelling optionality,
simulating future prices, and evaluating the model against the simulation. In
implementing the syntax and semantics of Quant DSL, this software provides an
application object class `QuantDslApplication` which has methods that support
this work: `compile()`, `simulate()`, and `evaluate()`.

During *compilation* of Quant DSL source code, the application `QuantDslApplication` constructs a
graph of Quant DSL expressions. The *simulation* is generated by a calibrated price process,
according to requirements derived from the compiled dependency graph. During *evaluation*, the nodes of
the dependency graph are evaluated when they are ready to be evaluated. Intermediate results
are discarded as soon as they are no longer required, such that memory usage is mostly constant during
evaluation. For the "greeks", nodes are selectively re-evaluated with perturbed values,
according to the periods and markets they involve, avoiding unnecessary computation.

In addition to the Quant DSL expressions above, function `def`
statements are supported. User defined functions can be used
to refactor complex Quant DSL expressions, in order to model complex
optionality concisely. The `import` statement is also supported, so Quant
DSL function definitions and expressions can be developed and maintained as
Python files. Since Quant DSL syntax is a strict subset of Python, the
full power of Python IDEs can be used to write, navigate and refactor
Quant DSL source code.


## Examples

### calc()

The examples below use the library function `calc()` to evaluate Quant DSL source code. `calc()` uses the
methods of the `QuantDslApplication` mentioned above.

```python
from quantdsl.interfaces.calcandplot import calc
```

When called, the function `calc()` returns a results object, with an attribute `fair_value` that is the
simulated value, under the semantics, of the given Quant DSL expression.

```python
results = calc(source_code="2 + 3")
assert results.fair_value == 5
```

The function `calc()` has many optional arguments that can be used to control the evaluation of an expression. 

**`source_code`**

The Quant DSL module that will be evaluated. It must contain one expression, and may
contain many function definitions. It may also contain import statements that import
functions from Quant DSL modules saved as normal Python files.

**`observation_date`**

Sets the time `t0` in the semantics from when the forward curve is evolved by the price process.
Also conditions the effective present time `t` of the outermost element in 
an evaluation.

**`interest_rate`**

The continuously compounding short run risk free rate, used for discounting (default `0`).

**`price_process`**

The name and configuration parameters for the price process that will estimate
(by simulation) prices that could be agreed in the future.

**`periodisation`**

Delta granularity can be set with the `periodisation` arg of `calc()` e.g. `'daily'` or `'monthly'`.

Todo: Rename the arg to `delta_granularity`.

**`is_double_sided_deltas`**

Evaluate with either single- or double-sided deltas (default `True`).

**`path_count`**

Determines the accuracy of the simulated random variables (default `20000`).

**`perturbation_factor`**

Used to calculate "greeks". If the `path_count` is larger, a smaller
perturbation factor may give better result (default `0.01`).

**`max_dependency_graph_size`**

Sets the limit on the maximum number of nodes the can be compiled from Quant DSL source with the
`max_dependency_graph_size` arg of `calc()`.

**`timeout`**

You can set a calculation to `timeout` after a given number of seconds.

**`dsl_classes`**

Custom DSL classes can be passed in using the `dsl_classes` argument of `calc()`.

**`is_verbose`**

Setting `is_verbose` will cause progress of a calculation to
be printed to standard output.

See also `calc_and_print()` which can format and print
results. And `calc_print_plot()` which plots results using matplotlib.


### Settlement

The `Settlement` element discounts the value of the included `Expression` from its given `Date` to the effective 
present time `t` when the element is evaluated.

```
<Settlement> ::= "Settlement(" <Date> ", " <Expression> ")"
```

```
[[Settlement(d, x)]](t) = e ** (r * (t−d)) * [[x]](t)
```

For example, with a continuously compounding `interest_rate` of `2.5` percent per year, the value `10` settled on 
`'2111-1-1'` has a present value of `82.08` on `'2011-1-1'`.

```python
results = calc("Settlement('2111-1-1', 1000)",
    observation_date='2011-1-1',
    interest_rate=2.5,
)

assert round(results.fair_value, 2) == 82.08, results.fair_value
```

Similarly, the value of `82.085` settled in `'2011-1-1'` has a present value of `1000.00` on `'2111-1-1'` 
.

```python
results = calc("Settlement('2011-1-1', 82.085)",
    observation_date='2111-1-1',
    interest_rate=2.5,
)

assert round(results.fair_value, 2) == 1000.00, results.fair_value
```

Discounting is a function of the `interest_rate` and the duration in time between the date of the `Settlement` 
element and the effective present time `t` of its evaluation. The formula used for discounting by the `Settlement` 
element is `e**-rt`. The `interest_rate` is the therefore the continuously compounding risk free rate (not the 
annual equivalent rate).


### Fixing

The `Fixing` element simply conditions the effective present time of its included `Expression` with its given `Date`.

```
<Fixing> ::= "Fixing(" <Date> "," <Expression> ")"
```

```
[[Fixing(d, x)]](t) = [[x]](d)
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


### Market

The `Market` element effectively estimates prices that could be agreed in the future.

```
<Market> ::= "Market(" <MarketName> ")"
```

```
[[Market(i)]](t) = Si * e ** (σi * z(t − t0)) − 0.5 * σi ** 2 * (t − t0)
```

When a `Market` element is evaluated, it returns a random variable selected from a simulation
 of market prices.

Selecting an estimated price from the simulation requires the name of the market,
a *fixing date* (when the price would be agreed), and a *delivery date* (when the goods would be delivered).
 
The name of the `Market` is included in the element (e.g. `'GAS'` or `'POWER'`). Both the fixing date and
the delivery date are determined by the effective present time when the element is evaluated.


#### Simulation
 
The price simulation is generated by a price process. In this example, the library's one-factor
multi-market Black Scholes price process `BlackScholesPriceProcess` is used to generate correlated
geometric Brownian motions.

The calibration parameters required by `BlackScholesPriceProcess` are `market` (a list of market names), and 
`sigma`, (a list of annualised historical volatilities, expressed as a fraction of 1, rather than as a 
percentage).

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

When a simulation generated by this price process involves two or more markets, an additional
parameter `rho` is  required, which represents the correlation between the markets (a symmetric
matrix expressed as a list of lists).


#### Forward curve

A forward `curve` is required by the price process to provide estimates of current prices for each market at the given 
`observation_date`. The prices in the forward curve are prices that can be agreed at the `observation_date` for 
delivery at the specified dates. These prices are then diffused according to the risk-neutral dynamics of the
price process.

Requirements for the simulation (dates and markets) are derived from the expression to be evaluated. If the 
expression involves observing at a future date the price for particular goods to be delivered at a particular 
date, then the simulation will provide a random variable which simulates that observation.

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

Evaluating at a later observation date will return the later value from the forward curve.
 
```python
results = calc("Market('GAS')",
    observation_date='2111-1-1',
    price_process=price_process,
)
assert results.fair_value.mean() == 1000
```

#### Stochastic evolution

In the examples so far, there has been no difference between the fixing date of the `Market` element and 
the `observation_date` of the evaluation. Therefore, there is no stochastic evolution of the forward curve, and the 
standard deviation of the result value is zero.

```python
assert results.fair_value.std() == 0.0
```

If a `Market` element is included within a `Fixing` element, the effective present time will be different from
the observation date, so the fixing date used to select a price from the simulation will be different from the
observation date. The simulated price will be taken from the forward curve, but will also be subjected to
stochastic evolution.

With Brownian motion provided by the `BlackScholesPriceProcess`, the random variable used to estimate a 
price observed in the future has a statistical distribution with non-zero standard deviation.

```python
results = calc("Fixing('2051-1-1', Market('GAS'))",
    observation_date='2011-1-1',
    price_process=price_process,
)
assert results.fair_value.std() > 0.0
```

#### Accuracy

The number of samples from the distribution in the simulated random variable defaults to `20000`.

```python
assert len(results.fair_value) == 20000
```
The number can be adjusted by setting `path_count`. The accuracy of results 
can be doubled by increasing the path count by a factor of four.

```python
results = calc("Market('GAS')",
    observation_date='2011-1-1',
    price_process=price_process,
    path_count=80000,
)
assert len(results.fair_value) == 80000
```

#### Different fixing and delivery dates

The `ForwardMarket` element can be used to specify a delivery date that is different from the fixing date (when the 
price for that delivery would be agreed). This can be used to model trading in forward markets, and also
to express the value of an index at maturity (see below).


### Wait

The `Wait` element combines `Settlement` and `Fixing`, so that its `Date` is used both to condition the 
effective present time of the included `Expression`, and also the value of that expression is discounted to the 
effective present time when evaluating the `Wait` element.

```
<Wait> ::= "Wait(" <Date> "," <Expression> ")"
```

```
[[Wait(d, x)]](t) = [[Settlement(d, Fixing(d, x))]](t)
```

For example, the present value at the `observation_date` of `'2011-1-1'` of one unit of `'GAS'` delivered on 
`'2111-1-1'` is approximately `82.18`. The `Wait` element sets the delivery date of the `Market` element,
 which is used to pick the value `1000` from the forward curve. The `Wait` element also sets the fixing date
 of the `Market` element, which is used to control the stochastic evolution of the forward value.  The evolved
 value is then discounted by the `Wait` element from `2111` to the observation date `2011`.


```python
import scipy
# Setting random seed makes test results repeatable.
scipy.random.seed(1234)

results = calc("Wait('2111-1-1', Market('GAS'))",
    price_process=price_process,
    observation_date='2011-1-1',
    interest_rate=2.5,
)

assert round(results.fair_value.mean(), 2) == 82.18
assert round(results.fair_value.std(), 2) == 16.46
```

### Choice

The `Choice` element uses the least-squares Monte Carlo approach (Longstaff
Schwartz, 1998) to compare the conditional expected value of each alternative `Expression`.

```
<Choice> ::= "Choice(" <Expression> "," <Expression> ")"
```

```
[[Choice(x, y)]](t) = max(E[[[x]](t) | F(t)], E[[[y]](t) | F(t)])
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
assert round(results.fair_value.mean(), 2) == 82.06
```

When the `Choice` element is evaluated, the value of each alternative is
regressed as a random variable by least squares to the simulated value of the underlyings
at the effective present time of the choice. The choice of alternative on each path is then
made using the regressed value (the "conditional expected value") but the chosen
value on each path is taken from the unregressed value of the chosen alternative for that path
(the "expected continuation value"). The result is a new simulated value that combines the
expected continuation value of the alternatives, according to information in the simulation
at the time of the choice. This conditioning gives a quantitatively different result from simple
maximisation of the alternative expected continuation values (`Max`). 

### Function definitions

Quant DSL source code can include function definitions. Expressions can involve calls to functions.

When evaluating an expression that involves a call to a function definition, the call to the 
function definition is effectively replaced with the expression returned by the function definition,
so that a larger expression is formed. Hence, the body of a function can have only one statement.

The call args of the function definition can be used as names in the function definition's
expressions. The call arg values will be substituted for those names in the expression when
the expression is returned by the function.

```python
results = calc("""
def Function(a):
    2 * a

Function(10)
""")

assert results.fair_value == 20
```   

Although the function body can have only one statement, that statement can be an if-else block.
The call args of the function definition can be used in an if-else block, to select different
expressions according to the value of the function call arguments. This is effectively implements
a "case branch".

Each function call becomes a node on a dependency graph. For efficiency, each call is cached, so if a 
function is called many times with the same argument values (and at the same effective present time),
the function is only evaluated once, and the result reused. This allows branched
calculations to recombine efficiently. For example, the following Fibonacci function definition will
evaluate in linear time (proportional to `n`).

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
assert round(results.fair_value, 2) == 114.59
```
 
Instead the expression could be refactored with a function definition.
 
```python
source_code = """
def Settlements(start, end, installment):
    if start <= end:
        Settlement(start, installment) + Settlements(start + TimeDelta('1m'), end, installment)
    else:
        0
        
Settlements(Date('2011-1-1'), Date('2011-12-1'), 10)
"""
results = calc(source_code,
    observation_date='2011-1-1',
    interest_rate=10,
)
assert round(results.fair_value, 2) == 114.67
```


### European and American options

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
An American option can be expressed as an `Option` to exercise at a given `strike` price on 
the `start` date, with the alternative being another `AmericanOption` starting on the next date - and so on until the 
`expiry` date, when the `alternative` becomes zero.

```python
def AmericanOption(start, expiry, strike, underlying, step):
    if start <= expiry:
        Option(
            start, strike, underlying, AmericanOption(
                start + step, expiry, strike, underlying, step
            )
        )
    else:
        0
```

A European put option can be expressed as a `EuropeanOption`, with negated underlying and strike expressions.

```python
def EuropeanPut(expiry, strike, underlying):
    EuropeanOption(expiry, -strike, -underlying)
```

A European stock option can be expressed as a `EuropeanOption`, with the `underlying` being the index at
maturity. The `IndexAtMaturity` has the spot price at the observation date
(using `ForwardMarket`) observed at a time in the future, discounted forward from the observation date (due
to the `Settlement`) to a time in the future. This "time in the future" is the effective
present time set by the `Wait` element of the `Option` definition above.

```python
def EuropeanStockOption(expiry, strike, stock):
    EuropeanOption(expiry, strike, IndexAtMaturity(stock))
```

The expression `IndexAtMaturity` is passed into the `Option` definition and, due to the `Wait` 
element in that definition, is evaluated with `expiry` as the present time. Therefore the spot 
price is discounted forward to the `expiry`, and the fixing date of the `ForwardMarket` is `expiry`, so that the
spot price is subjected to stochastic evolution as well as interest rates.

```python

def IndexAtMaturity(stock):
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
    EuropeanOption(expiry, strike, IndexAtMaturity(stock))

def IndexAtMaturity(stock):
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
    return round(results.fair_value.mean(), 2)
```

If the strike price of a European option is the same as the price of the underlying, without any volatility (`sigma` 
is `0`) the value is zero.

```python
assert calc_european(spot=10, strike=10, sigma=0, rate=0) == 0.0
```

If the strike price is less than the underlying, without any volatility, the value is the difference between the 
strike and the underlying.

```python
assert calc_european(spot=10, strike=8, sigma=0, rate=0) == 2.0
```

If the strike price is greater than the underlying, without any volatility, the value is zero.

```python
assert calc_european(spot=10, strike=12, sigma=0, rate=0) == 0.0
```

If the strike price is the same as the underlying, with some volatility in the price of the underlying, there
 is some value in the option.

```python
assert calc_european(spot=10, strike=10, sigma=0.9, rate=0) == 3.42
```

If the strike price is less than the underlying, with some volatility in the price of the underlying (`sigma`) there
 is more value in the option than without volatility.

```python
assert calc_european(spot=10, strike=8, sigma=0.9, rate=0) == 4.23
```

If the strike price is greater than the underlying, with some volatility in the price of the underlying (`sigma`) there
 is still a little bit of value in the option.

```python
assert calc_european(spot=10, strike=12, sigma=0.9, rate=0) == 2.90
```

These results compare well with results from the Black-Scholes analytic formula for European stock options.


### Gas storage

An evaluation of a gas storage facility. The value of the
gas storage facility follows from the difference between the price
when gas is injected and the price when gas is withdrawn.

The Quant DSL source code below models a gas storage facility as
a lattice of choices to inject or withdraw a quantity of gas.
If the facility is full, injecting gas is not an option. Similarly,
if the facility is empty, withdrawing gas is not an option.

```python
gas_storage = """
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


GasStorage(Date('2011-4-1'), Date('2012-4-1'), Market('GAS'), Empty(), Empty(), Full(), TimeDelta('1m'))
"""
```

This example uses a forward curve that has seasonal variation (prices are high in winter and low in 
summer).

```python
gas = {
    'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
    'market': ['GAS'],
    'sigma': [0.3],
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
```

This example uses the library function `calc_print()` to calculate and then print results.

```python
from quantdsl.interfaces.calcandplot import calc_print
```

Because the `periodisation` argument is set to `'monthly'`, the deltas for each market in each month will be 
calculated, and estimated risk neutral hedge positions will be printed for each market in each period, along
with the overall fair value.

```python
results = calc_print(
    source_code=gas_storage,
    observation_date='2011-1-1',
    interest_rate=2.5,
    periodisation='monthly',
    price_process=gas,
    verbose=True,
)

assert round(results.fair_value.mean(), 2) == 20.78, results.fair_value.mean()
```

Below are the results printed by `calc_and_print()`, showing
deltas for each month for each market, and the fair value.

```
Compiled 92 nodes 
Compilation in 0.463s
Simulation in 0.061s
Starting 844 node evaluations, please wait...
844/844 100.00% complete 99.68 eval/s running 9s eta 0s
Evaluation in 8.468s


2011-04-01 GAS
Price:     9.00
Delta:    -0.99
Hedge:     1.00 ± 0.00
Cash:     -8.94 ± 0.03

2011-05-01 GAS
Price:     7.50
Delta:    -0.99
Hedge:     1.00 ± 0.00
Cash:     -7.44 ± 0.03

2011-06-01 GAS
Price:     7.00
Delta:    -0.99
Hedge:     1.00 ± 0.00
Cash:     -6.93 ± 0.03

2011-07-01 GAS
Price:     6.49
Delta:    -0.99
Hedge:     1.00 ± 0.00
Cash:     -6.41 ± 0.03

2011-08-01 GAS
Price:     7.49
Delta:    -0.99
Hedge:     1.00 ± 0.00
Cash:     -7.38 ± 0.04

2011-09-01 GAS
Price:     8.49
Delta:    -0.98
Hedge:     1.00 ± 0.00
Cash:     -8.35 ± 0.04

2011-10-01 GAS
Price:     9.97
Delta:     0.98
Hedge:    -1.00 ± 0.00
Cash:      9.79 ± 0.05

2011-11-01 GAS
Price:    11.47
Delta:     0.98
Hedge:    -1.00 ± 0.00
Cash:     11.23 ± 0.07

2011-12-01 GAS
Price:    11.98
Delta:     0.98
Hedge:    -1.00 ± 0.00
Cash:     11.70 ± 0.07

2012-01-01 GAS
Price:    13.46
Delta:     0.98
Hedge:    -1.00 ± 0.00
Cash:     13.13 ± 0.09

2012-02-01 GAS
Price:    10.98
Delta:     0.97
Hedge:    -1.00 ± 0.00
Cash:     10.68 ± 0.07

2012-03-01 GAS
Price:     9.99
Delta:     0.97
Hedge:    -1.00 ± 0.00
Cash:      9.70 ± 0.07

Net hedge GAS:       0.00 ± 0.00
Net hedge cash:     20.78 ± 0.28

Fair value: 20.78 ± 0.28
```

The value obtained is the extrinsic value. The intrinsic value can be 
obtained by setting the volatility `sigma` to `0`, and can be evaluated
with `path_count` of `1`.

The recommended hedge positions suggest injecting gas when
the price is low, and withdrawing when the price is high.

An alternative to `calc_print()` is the function in the same module
`calc_print_plot()` which will also plot the prices, positions, and
cash. You will need to install matplotlib to use `calc_print_plot()`.


### Gas fired power station

An evaluation of a gas fired power station. The value of a gas fired power
station follows from selling generated power whilst paying for gas. 

The Quant DSL source code below models a gas fired power station as
a lattice of choices whether or not to run the power station. The efficiency
of generation is modelled to be lower if the power station has been stopped.

Dispatch decisions are made daily, with gas and power traded one day ahead.


```python
power_plant = """
from quantdsl.semantics import Choice, Market, TimeDelta, Wait, inline, Min


def PowerPlant(start, end, temp):
    if (start < end):
        Wait(start, Choice(
            PowerPlant(Tomorrow(start), end, Hot()) + ProfitFromRunning(start, temp),
            PowerPlant(Tomorrow(start), end, Stopped(temp))
        ))
    else:
        return 0

@inline
def Power(start):
    DayAhead(start, 'POWER')

@inline
def Gas(start):
    DayAhead(start, 'GAS')

@inline
def DayAhead(start, name):
    ForwardMarket(Tomorrow(start), name)
    
@inline
def Tomorrow(start):
    start + TimeDelta('1d')

@inline
def ProfitFromRunning(start, temp):
    if temp == Cold():
        return 0.3 * Power(start) - Gas(start)
    elif temp == Warm():
        return 0.6 * Power(start) - Gas(start)
    else:
        return Power(start) - Gas(start)

@inline
def Stopped(temp):
    if temp == Hot():
        Warm()
    else:
        Cold()

@inline
def Hot():
    2

@inline
def Warm():
    1

@inline
def Cold():
    0
    

PowerPlant(Date('2012-1-1'), Date('2012-1-5'), Cold())
"""
```

The prices process is calibrated with two correlated markets.

```python
gas_and_power = {
    'name': 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess',
    'market': ['GAS', 'POWER'],
    'sigma': [0.3, 0.3],
    'rho': [[1.0, 0.8], [0.8, 1.0]],
    'curve': {
            'GAS': [
                ('2012-1-1', 11.0),
                ('2012-1-2', 11.0),
                ('2012-1-3', 1.0),
                ('2012-1-4', 1.0),
                ('2012-1-5', 11.0),
            ],
            'POWER': [
                ('2012-1-1', 1.0),
                ('2012-1-2', 1.0),
                ('2012-1-3', 11.0),
                ('2012-1-4', 11.0),
                ('2012-1-5', 11.0),
            ]
        }
}
```

Because the `periodisation` is set to `'daily'`, the deltas for each market in each day will be 
calculated, and estimated risk neutral hedge positions will be printed for each market in each period, along
with the overall fair value.

```python
results = calc_print(
    source_code=power_plant,
    observation_date='2011-1-1',
    interest_rate=2.5,
    periodisation='daily',
    price_process=gas_and_power,
    verbose=True
)

assert round(results.fair_value.mean(), 2) == 12.82, results.fair_value.mean()
```

These are the results printed by `calc_and_print()`, showing
monthly deltas for each of the two markets.

```
Compiled 16 nodes 
Compilation in 0.038s
Simulation in 0.043s
Starting 112 node evaluations, please wait...
112/112 100.00% complete 115.57 eval/s running 1s eta 0s
Evaluation in 0.970s


2012-01-02 GAS
Price:    10.98
Delta:    -0.05
Hedge:     0.05 ± 0.00
Cash:     -0.41 ± 0.04

2012-01-02 POWER
Price:     1.00
Delta:     0.02
Hedge:    -0.02 ± 0.00
Cash:      0.02 ± 0.00

2012-01-03 GAS
Price:     1.00
Delta:    -0.98
Hedge:     1.00 ± 0.00
Cash:     -0.97 ± 0.01

2012-01-03 POWER
Price:    10.98
Delta:     0.32
Hedge:    -0.33 ± 0.00
Cash:      3.64 ± 0.05

2012-01-04 GAS
Price:     1.00
Delta:    -0.98
Hedge:     1.00 ± 0.00
Cash:     -0.97 ± 0.01

2012-01-04 POWER
Price:    10.98
Delta:     0.98
Hedge:    -1.00 ± 0.00
Cash:     10.71 ± 0.07

2012-01-05 GAS
Price:    10.98
Delta:    -0.49
Hedge:     0.50 ± 0.00
Cash:     -4.95 ± 0.11

2012-01-05 POWER
Price:    10.98
Delta:     0.49
Hedge:    -0.50 ± 0.00
Cash:      5.76 ± 0.13

Net hedge GAS:       2.55 ± 0.01
Net hedge POWER:    -1.85 ± 0.01
Net hedge cash:     12.82 ± 0.10

Fair value: 12.82 ± 0.10
```

The recommended hedge positions suggest running the plant when
the price of power is high and the price of gas is low.

## Jupyter notebooks

It's easy to use Quant DSL in a Jupyter notebook. See [valuation_template.ipynb](./valuation_template.ipynb).

Jupyter notebooks can be executed on a Jupyter hub.


## Library

There is a small collection of Quant DSL modules in a library under `quantdsl.lib`.
Putting Quant DSL source code in dedicated Python files makes it much easier to use
a Python IDE to develop and maintain Quant DSL function definitions.


## Acknowledgments

The Quant DSL language was partly inspired by the paper
*[Composing contracts: an adventure in financial engineering (functional pearl)](
http://research.microsoft.com/en-us/um/people/simonpj/Papers/financial-contracts/contracts-icfp.htm
)* by Simon Peyton Jones and others. The idea of orchestrating evaluations with a dependency graph,
to help with parallel and distributed execution, was inspired by a [talk about dependency graphs by
Kirat Singh](https://www.youtube.com/watch?v=lTOP_shhVBQ). This implementation uses NumPy and
SciPy packages, and the Python ast ("Absract Syntax Trees") module. We have also been encourged by members of the 
[London Financial Python User Group](https://www.google.co.uk/search?q=London+Financial+Python+User+Group),
where Quant DSL expression syntax and semantics were first presented.
