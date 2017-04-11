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

If you are working on a corporate LAN, with an HTTP proxy that requires authentication, then pip may fail to find
the Python Package Index. In this case you may need to download the distribution (and dependencies) by hand, and 
then use the path to the downloaded files instead of the package name in the `pip` command:

```
pip install ~/Downloads/quantdsl-X.X.X.tar.gz
```

### Trouble building NumPy and SciPy?

This package depends on [SciPy](https://www.scipy.org/), which depends on [NumPy](http://www.numpy.org/).

Both should be automatically installed as Python dependencies, however they depend on compilers being available on 
your system.

On GNU/Linux systems you may need to install developer tools (e.g. ```build-essential``` on Debian). If
you don't want to build the binaries, you can just install NumPy and SciPy as system packages and create a Python
virtual environment that uses system site packages (`--system-site-packages`).

Similarly, OSX users may benefit from reading [this page](https://www.scipy.org/scipylib/building/macosx.html):
install Apple’s Developer Tools, then install the Fortran compiler binary. Then you should be able
to install NumPy and SciPy using pip.

Windows users may also not be able to install NumPy and SciPy because they do not have a
compiler installed. Obtaining one that works can be frustrating. If so, one solution would
be to install the [PythonXY](https://code.google.com/p/pythonxy/wiki/Downloads?tm=2)
distribution of Python, so that you have NumPy and SciPy - other distributions are available.
Then create a virtual environment with the `--system-site-packages` option of `virtualenv`
so that NumPy and SciPy will be available in your virtual environment. If you get bogged down,
the simpler alternative is to install *Quant DSL* directly into your PythonXY installation,
using `pip install quantdsl` or `easy_install quantdsl` if `pip` is not available.

## Overview

Quant DSL is a functional programming language for modelling derivative instruments.

At the heart of Quant DSL is a set of elements - e.g. *Settlement*, *Fixing*, *Choice*, *Market* - which encapsulate 
maths used in finance and trading. The elements of the language can be freely composed into expressions
of value. User defined functions generate extensive dependency graphs that effectively model and evaluate exotic
derivatives.

The syntax of Quant DSL expressions has been
[formally defined](http://www.appropriatesoftware.org/quant/docs/quant-dsl-definition-and-proof.pdf),
the semantic model is supported with mathematical proofs. The syntax is a strict subset of the Python language
syntax. This package is an implementation of the language in Python. At this time, we are not aware of any 
other implementation of Quant DSL, in Python or in any other language.


## Usage

To create a working program, you can copy and paste the following code snippets into a single Python file. The code
snippets in this section have been tested. Please feel free to experiment by making variations.

If you are using a Python virtualenv, please check that your virtualenv is activated before installing the library
and running your program.

Let's jump in at the deep-end with a simple model of a gas-fired power station.

```python
quantdsl_module = """
PowerStation(Date('2012-01-01'), Date('2012-01-13'), Market('GAS'), Market('POWER'), Stopped(1))

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
"""
```

Construct a Quant DSL application object.

```python
from quantdsl.application.with_pythonobjects import QuantDslApplicationWithPythonObjects

app = QuantDslApplicationWithPythonObjects()
```

Compile the module into a dependency graph.

```python
dependency_graph = app.compile(quantdsl_module)
```

Calibrate from historical data. In this example, we can just register some calibration parameters.

```python
price_process_name = 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess'

calibration_params = {
    'GAS-LAST-PRICE': 10,
    'POWER-LAST-PRICE': 11,
    'GAS-ACTUAL-HISTORICAL-VOLATILITY': 30,
    'POWER-ACTUAL-HISTORICAL-VOLATILITY': 20,
    'GAS-POWER-CORRELATION': 0.4,
}

market_calibration = app.register_market_calibration(
    price_process_name,
    calibration_params
)
```

Make a simulation from the calibration.

```python
import datetime

market_simulation = app.simulate(
    dependency_graph,
    market_calibration,
    path_count=20000,
    observation_date=datetime.datetime(2011, 1, 1)
)
```

Make an evaluation using the simulation.

```python
evaluation = app.evaluate(dependency_graph, market_simulation)
```

Inspect the estimated value.

```python
estimated_value = evaluation.result_value.mean()
assert 17 < estimated_value < 18, estimated_value
```
