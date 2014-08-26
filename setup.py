#!/usr/bin/env python
import os
import sys
from setuptools import setup

long_description = """
Quant DSL is a hybrid functional programming language for modelling derivative financial instruments.

The core of Quant DSL is a set of primitive elements (such as "Wait", "Choice", "Market") that encapsulate common mathematical machinery used in finanace and trading (e.g. time value of money calculations, the least-squares Monte Carlo approach, models of market dynamics) and which can be composed into executable expressions of value.

User defined functions can be used to generate complex graphs of primitive expressions which can be evaluated in parallel. The syntax of Quant DSL expressions have been formally defined, and the semantics are supported with mathematical proofs.

This package is an implementation in Python of the Quant DSL syntax and semantics. 
"""

from quantdsl import __version__

setup(
    name='quantdsl',
    version=__version__,
    packages=['quantdsl'],
    # just use auto-include and specify special items in MANIFEST.in
    zip_safe = False,
    install_requires = [
        'scipy',
        'numpy',
        'argh',
        'requests',
    ],
    scripts = [
        os.path.join('scripts', 'quantdsl'),
    ],
    extras_require = {
        'test':  ["Mock"],
    },
    author='John Bywater',
    author_email='john.bywater@appropriatesoftware.net',
    license='BSD',
    url='https://github.com/johnbywater/quantdsl',
    description='Domain specific language for quantitative analytics in finance.',
    long_description = long_description,
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Office/Business :: Financial :: Spreadsheet',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
   ],
)
