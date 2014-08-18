#!/usr/bin/env python
import os
import sys
from setuptools import setup

long_description = open('README').read()

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
    ],
    author='Appropriate Software Foundation',
    author_email='quant-support@appropriatesoftware.net',
    license='BSD 3Clause',
    url='https://github.com/johnbywater/quantdsl',
    description='Domain specific language for quantitative analytics in finance.',
    long_description = long_description,
    # Todo: Review these classifiers.
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
