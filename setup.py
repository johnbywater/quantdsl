#!/usr/bin/env python
import os
import sys
from setuptools import setup

long_description = open('README.md').read()

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
